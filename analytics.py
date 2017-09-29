# Armin Pourshafeie
# Working: 
#    Normalization for centralized
#TODO: 
#     regression setup + pvalues
#     compute PCA 
#     include PCA in regression
#   Everything for decentralized
#
#


# write a generator that takes the chromosome and spits out data. do the regression in parallel 
# get data and add it to attrs later 

# Running the gwas
import os
import logging
import numpy as np
import gzip, h5py
import pdb
import re 
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import dill
from sklearn.decomposition import PCA 



class DO(object):
  """This object represents each data owner. It can compute things in a 
  centralized manner on it's own data, or if it has a center it can 
  communicate with the center"""
  def __init__(self, store_name, center=None):
    self.store_name = store_name
    self.center = center
    self.PCA_U, self.PCA_Sigma, self.PCA_V = None, None, None #perhaps write to hdf
    with h5py.File(self.store_name) as store:
      self.has_local_AF = store.attrs['has_local_AF']
      self.normalized = store.attrs['normalized']


  def run_logistic(self, PCA_correct=False, num_cores=1, **kwd):
    def _logistic_regression(val):
      val = add_constant(val)
      fit = smf.Logit(status, val).fit(disp=0)
      pval = fit.pvalues
      betas = fit.params
      return pval, betas

    def _snp_generator(f, chrom):
      group = f[chrom]
      for snp in group:
        dset = group[snp]
        yield dset.value

    with Pool(num_cores) as pool:
      with h5py.File(self.store_name, 'a', libver='latest') as writefp, \
          h5py.File(self.store_name, 'r', swmr=True, libver='latest') as f:
        status = f['meta/Status'].value
        status[100:400] = 1 #TODO get rid of this line
        for chrom in f.keys():
          if chrom == 'meta':
            continue
          generator = _snp_generator(f, chrom)
          logging.info("--Regression for Chromosome {}".format(chrom))
          vals = pool.map(_logistic_regression, generator, chunksize=2)
          writefp['meta'].require_dataset('Centralized_' + chrom, (len(vals),),
              dtype=np.float32)

  def AF_filter(self, threshold, chrom_dset):
    return [i for i in chrom_dset if chrom_dset[i].attrs['AF'
      ]>=threshold and chrom_dset[i].attrs['AF'] <= 1-threshold]

  def pruning(self, threshold, Af_threshold, win_sz, step_sz=None):
    """Threshold is for rsquared and win_sz is in number of snps"""
    def pruner(dset, threshold, window):
      to_delete = set()
      for i, snp in enumerate(window):
        if snp in to_delete:
          continue
        else :
          snpi = dset[str(snp)].value
          for j in range(i+1, len(window)):
            if window[j] in to_delete:
              continue
            elif np.cov(snpi, dset[str(window[j])].value)[0,1]**2 > threshold: # use only with normalzied data
              to_delete.add(window[j])
      return to_delete
    
    if step_sz == None:
      step_sz = int(win_sz/4)
    with h5py.File(self.store_name, 'a') as readfp:
      for chrom in readfp.keys():
        if chrom == 'meta':
          continue
        logging.info('--Pruning chrom: ' + chrom)
        dset = readfp[chrom]
        #snps = np.sort(np.array(dset.keys()).astype(int))
        snps = np.sort(np.array(self.AF_filter(Af_threshold, dset))).astype(int)
        win_sz = min(snps.shape[0], win_sz)
        finished, winstart, winend = False, 0, win_sz
        highLD = set()
        while not finished:
          winend = winstart + win_sz
          if winend >= len(snps) - 1:
            finished = True 
            winend = len(snps) - 1
          window = snps[winstart:winend]
          window = np.sort(np.array(list(set(window) - highLD)))
          to_delete = pruner(dset, threshold, window)
          highLD = highLD.union(to_delete)
          winstart += step_sz
        toKeep = set(snps) - highLD
        for snp in toKeep: 
          dset[str(snp)].attrs['local_prune_selected'] = True

  def local_PCA(self, n_components=30):
    with h5py.File(self.store_name, 'r') as store:
      to_PCA = np.array([ store[dset][i].value for dset in store for i in
        store[dset] if 'local_prune_selected' in store[dset][i].attrs])
    pca = PCA(n_components=n_components)
    pca.fit(to_PCA)
    with h5py.File(self.store_name) as store:
      dset = store['meta']
      pca_u = dset.require_dataset('pca_u', data=pca.components_, 
          shape=(pca.components_.shape), dtype=np.float32)
      pca_u.attrs['sigmas'] = pca.singular_values_

  def compute_local_AF(self):
    def __compute_AF(name, node):
      if isinstance(node, h5py.Dataset):
        node.attrs['local_AF'] = np.mean(node) / 2.
        node.attrs['n']        = node.len()
        node.attrs['local_sd'] = np.std(node)
        if self.center is None: 
          node.attrs['AF']     = node.attrs['local_AF']
          node.attrs['sd']     = node.attrs['local_sd']

    logging.info("-Computing local allele frequencies")
    if self.has_local_AF:
      logging.info("--Allele frequencies have already been computed")
      return
    with h5py.File(self.store_name, 'a') as f:
      f.visititems(__compute_AF)
      self.has_local_AF = True
      f.attrs['has_local_AF'] = True

  def normalize(self):
    """Normalizes the data. mean zero, variance 1
    if a data center is associated with the DO uses the overall
    data to normalize, otherwise normalizes based on the data present
    """
    def __normalize_centralized(name, node):
      if node.attrs['sd'] != 0:
        node[:] = (node - (node.attrs['AF'] * 2)) / node.attrs['sd']

    if self.normalized:
      logging.info("-Data is already normalized")
      return
    logging.info("-Normalizing the data...")
    if not self.has_local_AF:
      self.compute_local_AF()
    if self.center is None: 
      with h5py.File(self.store_name, 'a') as f:
        for chrom in f.keys():
          if chrom != 'meta':
            logging.info('--Normalizing chrom: ' + chrom)
            f[chrom].visititems(__normalize_centralized)
        f.attrs['normalized'] = True
      self.normalized = True

    else:
      #centralized setting
      pass


  def PCA():
    """Computes PCA"""
    pass


# define a class that inherits from above for the group that has centers 

class Decentered_DO(object):
  def __init__(self):
    pass




class center(object):
  """This class implements the central hub"""
  def __init__(self, all_data, num_DOs, n):
    self.repos = []
    self.n = n
    self.numbers, num = None, None
    while (num == None or any(num <= 0)):
      num = np.random.poisson(self.n/float(num_DOs), num_DOs-1)
      num.append(self.n - np.sum(num))
    with h5py.File(all_data, 'r') as f:
      pass
  
  def cound_data(self):
    pass







if __name__=='__main__':
  print "no commands here yet. Test using WTCCC_run.py"


