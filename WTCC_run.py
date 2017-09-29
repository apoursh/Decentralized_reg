# Coder: Armin Pourshafeie
# runs a small GWAS in a centralized and "decentralized" fashion on a smallish dataset
# Here Decentralized only means that the data is in different locations and does 
# not get copied into a centralized locationa. There is no actual massage being 
# passed over the internet but, that's the idea. 


import os, sys
import argparse
import process_chiamo
import analytics
import logging
import datetime
import pdb
import h5py
import numpy as np
from functools import partial


def parse_arguments():
  parser = argparse.ArgumentParser(description="""
    Run a comparison between centralized and decentralized GWAS""")
  parser.add_argument('--data_to_use', dest='data_address',
      default='Dsets_to_include.txt')
  parser.add_argument('--log_dir', dest='log_dir', default='logs/')
  
  args = parser.parse_args([])
  return args

def prepare_logger(log_dir):
  date = datetime.datetime.now()
  date = [date.year, date.month, date.day, date.hour,
      date.minute]
  date = [str(item) for item in date]
  logname = os.path.join(log_dir, '_'.join(date))
  logging.basicConfig(filename=logname, level=logging.DEBUG)
  # also print to screen
  root = logging.getLogger()
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  root.addHandler(ch)

def dset_split(to_split, num_splits, n_tot, split_prefix):
  """Distributes the rows of h5py dataset at to_split into num_splits groups 
  of random size
  This function copies by shamelessly iterating over everything so it can be 
  very slow"""
  while (True):
    num = np.random.poisson(n_tot / float(num_splits), num_splits - 1)
    np.append(num, n_tot - np.sum(num))
    if all(num > 0):
      break


  def group_copy(name, node, rows, fp):
    dtype = node.dtype
    value = node[...]
    fp.require_dataset(name, data=value[rows], shape=(len(rows),), dtype=dtype)
    
  with h5py.File(to_split, 'r') as to_split_fp:
    for i, number in enumerate(num):
      split_name = split_prefix + str(i) + '.h5py'
      logging.info("-Constructing: " + split_name)
      chosen_rows = np.random.random_integers(0, n_tot-1, number)
      with h5py.File(split_name, 'w') as copy_to_fp: 
        for key in to_split_fp.keys():
          dset_to_copy = to_split_fp[key]
          dset_to_copyto = copy_to_fp.require_group(key)
          if key != 'meta':
            copier = partial(group_copy, rows=chosen_rows, fp=dset_to_copyto)
            dset_to_copy.visititems(copier)
          else:
            group_copy("meta/Status", dset_to_copy['Status'], chosen_rows,
                dset_to_copyto)


      


def run_experiment(shuffled=True):
  args = parse_arguments()
  prepare_logger(args.log_dir)
  store_name = 'hdfstore.h5py'
  if not os.path.isfile(store_name):
    process_chiamo.read_datasets(args.data_address, shuffled, store_name=store_name)
  else: 
    logging.info("-SNP file has already been made")
  # Split the dataset into parts for decentralized analysis
  dset_split(store_name, 5, 2000, 'split_data/DO')

  # Continue with centralized analysis
  centralized_DO = analytics.DO(store_name=store_name)
  centralized_DO.compute_local_AF()
  centralized_DO.normalize()
#  centralized_DO.run_logistic(num_cores=2)
  centralized_DO.pruning(0.5, 0.1, 50) # add this to local_PCA
  centralized_DO.local_PCA()
  
  # Continue with decentralized analysis


if __name__=='__main__':
  run_experiment()
