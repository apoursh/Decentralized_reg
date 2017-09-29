# Armin Pourshafeie

# Reads Chiamo files and parses them into a numpy datastore
import os
import logging
import numpy as np
import gzip, h5py
import pdb
import re 



def reader(fp):
  line = fp.readline()
  if len(line) > 0:
    dummy = line[:-2].split(' ')[5:]
    num_inds = len(dummy)/3 + 5

  while len(line) > 0:   # Poor mans' end of line detection
    line = line[:-2].split(' ')
    snp, rsid, pos = line[0], line[1], line[2]
    gts = [float(line[i + 1]) + float(line[i + 2]) * 2 for i in range(5, num_inds,3)]
    yield snp, rsid, pos, gts
    line =  fp.readline()


def files_to_read(loc_fname):
  logging.info("-Reading data files from: " + loc_fname)
  all_to_reads = []
  directories = []
  with open(loc_fname, 'r') as fp:
    lines = fp.readlines()
  lines = [line[:-1] for line in lines] # strip newline char
  n_tot = int(lines[0])
  logging.info("-Will analyze {} individuals".format(n_tot))
  # loop over the file and read/write the data
  for ln in lines[1:]:
    ln = ln.split(',')
    directory = ln[0]
    directories.append(directory)
    dirList = os.listdir(directory)
    toRead = sorted([(os.path.join(directory,item), ln[1])
      for item in dirList if item[-9:] == 'chiamo.gz'])
    all_to_reads.append(toRead)
    all_to_reads = [item for group in all_to_reads for item in group]
  return n_tot, all_to_reads, directories

def get_chrom(filename):
  """Given a file path tries to guess the chromsome. 
  This just looks for digits so make sure the path name doesn't have
  other digits in it"""
  basename = os.path.basename(filename)
  pattern = re.compile("[0-2][0-9]")
  m = pattern.search(basename)
  try:
    chrm = m.group(0)
  except AttributeError:
    logging.error("Couldn't find the Chromosome name :(")
    sys.exit("Failed at finding the chromosome name")
  return chrm



def read_datasets(loc_fname, shuffle, store_name='test.h5', seed=25):
  n_tot, to_read, directories= files_to_read(loc_fname)
  data_order = range(n_tot)
  if shuffle:
    np.random.seed(seed)
    np.random.shuffle(data_order)
  with h5py.File(store_name, 'w-', libver='latest', swmr=True) as store:
    store.attrs['has_local_AF'] = False
    store.attrs['normalized']   = False
    filled, dir_num = 0, 0
    for filepath, status in to_read:
      directory = directories[dir_num]
      logging.debug("--Working on: " + filepath)
      chrom = get_chrom(filepath)
      # see if there are new individuals
      if directory not in filepath:
        # Logically this should be executed once per every cohort
        filled += len(gts)
        dir_num += 1
        # we are done with this directory! write the status!
        dset = store.require_dataset('meta/Status', (n_tot,), dtype=np.int8)
        dset[to_fill,] = status
      with gzip.open(filepath, 'rb') as file_pointer:
        current_group = store.require_group(chrom)
        gen = reader(file_pointer)
        i = 0
        for snp, rsid, pos, gts in gen:
          to_fill = data_order[filled:filled+len(gts)]
          to_fill, gts = zip(*sorted(zip(to_fill, gts))) #TODO fix this so 
            # We don't have to sort every time
          dset = current_group.require_dataset(pos, (n_tot,), dtype=np.float32)
          # check to make sure ref/alts/rsids are not screwed up
          if 'rsid' in dset.attrs:
            if dset.attrs['rsid'] != rsid:
              sys.exit("rsid's don't match for chrom {}. pos {}".format(
                chrom, pos))
          else:
            dset.attrs['rsid'] = rsid
            dset.attrs['snp'] = snp
          dset[to_fill,] = gts
          i += 1
          if i > 100:
            break
        # end of generator loop
      # end of context manager for filepath
    # end of to_read loop
    dset = store.require_dataset('meta/Status', (n_tot,), dtype=np.int8)
    dset[to_fill,] = float(status)



if __name__=='__main__':
  with gzip.open('../WTCCC_data/EGAD00000000002/Oxstat_format/NBS_04_chiamo.gz', 'rb') as fp:
    gen = reader(fp)

    i = 0
    for snp, rsid, pos, item in gen:
      pdb.set_trace()
#      print item 
#      i += 1
#      print i
#      if i > 3:
#        break
#


