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



def run_experiment(shuffled=True):
  args = parse_arguments()
  prepare_logger(args.log_dir)
  store_name = 'hdfstore.h5py'

  process_chiamo.read_datasets(args.data_address, shuffled, store_name=store_name)
  centralized_DO = analytics.DO(store_name=store_name)
  #centralized_DO.compute_local_AF()
  centralized_DO.normalize()



if __name__=='__main__':
  run_experiment()
