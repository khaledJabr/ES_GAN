from src.optimizers import OpenAIOptimizer,  NSRESOptimizer
# from src.policy import Policy
from src.gan import simple_GAN
from src.logger import Logger
from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym


# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'NSRESOptimizer' : NSRESOptimizer

}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name, dataset_dir):
    
    # Read config file 
    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())
    configuration['dataset_params']['dataset_dir'] = dataset_dir
    
    # Build GAN network

    if configuration['gan']['network'] == 'simpleGAN' : 
        gan = simple_GAN(configuration['gan'] , configuration['dataset_params'], 0)
    else : 
        gan = GAN(configuration['gan'] , configuration['dataset_params'])

    # Train GAN 
    gan.train_with_SGD(run_name , configuration)



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', help='Path to configuration file')
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    parser.add_argument('-d', '--dataset_dir', help='Directory where dataset is located', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name, args.dataset_dir


if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name , dataset_dir = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name, dataset_dir)

