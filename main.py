from src.optimizers import OpenAIOptimizer,  NSRESOptimizer
from src.gan import simple_GAN
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

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
    start_time = time.time()

    # Reading configuration file
    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())
    configuration['dataset_params']['dataset_dir'] = dataset_dir



    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    train_cpus = cpus - 1

    # Deduce population size (I don't really need this anymore)
    lam = train_cpus * ep_per_cpu

    # Shared Noise Table : 
    noise_table_size = 5e7
    itemsize = MPI.FLOAT.Get_size()
    if comm.Get_rank() == 0 : 
        nbytes = noise_table_size * itemsize
    else : 
        nbytes = 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 
    buf, itemsize = win.Shared_query(0) 
    assert itemsize == MPI.FLOAT.Get_size() 

    # All workers allocate the same table
    noise_table = np.ndarray(buffer=buf, dtype='float32', shape=(int(noise_table_size),)) 
    noise_table[:] =  np.random.RandomState(123).randn(int(noise_table_size)).astype('float32')



    # Build GAN network (simple gan is the gaussian 2d gan)
    if configuration['gan']['network'] == 'simpleGAN' : 
        gan = simple_GAN(configuration['gan'] , configuration['dataset_params'], rank)
    else : 
        gan = GAN(configuration['gan'] , configuration['dataset_params'])

    # Create reference batch used for normalization


    # Extract vector with current parameters.
    parameters_d = gan.get_parameters('D')
    parameters_g = gan.get_parameters('G')

    # Extract Virtual Batch  
    if configuration['gan']['args']['use_vb_d'] == 'True' :
        vb_d = gan.get_vb('D')

    if configuration['gan']['args']['use_vb_g'] == 'True' : 
         vb_g = gan.get_vb('G')

    # Novelty Archive Stuff 
    if 'novelty_search' in configuration['gan'].keys() : 
        nov_arch_first = np.array(gan.get_novelty_archive()[0], dtype=np.float32) # There should be ONLY ONE entry in the archive here


    # Send parameters, novelty archive (and virtual batches) from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position

    # Network parameters
    comm.Bcast([parameters_d, MPI.FLOAT], root=0)
    comm.Bcast([parameters_g, MPI.FLOAT], root=0)

    # Novelty Archive 
    if 'novelty_search' in configuration['gan'].keys() : 
        comm.Bcast([nov_arch_first, MPI.FLOAT], root=0)

    # Virtual Batch
    if configuration['gan']['args']['use_vb_d'] == 'True' : 
        comm.Bcast([vb_d, MPI.FLOAT], root=0)
    if configuration['gan']['args']['use_vb_g'] == 'True' : 
        comm.Bcast([vb_g, MPI.DOUBLE], root=0)


    # Setting the Virtual Batch Normalization for each work
    if configuration['gan']['args']['use_vb_d'] == 'True' :
        if rank != 0:
            gan.set_vb(vb_d , 'D')
    if configuration['gan']['args']['use_vb_g'] == 'True' : 
        if rank != 0:
            gan.set_vb(vb_g , 'G')

    # Novelty search stuff 
    if 'novelty_search' in configuration['gan'].keys() : 
        if rank != 0:
            gan.set_novelty_archive(nov_arch_first)



    # Create optimizer with user defined settings (hyperparameters)
    # Could have created a class that encompasses both but this works too
    OptimizerClass_d = optimizer_dict[configuration['optimizer_d']]
    optimizer_d = OptimizerClass_d(parameters_d, lam, rank, noise_table, configuration["settings_d"])

    OptimizerClass_g = optimizer_dict[configuration['optimizer_g']]
    optimizer_g = OptimizerClass_g(parameters_g, lam, rank, noise_table, configuration["settings_g"])

    
    if 'novelty_search' in configuration['gan'].keys() : 
        # gan.add_to_novelty_archive(gan.get_mean_bc())
        # print('{} --> Woker : {} | g_w  : {} '.format(optimizer_g.iteration , rank, optimizer_g.parameters[:5]))
        print("{} :: WORKER : {} --> novelty seed : {} | novelty_archive[0][:5] : {}".format(optimizer_g.iteration, rank, gan.novelty_seed, gan.get_novelty_archive()[0][:5]))


            # print("##### CREATED NOVELTY ARCHIVE")

    # Only rank 0 worker will log information from the training
    logger = None
    if rank == 0:
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger = Logger(optimizer_d.log_path(game, configuration['gan']['network'], run_name))
        # logger.save_vb(vb)

        # Log basic stuff
        logger.log('Run Name'.ljust(25) + '%s' % game)
        logger.log('--------------- GAN ---------------------')
        logger.log('Network'.ljust(25) + '%s' % configuration['gan']['network'])
        logger.log('Optimizer (D)'.ljust(25) + '%s' % configuration['optimizer_d'])
        logger.log('Optimizer (G)'.ljust(25) + '%s' % configuration['optimizer_g'])
        logger.log('LOSS'.ljust(25) + '%s' % configuration['gan']['args']['loss'])
        logger.log('Dataset'.ljust(25) + '%s' % configuration['dataset_params']['dataset_name'])
        logger.log('Batch Size'.ljust(25) + '%d' % configuration['gan']['args']['batch_size'])
        logger.log('Z Dimension'.ljust(25) + '%d' % configuration['gan']['args']['z_dim'])
        logger.log('nonlin (D)'.ljust(25) + '%s' % configuration['gan']['args']['nonlin_d'])
        logger.log('nonlin (G)'.ljust(25) + '%s' % configuration['gan']['args']['nonlin_g'])
        logger.log('--------------- ES ---------------------')
        logger.log('Number of CPUs'.ljust(25) + '%d' % cpus)
        logger.log('Population'.ljust(25) + '%d' % lam)
        logger.log('Dimensionality (D)'.ljust(25) + '%d' % len(parameters_d))
        logger.log('Dimensionality (G)'.ljust(25) + '%d' % len(parameters_g))
        if gan.use_novelty:
            logger.log('(Novelty) Nov_Calc Fun '.ljust(25) + '%s' % configuration['gan']['novelty_search']['novelty_calculation']) 
            logger.log('(Novelty) Dist Fun'.ljust(25) + '%s' % configuration['gan']['novelty_search']['distance_function']) 


        logger.log('--------------- Misc---------------------')
        logger.log('Save Params Every'.ljust(25) + '%d' % configuration['misc']['save_params_every'])
        logger.log('Collect Stats Every'.ljust(25) + '%d' % configuration['misc']['collect_stats_every'])
        logger.log('Vis Every'.ljust(25) + '%d' % configuration['misc']['vis_every'])
        logger.log('Density Sample Every'.ljust(25) + '%d' % configuration['misc']['density_sample_every'])
        logger.log('Density Plot Every'.ljust(25) + '%d' % configuration['misc']['density_every'])



        # Log basic info from the optimizer
        logger.log('--------------- DISCRIMINATOR OPTIMIZER---------------------')
        optimizer_d.log_basic(logger)
        logger.log('--------------- GENERATOR     OPTIMIZER---------------------')
        optimizer_g.log_basic(logger)

        # variables to hold statistics
        np_samples =[]
        loss_d = []
        loss_g = []
        
        if gan.use_novelty :
            bc_samples_c = []
            novelty_c = []
        

    # START 
    steps_passed = 0 # Iterations ( D + G )
    task_type = 0 # 0 for D , 1 for G 
    while True:
      
        # Determind task type
        optimizer = optimizer_d if task_type == 0 else optimizer_g
        d_or_g = 'D' if task_type == 0 else 'G'

        # sad debuggin way 
        # print("#### Worker : {}  -- > Doing  ::::  {}  ::: now".format(rank , d_or_g))
        # if d_or_g == 'D' :
        #     print('{} --> Woker : {} | d_w  : {} '.format(optimizer_d.iteration , rank, optimizer_d.parameters[:5]))
        # if d_or_g == 'G' :
        #     print('{} --> Woker : {} | g_w  : {} '.format(optimizer_g.iteration , rank, optimizer_g.parameters[:5]))



        # if d_or_g == 'D' :
        #     print('{} --> Woker : {} | d_w  : {} '.format(optimizer_d.iteration , rank, optimizer_d.noise_table[:5]))
        # if d_or_g == 'G' :
        #     print('{} --> Woker : {} | g_w  : {} '.format(optimizer_g.iteration , rank, optimizer_g.noise_table[:5]))



        # Iteration start time
        iter_start_time = time.time()
        # Workers that run training iteration
        if rank != 0:
            # Empty arrays for each episode. We save: returns P, Return N, noise index. 
            rets_p = [0] * ep_per_cpu # returns from positive noise eval
            rets_n = [0] * ep_per_cpu # returns from negative noise eval
            inds = [0] * ep_per_cpu
            novs_p = [0] * ep_per_cpu # novelty return for positive noise eval 
            novs_n = [0] * ep_per_cpu # novelty return for negative noise eval

            # For each iteration in this CPU we get new parameters,
            # update gan (d or g) network and run d/g 
            # Hold the other network (D or G constant tho)
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                ret_p , ret_n , nov_p , nov_n = gan.set_parameters_and_run(p , d_or_g)
                # e_ret = gan.run_D() if task_type == 0 else gan.run_G()
                rets_p[i] = ret_p
                rets_n[i] = ret_n
                inds[i] = ind
                novs_p[i] = nov_p 
                novs_n[i] = nov_n 

                # print("{} :: WORKER : {} --> ret_p : {} | ret_n : {} | ind :{} | nov_p : {} | nov_n : {}".format(optimizer.iteration, rank, ret_p, ret_n, ind , nov_p , nov_n))
            # Aggregate information, will later send it to each worker using MPI
            # what if I aggregate here  tho instead in master 
            msg = np.array(rets_p + rets_n + inds + novs_p + novs_n, dtype=np.float64)

        # # Worker rank 0 that runs do stats collection
        else:
            # Use this worker to collect statics and plot results
            #gan_iter= 1 if optimizer.iteration == 0 else optimizer.iteration # to collect samples at very first iteration

            # viz results 
            if (optimizer.iteration -1 ) % configuration['misc']['vis_every'] == 0 and task_type == 1:  # End of a GAN iteration
                g_samples = gan.save_samples(optimizer.iteration -1 , logger.plot_dir)


            # saving parameters
            if (optimizer.iteration -1 ) % configuration['misc']['save_params_every'] == 0 and task_type == 1:  # End of a GAN iteration
                logger.save_parameters(optimizer_d.parameters , 'D', optimizer_d.iteration -1)
                logger.save_parameters(optimizer_g.parameters , 'G', optimizer_g.iteration -1)


             # Collect samples for density plot
            if (optimizer.iteration -1) % configuration['misc']['density_sample_every'] == 0 and task_type == 1: # End of a GAN iteration
                samples  = gan.sample_g(1000)
                np_samples.append(samples)
            #  # Jus a test
            # if (optimizer.iteration -1 ) % configuration['misc']['save_params_every'] == 0 and gan.save_bc_samples and task_type == 1:  # End of a GAN iteration
            #     logger.save_bc_samples(bc_samples)


            # Collecting gan statistics
            if (optimizer.iteration -1) % configuration['misc']['collect_stats_every'] == 0 and task_type == 1: # End of a GAN iteration
                
                # Loss statistics
                d_batch = gan.sample_real_data()
                z_batch = gan.sample_z()

                l_d_curr = gan.run_D(d_batch , z_batch)
                l_g_curr = gan.run_G(z_batch , d_batch)[0]



                loss_d.append(l_d_curr)
                loss_g.append(l_g_curr)

                # print('l d : {} | l_g : {} | l_d_curr : {} | l_g_curr : {}'.format(l_d_curr,l_g_curr, loss_d , loss_g))

                # Novelty Statistics
                if gan.use_novelty :
                    bc_curr = gan.get_mean_bc()
                    nov_curr = gan.compute_novelty_vs_archive(gan.get_mean_bc())
                    bc_samples_c.append(bc_curr)
                    novelty_c.append(nov_curr)
                    print(novelty_c)

            
            # # bc samples special 
            # if (optimizer.iteration - 1 )% 1000 and task_type == 1 :
            #     logger.save_stats(bc_samples_c , 'bc_samples{}'.format(optimizer.iteration - 1))

            # Save all (Do once)
            if (optimizer.iteration - 1 ) == configuration['misc']['density_every'] and task_type == 1:  # (task_type = 1 ) End of a GAN iteration
                title = 'NSR-ES' if gan.use_novelty else 'ES'
                gan.save_density_plot(logger.log_dir , np_samples, title, configuration['misc']['density_sample_every']) # kde density plot

                # save statas
                logger.save_stats(np_samples, 'np_samples')
                logger.save_stats(loss_d, 'loss_d')
                logger.save_stats(loss_g, 'loss_g')
                if gan.use_novelty : 
                    logger.save_stats(novelty_c , 'novelty')
                    logger.save_stats(bc_samples_c , 'bc_samples')


            msg = np.zeros(5 * ep_per_cpu, dtype=np.float64) # empty results


        # MPI stuff
        # Initialize array which will be updated with information from all workers using MPI
        # 3 : returns (positive and negative) and indicies
        results = np.empty((cpus, 5 * ep_per_cpu), dtype=np.float64)
        comm.Allgather([msg, MPI.DOUBLE], [results, MPI.DOUBLE])

        # Skip empty evaluation results from worker with id 0
        results = results[1:, :]

        # Extract IDs and rewards
        rets_p = results[:, :ep_per_cpu].flatten()
        rets_n = results[:, ep_per_cpu:(2*ep_per_cpu)].flatten()
        ids = results[:,(2*ep_per_cpu):(3*ep_per_cpu)].flatten().astype('int32')
        novs_p = results[:,(3*ep_per_cpu):(4*ep_per_cpu)].flatten()
        novs_n = results[:,(4*ep_per_cpu):].flatten()

        # if rank ==0 : 
        #     print('##### RESULTS Iteration : {}#####'.format(optimizer.iteration))
        #     print('rets_p :{}'.format(rets_p))
        #     print('rets_n :{}'.format(rets_n))
        #     print('ids :{}'.format(ids))
        #     print('novs_p :{}'.format(novs_p))
        #     print('novs_n :{}'.format(novs_n))
        #     print('##### END RESULTS #####')


        # Update parameters. warining : this is not master client arch stop thinking about it that way. 
        # Note that OpenAIOptimizers will Ignore the novelty_r argument as it does not factor in
        new_params = optimizer.update(ids=ids, rewards=[rets_p , rets_n] , novelty_r = [novs_p , novs_n])
        gan.set_parameters(new_params , d_or_g) # updat all workder's D so it's fixed when updating G next Iteration(vice versa)
        # print("----> worker :{} | updated : {}".format(rank, d_or_g))
        # update novelty archive after gnenerator update (if it is being used)
        # debugging shit
        # if gan.use_novelty : 
        #     print("{} :: WORKER : {} --> novelty seed : {} | novelty_archive[:5] : {}".format(optimizer.iteration, rank, gan.novelty_seed, gan.get_novelty_archive()[:5]))
        

        if gan.use_novelty  and (optimizer.iteration) % configuration['gan']['novelty_search']['bc_every'] == 0 and task_type == 1: 
            # # we're only using novelty seach with generator, so we update archive after we update it 
            # if d_or_g == 'G' : 
            gan.add_to_novelty_archive(gan.get_mean_bc())
            # print("{} :: WORKER : {} --> novelty seed : {} | novelty_archive[:5] : {}".format(optimizer.iteration, rank, gan.novelty_seed, gan.get_novelty_archive()[-1][0][:5]))

        steps_passed +=1 
        
        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        if rank == 0   :
            # Log stuff every 100
            if (optimizer.iteration) % 50 == 0 : 
                # print('---> Task Type :{}'.format(task_type))
                iteration_time = (time.time() - iter_start_time)
                time_elapsed = (time.time() - start_time)/60
                train_mean_ret_p = np.mean(rets_p)
                train_mean_ret_n = np.mean(rets_n)

                train_mean_nov_p  = np.mean(novs_p)
                train_mean_nov_n  = np.mean(novs_n)


                logger.log('------------------** {} **------------------'.format(d_or_g), print_message = True)
                logger.log('(GAN) Iteration'.ljust(25) + '%f' % (optimizer.iteration ) ,print_message=True)
                logger.log('(D+G) Iteration'.ljust(25) + '%f' % (optimizer_d.iteration + optimizer_g.iteration) ,print_message=True)
                logger.log('(+P :LOSS) Mean Loss'.ljust(25) + '%f' % train_mean_ret_p,print_message=True)
                logger.log('(-P :LOSS) Mean Loss'.ljust(25) + '%f' % train_mean_ret_n,print_message=True)
                logger.log('(+P :NOV) Mean Loss'.ljust(25) + '%f' % train_mean_nov_p,print_message=True)
                logger.log('(-P :NOV) Mean Loss'.ljust(25) + '%f' % train_mean_nov_n,print_message=True)
                logger.log('Steps Since Start'.ljust(25) + '%f' % steps_passed ,print_message=True) # D+G iteration / sanity check 
                if gan.use_novelty : 
                    logger.log('(ARCH) Nov Archiv Size'.ljust(25) + '%f' % gan.get_archive_size(),print_message=True)
                logger.log('Iteration Time'.ljust(25) + '%f' % iteration_time,print_message=True)
                logger.log('TimeSince Start'.ljust(25) + '%f' % time_elapsed,print_message=True)

                # Give optimizer a chance to log its own stuff
                optimizer.log(logger)
                logger.log('------------------*******------------------', print_message=True)
            



        # ALTERNATE !!!!
        task_type = 1 if task_type == 0 else 0 # if D, switch to G and vice versa 



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

