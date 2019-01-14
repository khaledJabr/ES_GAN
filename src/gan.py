from src.models import simpleGAN
import tensorflow as tf
import numpy as np
import os
import random
import sklearn.datasets
from shapely.geometry import Point, box
from shapely.geometry.polygon import Polygon,LinearRing
from src.logger import Logger



# If you add a new network you should add "string --> class" mapping here.
network_dict = {
    "simpleGAN" : simpleGAN
}


## 2d gan
class simple_GAN(object):
    def __init__(self, gan_args, dataset_params , rank):

        # Placeholder for the input state
        self.z_dim = gan_args['args']['z_dim']
        self.batch_size = gan_args['args']['batch_size']
        self.output_dim = 2 
        self.real_data_shape = (None , self.output_dim)
        self.z_shape = (None , self.z_dim)
        self.vb_d = None 
        self.vb_g = None
        self.use_novelty = True if 'novelty_search' in gan_args.keys() else False
        self.g_loss_type = gan_args['args']['loss']
        # setup data 
        self.data = self.inf_train_gen(dataset_params['dataset_name'] , self.batch_size) # need to make this 
        self.rank  = rank

        #Novelty Archive Stuff
        if self.use_novelty : 
            print('##### Novelty search in GAN args')
            self.novelty_archive = []
            self.bc_samples = []
            self.novelty_archive_av = [] 
            self.novelty_population_size = gan_args['novelty_search']['population_size']
            self.k = gan_args['novelty_search']['k'] # k-means parameters
            self.bc = gan_args['novelty_search']['bc'] # type of bc used 
            self.num_novelty_evals = gan_args['novelty_search']['num_novelty_evals']
            self.collect_bc_samples = True if gan_args['novelty_search']['collect_bc_samples'] == 'True' else False
            self.novelty_seed = 123 # using to synchornize bc extraction between workers
            self.novelty_rs = np.random.RandomState(self.novelty_seed)

            # related functions 

            # novelty calculation
            if gan_args['novelty_search']['novelty_calculation'] == 'arch_knn' : 
                self.nov_calc_method = 'arch_knn'
                self.compute_novelty_vs_archive  = self.compute_novelty_1
            elif gan_args['novelty_search']['novelty_calculation'] == 'average' : 
                self.nov_calc_method = 'average'
                self.compute_novelty_vs_archive  = self.compute_novelty_2
            else :
                raise NotImplementedError(gan_args['novelty_search']['novelty_calculation'] )

            # distance function 
            if gan_args['novelty_search']['distance_function'] == 'euclidean' : 
                self.dist_func = self.euclidean_distance
            else : 
                raise NotImplementedError(gan_args['novelty_search']['distance_function'])

            


        # Create session for 1 CPU
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)

        # gut from the network, computed values for each action.
        #Placeholders
        #Build the ting
        NetworkClass = network_dict[gan_args['network']]
        #[D_loss,G_loss,g_sample,input_placeholder_X,input_placeholder_Z, d_features,D_loss_real,D_loss_fake,is_training]
        # self.d_loss , self.g_loss , self.g_sample, self.input_placeholder_X , self.input_placeholder_Z,self.d_features, self.is_training , self.d_on_real , self.d_on_fake
        model = NetworkClass(gan_args, dataset_params)
        self.d_loss = model[0]
        self.g_loss = model[1]
        self.g_sample = model[2]
        self.input_placeholder_X = model[3]
        self.input_placeholder_Z = model[4]
        self.d_features = model[5]
        self.d_on_real = model[6]
        self.d_on_fake = model[7]
        self.is_training = model[8]
        
        if self.use_novelty: # set up bc
            if self.bc =='d_features'  : 
                self.bc_nov =  self.d_features
            elif self.bc =='g_output' : 
                self.bc_nov = self.g_sample
                # mimimal criteria polygon
                self.mc_polygon = Polygon([(-2.5, -2.5), (-2.5, 2.5), (2.5, 2.5), (2.5, -2.5)])
            else : 
                raise NotImplementedError(self.bc)


        # Tensorflow operation to compute and save virtual batch normalization statistics. Maybe
        self.vb_op_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS , scope = 'Discriminator')
        self.vb_op_g = tf.get_collection(tf.GraphKeys.UPDATE_OPS , scope = 'Generator')



        self.sess.run(tf.global_variables_initializer())

        # Those variables will be updated using ES algorithm.
        self.parameters_d = tf.trainable_variables(scope='Discriminator')
        self.parameters_g = tf.trainable_variables(scope='Generator')

        # We need to save parameter shapes. Those are used when extracting parameters from flat array.
        self.parameter_shapes_d = [simple_GAN.shape2int(p) for p in self.parameters_d]
        self.parameter_shapes_g = [simple_GAN.shape2int(p) for p in self.parameters_g]


        # Operations to assign new values to the parameters.
        self.parameters_placeholders_d = [tf.placeholder(dtype=tf.float32, shape=s) for s in self.parameter_shapes_d]
        self.parameters_placeholders_g = [tf.placeholder(dtype=tf.float32, shape=s) for s in self.parameter_shapes_g]

        self.set_parameters_ops_d = [par.assign(placeholder) for par, placeholder in
                                   zip(self.parameters_d, self.parameters_placeholders_d)]
        self.set_parameters_ops_g = [par.assign(placeholder) for par, placeholder in
                                   zip(self.parameters_g, self.parameters_placeholders_g)]

        if self.use_novelty : 
            start_bc = self.get_mean_bc()

            if self.bc == 'g_output' : # check if meet minimal criterion
                nov_poly = box(*self.get_bounding_box(start_bc))
                if not self.mc_polygon.contains(nov_poly) :  # need to be seeded
                    print("######### NEEDS SEEDING")
                    self.add_to_novelty_archive(self.get_dummy_bc())
                else : 
                    self.add_to_novelty_archive(start_bc)
            else : 
                self.add_to_novelty_archive(start_bc)

            if self.nov_calc_method == 'average' :
                self.novelty_archive_av = np.mean(self.novelty_archive, axis = 0 )




    @staticmethod
    def shape2int(x):
        s = x.get_shape()
        return [int(si) for si in s]

   
    def get_vb(self, d_or_g) : 
        # return a random batch from dataset or random z
        if d_or_g == 'D' : 
            vb_d = self.sample_real_data()
            vb_g = self.sample_z()
    
            vb = np.concatenate((vb_d,vb_g), axis = 1)
            self.set_vb(vb , 'D')
            return vb
        else  : 
            vb = self.sample_z_fixed2()
            self.set_vb(vb , 'G')
            return vb

    def set_vb(self, vb , d_or_g) : 
        if d_or_g  == 'D' : 
            self.vb_d = vb
            d_idx = [0,1]
            g_idx = [i for i in range(self.vb_d.shape[1]) if i not in d_idx]
            _d = self.vb_d[:,d_idx]
            _g = self.vb_d[:,g_idx]
            # print("#### vd.shape {}".format(_d.shape))
            # print("#### vg.shape {}".format(_g.shape))

            self.sess.run(self.vb_op_d, feed_dict={self.input_placeholder_X: _d , self.input_placeholder_Z: _g , self.is_training: True})
        else  : # i should be really checking explicity for 'G' here
            self.vb_g = vb 
            self.sess.run(self.vb_op_g, feed_dict={self.input_placeholder_Z: self.vb_g, self.is_training: True})
            

    def get_parameters(self , d_or_g):
        # Extracts parameters from the network and returns flat 1D array with parameter values.
        target_parameters = self.parameters_d if d_or_g == 'D' else self.parameters_g
        parameters = self.sess.run(target_parameters)
        return np.concatenate([p.flatten() for p in parameters])

    def set_parameters(self, parameters, d_or_g):
        # which ones
        target_parameters_placeholders = self.parameters_placeholders_d if d_or_g =='D' else self.parameters_placeholders_g
        target_parameters_shapes = self.parameter_shapes_d if d_or_g =='D' else self.parameter_shapes_g
        target_parameters_ops = self.set_parameters_ops_d if d_or_g == 'D' else self.set_parameters_ops_g
        target_vb = self.vb_d if d_or_g == 'D' else self.vb_g
        target_vb_op = self.vb_op_d if d_or_g == 'D' else self.vb_op_g
        target_placeholder = self.input_placeholder_X if d_or_g =='D' else self.input_placeholder_Z

        # Sets network parameters from flat 1D array with parameter values.
        feed_dict = {}
        current_position = 0
        for parameter_placeholder, shape in zip(target_parameters_placeholders, target_parameters_shapes):
            length = np.prod(shape)
            feed_dict[parameter_placeholder] = parameters[current_position:current_position+length].reshape(shape)
            current_position += length
        self.sess.run(target_parameters_ops, feed_dict=feed_dict)

        # We need to update normalization statistics each time new parameters are set.
        if target_vb is not None :
            self.set_vb(target_vb , d_or_g)
            

        # We need to update normalization statistics each time new parameters are set.
        # if self.vb is not None:
        #     self.sess.run(self.vb_op, feed_dict={self.input_placeholder: self.vb, self.is_training: True}
    def set_parameters_and_run(self, parameters , d_or_g)  : 
        # print("##RUNNING## --> {} | RANK : {}".format(d_or_g, self.rank))

        data_batch = self.sample_real_data()
        batch_z = self.sample_z()
        # Is is antithetic sampling or not 
        # parameters[0] positive noise, parameters[1] negative
        if len(parameters) == 2 : # antithetic sampling : 
            #Positive Noise
            self.set_parameters(parameters[0] , d_or_g) # set d while g is fixed from previous update (vice versa)
            if d_or_g == 'D' : 
                loss_pos = self.run_D(data_batch , batch_z)
                nov_pos = 0
            elif d_or_g == 'G' : 
                loss_pos, nov_vec = self.run_G(batch_z,data_batch)
                nov_pos = self.compute_novelty_vs_archive(nov_vec) if self.use_novelty else 0 

            # Negative Noise
            self.set_parameters(parameters[1] , d_or_g)
            if d_or_g == 'D' : 
                loss_neg = self.run_D(data_batch , batch_z)
                nov_neg = 0 
            elif d_or_g == 'G' : 
                loss_neg, nov_vec = self.run_G(batch_z, data_batch)
                nov_neg = self.compute_novelty_vs_archive(nov_vec) if self.use_novelty else 0 

            return loss_pos,  loss_neg, nov_pos , nov_neg

        # else :  # No antithetic sampling (THERE WILL ALWAYS BE ONE NO KIDDING)
        #     self.set_parameters(parameters, d_or_g)
        #     if d_or_g == 'D' : 
        #         return self.run_D(data_batch , batch_z) ,  0 , 0 
        #     elif d_or_g == 'G' : 
        #         return  self.run_G(batch_z) , 0 , 0 


    # useless method for now
    def set_parameters_and_run_specific_batch(self, parameters , d_or_g , data_batch, batch_z)  : 

        # data_batch = self.sample_real_data()
        # batch_z = self.sample_z()
        # Is is antithetic sampling or not 
        # parameters[0] positive noise, parameters[1] negative
        if len(parameters) == 2 : # antithetic sampling : 
            #Positive Noise
            self.set_parameters(parameters[0] , d_or_g)
            if d_or_g == 'D' : 
                loss_pos = self.run_D(data_batch , batch_z)
            elif d_or_g == 'G' : 
                loss_pos = self.run_G(data_batch , batch_z)

            # Negative Noise
            self.set_parameters(parameters[1] , d_or_g)
            if d_or_g == 'D' : 
                loss_neg = self.run_D(data_batch , batch_z)
            elif d_or_g == 'G' : 
                loss_neg = self.run_G(data_batch , batch_z)

            return loss_pos,  loss_neg

        else :  # No antithetic sampling
            self.set_parameters(parameters, d_or_g)
            if d_or_g == 'D' : 
                return self.run_D(data_batch , batch_z) ,  0 
            elif d_or_g == 'G' : 
                return  self.run_G(data_batch , batch_z) , 0

    def inf_train_gen(self, DATASET , BATCH_SIZE):
        # Taken from Improved wgan paper : github link goes here
        if DATASET == '25gaussians':
        
            dataset = []
            for i in range(int(100000/25)):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2)*0.05
                        point[0] += 2*x
                        point[1] += 2*y
                        dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            dataset /= 2.828 # stdev
            while True:
                for i in range(int(len(dataset)/BATCH_SIZE)):
                    yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        elif DATASET == 'swissroll':

            while True:
                data = sklearn.datasets.make_swiss_roll(
                    n_samples=BATCH_SIZE, 
                    noise=0.25
                )[0]
                data = data.astype('float32')[:, [0, 2]]
                data /= 7.5 # stdev plus a little
                yield data

        elif DATASET == '8gaussians':
        
            scale = 2.
            centers = [
                (1,0),
                (-1,0),
                (0,1),
                (0,-1),
                (1./np.sqrt(2), 1./np.sqrt(2)),
                (1./np.sqrt(2), -1./np.sqrt(2)),
                (-1./np.sqrt(2), 1./np.sqrt(2)),
                (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x,scale*y) for x,y in centers]
            while True:
                dataset = []
                for i in range(BATCH_SIZE):
                    point = np.random.randn(2)*.02
                    center = random.choice(centers)
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414 # stdev
                yield dataset
        elif DATASET == '4gaussians':
            scale = 2.
            centers = [
                (1,0),
                (-1,0),
                (0,1),
                # (0,-1),
                # (1./np.sqrt(2), 1./np.sqrt(2)),
                # (1./np.sqrt(2), -1./np.sqrt(2)),
                # (-1./np.sqrt(2), 1./np.sqrt(2)),
                # (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x,scale*y) for x,y in centers]
            while True:
                dataset = []
                for i in range(BATCH_SIZE):
                    point = np.random.randn(2)*.02
                    center = random.choice(centers)
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414 # stdev
                yield dataset
            
        elif DATASET == '2gaussians':

            scale = 2.
            centers = [
        #             (1,0),
        #             (-1,0),
        #             (0,1),
        #             (0,-1),
                (1./np.sqrt(2), 1./np.sqrt(2)),
        #             (1./np.sqrt(2), -1./np.sqrt(2)),
        #             (-1./np.sqrt(2), 1./np.sqrt(2)),
                (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x,scale*y) for x,y in centers]
            while True:
                dataset = []
                for i in range(BATCH_SIZE):
                    point = np.random.randn(2)*.1
                    center = random.choice(centers)
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414 # stdev
                yield dataset

    def get_y(self, x):
        return  x*x

    def sample_data_2(self,n=256, scale=5):
        n= self.batch_size
        data = []

        x = scale*(np.random.random_sample((n,))-0.5)

        for i in range(n):
            yi = self.get_y(x[i])
            data.append([x[i], yi])

        return np.array(data)

    def sample_data_3(self) : 
        data = np.random.normal(0.6,0.01,(256,2))           # Uncomment this line for multimodal data
        data1 = np.random.normal(-0.6,0.01,(256,2))     # Uncomment this line for multimodal data
        data = np.vstack((data,data1))
        data = data.reshape([-1,2])
        return data 


    def sample_real_data(self , use_simple_func = False): 
        # return self.data[np.random.randint(len(self.data),size=self.batch_size)]
        if use_simple_func : 
            return self.sample_data_2()
        return next(self.data)

    #kde density plot
    def save_density_plot(self, path , np_samples , title , plot_interval) : 
        import seaborn as sns
        # import matplotlib as mtb 
        # mtb.use('agg')
        import matplotlib.pyplot as plt
        xmax = 2.0
        np_samples_ = np_samples[::1]
        cols = len(np_samples_)
        bg_color  = sns.color_palette('Oranges', n_colors=256)[0]
        plt.figure(figsize=(2*cols, 2))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            ax2 = sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Oranges', n_levels=20, clip=[[-xmax,xmax]]*2)
            ax2.set_facecolor(bg_color)
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*plot_interval)) # each 20 iterations we do that
        ax.set_ylabel(title)
        plt.gcf().tight_layout()
        plt.savefig((path + 'density_plot_{}.png').format(len(np_samples)))
        plt.gcf().clear()
        plt.close() 


    def save_samples(self , iteration, path , z_dist= None ) :
        import matplotlib as mtb 
        mtb.use('agg')
        import matplotlib.pyplot as plt
        xx = self.sample_real_data()
        yy = self.sample_g()
        
        xax = plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
        gax = plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title(('Samples at Iteration {}').format(iteration))
        plt.tight_layout()
        plt.savefig((path + '/iteration_{}.png').format(iteration))
        plt.gcf().clear()
        plt.close() 
        return  yy 


    def sample_z(self , bz = None) :
        # return np.random.normal(size=[self.batch_size, self.z_dim])
        batch_size = bz if bz else self.batch_size

        return  np.random.normal(0, 1, size=(batch_size, self.z_dim))
        # return np.random.(1, 0.5, size=(batch_size, self.z_dim))



    def sample_z_novelty_workers(self) : 
        # fix the random state to be the same accross all workers
        return  self.novelty_rs.uniform(-1, 1, size=(self.batch_size, self.z_dim))

    def sample_z_fixed2(self) : 
        # fix the random state to be used accross all workers
        rs = np.random.RandomState(seed=123)
        return  rs.uniform(-1, 1, size=(self.batch_size, self.z_dim))

    def sample_g(self , z_batch_size = None  ) : 
        z_batch = self.sample_z(z_batch_size) if z_batch_size else  self.sample_z() 
        g_sample = self.sess.run(self.g_sample , feed_dict={self.input_placeholder_Z : z_batch , self.is_training : False})
        return g_sample

    def run_D(self , data_batch , z_batch) : 
        # data_batch = self.sample_real_data()
        # z_batch = self.sample_z()
        loss = self.sess.run(self.d_loss, feed_dict={self.input_placeholder_X: data_batch , self.input_placeholder_Z: z_batch, self.is_training : True})
        return loss

    def run_G(self , z_batch , data_batch) : 
        # z_batch = self.sample_z()
        # data_batch = self.sample_real_data()
        if self.use_novelty : 
            if self.g_loss_type == 'RA_LOSS': 
                gloss , nov = self.sess.run([self.g_loss , self.bc_nov] , feed_dict={self.input_placeholder_Z: z_batch, self.input_placeholder_X: data_batch , self.is_training : True})
            else : 
                gloss , nov = self.sess.run([self.g_loss , self.bc_nov] , feed_dict={self.input_placeholder_Z: z_batch , self.is_training : True})

        else : 
            if self.g_loss_type == 'RA_LOSS':
                gloss  = self.sess.run(self.g_loss , feed_dict={self.input_placeholder_Z: z_batch , self.input_placeholder_X : data_batch , self.is_training : True})

            else : 
                gloss  = self.sess.run(self.g_loss , feed_dict={self.input_placeholder_Z: z_batch , self.is_training : True})
            nov = 0 
        # print("Types : loss {} | nov : {} ### Shape Nov : {} ".format(type(gloss), type(nov) , nov.shape))
        return gloss, nov

    def get_mean_bc(self):
        # Need to fix the z to synchronize workers 
        # if self.bc == 'g_output' : # bc is the output of the generator 
        novelty_vector = []
        for i in range(self.num_novelty_evals) :
            z_batch = self.sample_z_novelty_workers()  # is important that z is the sample accross all workers
            nv_ = self.sess.run(self.bc_nov , feed_dict={self.input_placeholder_Z : z_batch , self.is_training : False})
            novelty_vector.append(nv_)

        self.novelty_seed += 1  # just to make things a bit different next time this is run
        self.novelty_rs.seed(self.novelty_seed) # chaning things up for next iteration
        return np.mean(novelty_vector , axis = 0)

    def get_dummy_bc(self) : 
        if self.bc == 'g_output':
            return np.random.uniform(-1 , 1 , size=(self.batch_size, self.real_data_shape[1]))
        else : 
            raise NotImplementedError(self.bc)
    def get_novelty_archive(self) : 
        return self.novelty_archive

    def get_bounding_box(self, nov): 
        smallest_on_x = np.amin(nov[:,0])
        smallest_on_y = np.amin(nov[:,1])

        biggest_on_x = np.amax(nov[:,0])
        biggest_on_y = np.amax(nov[:,1])
        return smallest_on_x, smallest_on_y, biggest_on_x, biggest_on_y

    def add_to_novelty_archive(self, nv) : 
        # need to decided what happens on first time (with rank 0 only tho)
    

        if self.bc == 'g_output' : # check for me
            nov_poly = box(*self.get_bounding_box(nv))
            if not self.mc_polygon.contains(nov_poly) :  # we want it to meet mc
                return
            else :
                self.novelty_archive.append(nv)
        else : 
            self.novelty_archive.append(nv)
        if self.nov_calc_method == 'average' :
            self.novelty_archive_av = np.mean(self.novelty_archive, axis = 0 )


    def set_novelty_archive(self, arch ) :
        # this is awful because you have to synrhonize the fuckin workers 
        # caution  : only use this at the very first of starting the algoritthm
        print("### arch : ", arch.shape)
        self.novelty_archive = []
        self.novelty_archive.append(np.array(arch , dtype = np.float64)) # elements inside archive (list) are np vectors 
        if self.nov_calc_method == 'average' :
            self.novelty_archive_av = np.mean(self.novelty_archive, axis = 0 )

    def get_archive_size(self) : 
        return len(self.novelty_archive)

    # using archive 
    def compute_novelty_1(self, novelty_vector):
        # check minimal critiertia
        if self.bc == 'g_output' : 
            nov_poly = box(*self.get_bounding_box(novelty_vector))
            if not self.mc_polygon.contains(nov_poly) :  # we want it to meet mc
                return 0

        distances = []
        nov = novelty_vector.astype(np.float)
        for point in self.novelty_archive:
            distances.append(self.euclidean_distance(point.astype(np.float), nov))
        # Pick k nearest neighbors
        distances = np.array(distances)
        top_k_indicies = (distances).argsort()[:self.k]
        top_k = distances[top_k_indicies]
        return top_k.mean()

    # using a running average of the archive
    def compute_novelty_2(self, novelty_vector): # test method do not use
        # distances = []
        if self.bc == 'g_output' : 
            nov_poly = box(*self.get_bounding_box(novelty_vector))
            if not self.mc_polygon.contains(nov_poly) :  # we want it to meet mc
                return 0

        nov = novelty_vector.astype(np.float)
        return self.euclidean_distance(self.novelty_archive_av, novelty_vector)

    # helper method 
    def euclidean_distance(self,x, y):
        return np.sqrt(np.sum((x-y)**2))


    def train_with_SGD(self, run_name, config) :
        iterations = 25001

        # for density plot 
        np_samples = []
        bc_samples = []
        # Logging space 
        log_path = "logs_mpi/gan_with_sgd/" + run_name
        logger = Logger(log_path) # deals with dir creations
        # Set up solvers
        gen_vars = self.parameters_g
        disc_vars = self.parameters_d

        print('******************** TRAINING WITH SGD ******************************')
        print("D Dim :" , np.sum([np.prod(v.get_shape().as_list()) for v in disc_vars]))
        print("G Dim :" , np.sum([np.prod(v.get_shape().as_list()) for v in gen_vars]))
        print('********************     START         ******************************')


        self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(-self.d_loss, var_list=disc_vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(-self.g_loss, var_list=gen_vars)

        #
        self.sess.run(tf.global_variables_initializer())


        # Lets go 
        for i in range(iterations) : 

            # D 
            batch_data = self.sample_real_data()
            batch_z = self.sample_z()

            _, D_loss_curr = self.sess.run([self.D_solver, self.d_loss], feed_dict={self.input_placeholder_X: batch_data, self.input_placeholder_Z: batch_z})

            # batch_z = self.sample_z() 
            if config['gan']['args']['loss'] == 'RA_LOSS' : 
                _, G_loss_curr = self.sess.run([self.G_solver, self.g_loss], feed_dict={self.input_placeholder_Z: batch_z ,self.input_placeholder_X : batch_data}) 
            else : 
                _, G_loss_curr = self.sess.run([self.G_solver, self.g_loss], feed_dict={self.input_placeholder_Z: batch_z}) 

            if i % 100 == 0 : # plot results every 20 iterations
                print("-- > Iteration : {}".format(i))
                print("\tD loss : {} ".format(D_loss_curr))
                print("\tG loss : {} ".format(G_loss_curr))



            if i % 100 == 0 : 
                self.save_samples(i , logger.plot_dir)
            if i % 5000  == 0 : 
                samples  = self.sample_g(1000)
                # np_samples.append(np.vstack([self.sample_g() for _ in range(10)]))
                np_samples.append(samples)

        # # Density Plot stuff 
        # logger.save_stats(bc_samples , 'bc_samples')
        logger.save_stats(np_samples , 'np_samples')

        self.save_density_plot(log_path , np_samples , 'z dim = 8' , 5000)
        # Done



