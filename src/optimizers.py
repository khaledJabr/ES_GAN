import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import namedtuple
import time


class BaseOptimizer(object):
    def __init__(self, parameters, rank , noise_table):
        # Worker id (MPI stuff).
        self.rank = rank
        # 2 GB of random noise as in OpenAI paper.
        # self.noise_table = np.random.RandomState(123).randn(int(5e7)).astype('float32')
        self.noise_table = noise_table
        # Dimensionality of the problem
        self.n = len(parameters)
        # Current solution (The one that we report).
        self.parameters = parameters
        # Computed update, step in parameter space computed in each iteration.
        self.step = 0
        # Should be increased when iteration is done.
        self.iteration = 0

    # Sample random index from the noise table.
    def r_noise_id(self):
        return np.random.random_integers(0, len(self.noise_table) - self.n - 1  )

    # Returns parameters to evaluate for a worker and an ID.
    # ID is used when computing update step (It might indicate index in the noise table).
    def get_parameters(self):
        raise NotImplementedError

    # Function to compute how far from the origin the current solution is and how
    # big are the update steps.
    def magnitude(self, vec):
        return np.sqrt(np.sum(np.square(vec)))

    # Updates Optimizer based on IDs and rewards from evaluations
    def update(self, ids, rewards):
        raise NotImplementedError

    # Use logger to log basic info after each iteration
    def log(self, logger):
        raise NotImplementedError

    # Use logger to log basic info.
    # Will be called at the beginning of the training.
    def log_basic(self, logger):
        raise NotImplementedError

    # Each optimizer might have different folder structure to log results.
    # Advice: derive log_path from optimizer parameters and this function
    # parameters.
    def log_path(self, game, network, run_name):
        raise NotImplementedError

    # Used to log optimizer specific statistics.
    # Will be called after each iteration and saved in separate file.
    def stat_string(self):
        return None


class OpenAIOptimizer(BaseOptimizer):
    # Place for OpenAI algorithm from:
    # Evolution Strategies as a Scalable Alternative to Reinforcement Learning
    # https://arxiv.org/pdf/1703.03864.pdf
    # Adam Optimizer
    def __init__(self, parameters, lam, rank, noise_table,  settings):
        super().__init__(parameters, rank, noise_table)

        self.return_proc_mode = settings['return_proc_mode']
        self.lam = lam # population, but do i even need this ? 
        self.update_ratio = 0  

        # Extract parameters from configuration file.
        self.sigma = settings['sigma']
        self.weight_decay = settings['l2coeff'] # l2  
        self.lr = settings['learning_rate']
        self.beta1 = settings['beta1']
        self.beta2 = settings['beta2']
        self.c_sigma_factor = settings['c_sigma_factor']
        self.epsilon = 1e-08


        # Related Variables 
        self.v = np.zeros(self.n, dtype = np.float32)
        self.m = np.zeros(self.n, dtype = np.float32)


        # Gradient.
        self.g = np.zeros(self.n, dtype = np.float32)


    def get_parameters_current(self) : 
        return None, self.parameters


    def get_parameters(self):
        # antithetic sampling : return parameters with positive and negative noise.
    
        # print("### WORKER {} called for parameters".format(self.rank))
        self.r_id = self.r_noise_id()
        try:
            p_pos = self.parameters + self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]
            p_neg = self.parameters - self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]
        except ValueError: 
            print('---> DIS  : {}'.format(self.r_id))

        return self.r_id, [p_pos, p_neg] 

    # Noise index of evaluated solution is used as an ID in this optimizer.
    def update(self, ids, rewards , novelty_r):

        #Processing Ids and Rewards (Mirriod Sampling
        returns = np.column_stack((rewards[0] , rewards[1]))

        # if self.rank == 0 : 
        #     print('--> ### WORKER  :{} |returns : {}'.format(self.rank, returns))

        assert ids.shape[0] == returns.shape[0]

        if self.return_proc_mode == 'centered_rank' : 
            returns = self.compute_centered_ranks(returns)
        elif self.return_proc_mode == 'none' : 
            returns = returns
        else : 
            raise NotImplementedError(self.return_proc_mode)

        # Caculate Step/Gradiate : 
        t1 = time.time()
        rets = returns[:,0] - returns[:,1] # mirrored sampling stuff

        # standardize the rewards to have a gaussian distribution(maybe nat)
        # normalized_rets = (rets - np.mean(rets)) / np.std(rets)

        g = np.zeros(self.n , dtype=np.float32) # same as zeros(self.n)
        for i in range(len(ids)) :
            g += rets[i] *  self.noise_table[ids[i]:ids[i]+ self.n]
            # self.noise_table[int(ind):int(ind) + self.n]

    
        # g /= (returns.size * self.sigma)# negative and postivie sizes should be the same
        g /= (rets.size)# negative and postivie sizes should be the same

        # g = g.sum()
        t2 = time.time()

        # print('---> AGGREGATED {} RESULTS IN {} | SHAPE  :{}'.format(returns.shape[0], t2-t1 , returns.shape))

        assert g.shape == (self.n,) and g.dtype == np.float32 

        self.g = g 
        # take step / update 
        self._update(-g + self.weight_decay * self.parameters)
        return self.parameters 

    def _compute_step(self, globalg) : 
        a = self.lr * np.sqrt(1 - self.beta2 ** self.iteration) / (1 - self.beta1 ** self.iteration)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def _update(self , globalg) : 
        self.iteration +=1 
        self.step = self._compute_step(globalg)
        # parameters = self.parameters
        self.update_ratio = np.linalg.norm(self.step) / np.linalg.norm(self.parameters)
        self.parameters += self.step



    def log_basic(self, logger):
        # logger.log('Lambda'.ljust(25) + '%d' % self.lam)
        logger.log('Learning Rate'.ljust(25) + '%f' % self.lr)
        logger.log('Sigma'.ljust(25) + '%f' % self.sigma)
        logger.log('Weight Decay'.ljust(25) + '%f' % self.weight_decay)
        # logger.log('Momentum'.ljust(25) + '%f' % self.momentum)
        logger.log('Param Norm'.ljust(25) + '%f' % self.magnitude(self.parameters))

    def log(self, logger):
        logger.log('Param Norm'.ljust(25) + '%f' % self.magnitude(self.parameters), print_message = True)
        logger.log('Grad Norm'.ljust(25) + '%f' % self.magnitude(self.g), print_message = True)
        # logger.log('WeightDecayNorm'.ljust(25) + '%f' % self.magnitude(self.wd))
        logger.log('Ste pNorm'.ljust(25) + '%f' % self.magnitude(self.step), print_message = True)
        logger.log('Update Ratio'.ljust(25) + '%f' % self.magnitude(self.update_ratio), print_message = True)


    def log_path(self, game, network, run_name):
        return "logs_mpi/%s/ES/%s/%s" % \
               (game, network, run_name)

    

    # Helper Methods 
    # Same as OpenAI Implementation 
    # https://arxiv.org/pdf/1703.03864.pdf

    def compute_ranks(self, x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks


    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y



class NSRESOptimizer(BaseOptimizer):
    # Novelty Search Reward Evolution Stratigies (Quality diversity)
    # Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents
    # https://arxiv.org/abs/1712.06560
    # same implementation as OpenAIOptimizer but with novelty factored in. Seperate class for ease of use by others
    def __init__(self, parameters, lam, rank, noise_table,  settings):
        super().__init__(parameters, rank, noise_table)

        
        self.lam = lam
        self.update_ratio = 0 
        # Extract parameters from configuration file.
        self.return_proc_mode = settings['return_proc_mode']
        self.novelty_proc_mode = settings['novelty_proc_mode']
        self.sigma = settings['sigma']
        self.weight_decay = settings['l2coeff'] # l2  
        self.lr = settings['learning_rate']
        self.beta1 = settings['beta1']
        self.beta2 = settings['beta2']
        self.c_sigma_factor = settings['c_sigma_factor']
        self.reward_p = settings['reward_p']
        self.epsilon = 1e-08
        self.novelty_type = settings['novelty_type']

        # print('-------> NOVELTY TYPE :{}'.format(self.novelty_type))

        # Related Variables 
        self.v = np.zeros(self.n, dtype = np.float32)
        self.m = np.zeros(self.n, dtype = np.float32)


        # Gradient.
        self.g = np.zeros(self.n, dtype = np.float32)

   

    def get_parameters_current(self) : 
        return None, self.parameters


    def get_parameters(self):
        # antithetic sampling : return parameters with positive and negative noise.

        self.r_id = self.r_noise_id()
        p_pos = self.parameters + self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]
        p_neg = self.parameters - self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]

        return self.r_id, [p_pos, p_neg] 

    # Noise index of evaluated solution is used as an ID in this optimizer.
    def update(self, ids, rewards , novelty_r):

        #Processing Ids and Rewards (Mirriod Sampling
        # returns_positive  = rewards[0]
        # returns_negative  = rewards[1]

        # returns = np.array([rewards[0].reshape(-1,1) , rewards.reshape(-1,1)], axis = 1)
        returns = np.column_stack((rewards[0] , rewards[1]))
        novelty = np.column_stack((novelty_r[0] , novelty_r[1]))

        assert ids.shape[0] == returns.shape[0]

        # Processing Returns (rewards)
        if self.return_proc_mode == 'centered_rank' : 
            returns = self.compute_centered_ranks(returns)
        elif self.return_proc_mode == 'none' : 
            returns = returns
        else : 
            raise NotImplementedError(self.return_proc_mode)

        # Processing Novelty returns
        if self.novelty_proc_mode == 'centered_rank' : 
            novelty = self.compute_centered_ranks(novelty)
        elif self.novelty_proc_mode == 'none' : 
            novelty = novelty
        else : 
            raise NotImplementedError(self.novelty_proc_mode)

        # Caculate Step/Gradiate : 
        t1 = time.time()

        if self.novelty_type =='novelty_only' : # use novelty information only 
            overall_returns = novelty
        elif self.novelty_type =='quality_diversity' : 
            overall_returns = (self.reward_p * returns) + ( (1-self.reward_p) * novelty) # discriminitor and novelty combined here
        else :
            raise NotImplementedError(self.novelty_type)

        rets = overall_returns[:,0] - overall_returns[:,1] # mirrored sampling stuff

        # should I standarize results here or nah ? feels like it should but whatever
        # standardize the rewards to have a gaussian distribution
        # normalized_rets = (rets - np.mean(rets)) / np.std(rets)
        g = np.zeros_like(self.parameters) # same as zeros(self.n)
        for i in range(len(ids)) :
            g += rets[i] *  self.noise_table[int(ids[i]):(int(ids[i])+ self.n)]

    
        # g /= (returns.size * self.sigma)# negative and postivie sizes should be the same
        g /= (rets.size)# negative and postivie sizes should be the same

        # g = g.sum()
        t2 = time.time()

        # print('---> AGGREGATED {} RESULTS IN {} | SHAPE  :{}'.format(returns.shape[0], t2-t1 , returns.shape))

        assert g.shape == (self.n,) and g.dtype == np.float32 

        # take step / update 
        self._update(-g + self.weight_decay * self.parameters)
        self.g = g 

        # Update meta population 
        # self.parameters_dict[self.current_parent] = self.parameters
        # Remeber to add a new novelty vector to the archive
        return self.parameters 

    def _compute_step(self, globalg) : 
        a = self.lr * np.sqrt(1 - self.beta2 ** self.iteration) / (1 - self.beta1 ** self.iteration)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def _update(self , globalg) : 
        self.iteration +=1 
        self.step = self._compute_step(globalg)
        # parameters = self.parameters
        self.update_ratio = np.linalg.norm(self.step) / np.linalg.norm(self.parameters)
        self.parameters += self.step


        # Maybe do noise adaptation stuff here doe? 

    # Helper Methods 
    # Same as OpenAI Implementation 
    # https://arxiv.org/pdf/1703.03864.pdf
    def compute_ranks(self, x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks


    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y


    def log_basic(self, logger):
        logger.log('LearningRate'.ljust(25) + '%f' % self.lr)
        # logger.log('Lambda'.ljust(25) + '%d' % self.lam)
        logger.log('Sigma'.ljust(25) + '%f' % self.sigma)
        logger.log('Weight Decay'.ljust(25) + '%f' % self.weight_decay)
        
        
        # logger.log('Momentum'.ljust(25) + '%f' % self.momentum)
        logger.log('ParamNorm'.ljust(25) + '%f' % self.magnitude(self.parameters))

    def log(self, logger):
        logger.log('Param Norm'.ljust(25) + '%f' % self.magnitude(self.parameters), print_message = True)
        logger.log('Grad Norm'.ljust(25) + '%f' % self.magnitude(self.g), print_message = True)
        # logger.log('WeightDecayNorm'.ljust(25) + '%f' % self.magnitude(self.wd))
        logger.log('Ste pNorm'.ljust(25) + '%f' % self.magnitude(self.step), print_message = True)
        logger.log('Update Ratio'.ljust(25) + '%f' % self.magnitude(self.update_ratio), print_message = True)


    def log_path(self, game, network, run_name):
        return "logs_mpi/%s/ES/%s/%s" % \
               (game, network, run_name)



