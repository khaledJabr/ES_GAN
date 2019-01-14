import pickle
import os
import numpy as np


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.plot_dir = log_dir +'/iterations'
        self.parameters_dir = log_dir + '/parameters'
        self.vb_dir = log_dir + '/vb'
        self.stats_dir = log_dir + '/stats'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(self.plot_dir): 
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.parameters_dir): 
            os.makedirs(self.parameters_dir)
        if not os.path.exists(self.vb_dir): 
            os.makedirs(self.vb_dir)
        if not os.path.exists(self.stats_dir): 
            os.makedirs(self.stats_dir)


    def log(self, message , print_message =False):
        with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
            f.write(message+"\n")
        if print_message : 
            print(message)

    def write_general_stat(self, stat_string):
        with open(os.path.join(self.log_dir, "stat.txt"), "a") as f:
            f.write(stat_string)

    def write_optimizer_stat(self, stat_string):
        if stat_string is not None:
            with open(os.path.join(self.log_dir, "optimizer_stat.txt"), "a") as f:
                f.write(stat_string)

    def save_parameters(self, parameters, params_type, iteration ):
        with open(os.path.join(self.parameters_dir, "%s_parameters_%d" % (params_type, iteration)), 'wb') as f:
            pickle.dump({"parameters": parameters}, f)

    def save_vb(self, vb):
        np.save(os.path.join(self.vb_dir, "vb.npy"), vb)

    def save_stats(self, stat , stat_label) : 
        np.save(os.path.join(self.stats_dir, "{}.npy".format(stat_label)), stat)


    # def save_np_samples(self, np_samples):
    #     np.save(os.path.join(self.np_samples_dir, "np_samples.npy"), np_samples)

    # def save_bc_samples(self, bc_samples):
    #     np.save(os.path.join(self.np_samples_dir, "bc_samples.npy"), bc_samples)
