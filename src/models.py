# from tensorflow.contrib.layers import conv2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import batch_norm as bn
# from . import ops as U # U for utilities
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Some configuration parameters for convolution and dense layers
# This will avoid unnecessary repeats
# Be careful when you change those values as many models will overwrite
# some fields
conv_args = {
    "padding": "VALID",
    "biases_initializer": None,
    "activation_fn": None,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}
dense_args = {
    "biases_initializer": None,
    "activation_fn": None,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}

bn_args = {
    "decay": 0.,
    "center": True,
    "scale": True,
    "epsilon": 1e-8,
    "activation_fn": tf.nn.relu,
    "is_training": False
}

nonlin_dict = { 
    "tanh" : tf.nn.tanh, 
    "lrelu" : tf.nn.leaky_relu
}

def simpleGAN(gan_args, dataset_params) : # gan used for 2d gaussian stuff. Hardcoded
    # weights_initializer= tf.random_normal_initializer(0, 0.05)
    # initializer=tf.random_normal_initializer(stddev=stddev)
    # initializer=tf.contrib.layers.xavier_initializer()
    # tf.orthogonal_initializer(gain=1.4)

    def linear(input, output_dim, scope='linear', stddev=0.01):
        with tf.variable_scope(scope):
            w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.matmul(input, w) + b


    # def generator(z, discretize ,  is_training,  n_hidden=128 , output_dim=2,):
            # with tf.variable_scope("Generator" ):
            #     # bn_args["is_training"] = is_training
            #     # hidden = tf.nn.relu(linear(z, n_hidden, 'g_hidden1'))
            #     # hidden = tf.nn.relu(linear(hidden, n_hidden, 'g_hidden2'))

            #     # # discritize to encourage exploration
            #     # if discretize : 
            #     #     num_ac_bins = 10
            #     #     # aidx_na = bins(x, output_dim, num_ac_bins, 'out')
            #     #     hidden_disc = linear(hidden, n_hidden * num_ac_bins, 'g_hidden3')
            #     #     hidden_disc = tf.reshape(hidden_disc, [-1, n_hidden, num_ac_bins])
            #     #     hidden_disc = tf.argmax(hidden_disc, 2)
            #     #     # x = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
            #     #     hidden_disc = 1. / (num_ac_bins - 1.) * tf.to_float(hidden_disc)
            #     #     out = linear(hidden_disc, output_dim , 'g_out')
            #     #     out += tf.random_uniform(shape = tf.shape(out)) * 0.01

            #     # else : 
            #     #     out = linear(hidden, output_dim , 'g_out')

            #     # # add random noise to encourge exploration 
            #     # # x += tf.random_uniform(shape = tf.shape(x)) * 0.01
               
            # return out 
    def generator(z, discretize , is_training,  n_hidden=128 , output_dim=2, n_layer=3):
            with tf.variable_scope("Generator"):
                bn_args["is_training"] = is_training
                h1 = slim.fully_connected(z, n_hidden,   activation_fn=tf.nn.leaky_relu)
                #h1 = bn(h1, **bn_args)
                h2 = slim.fully_connected(h1, n_hidden, activation_fn=tf.nn.leaky_relu)
                #h2 = bn(h2, **bn_args)
  

                # discritize to encourage exploration
                if discretize : 
                    num_ac_bins = 10
                    # aidx_na = bins(x, output_dim, num_ac_bins, 'out')
                    x = slim.fully_connected(h2, n_hidden * num_ac_bins)
                    x = tf.reshape(x, [-1, n_hidden, num_ac_bins])
                    x = tf.argmax(x, 2)
                    # x = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
                    x = 1. / (num_ac_bins - 1.) * tf.to_float(x)

                    x = slim.fully_connected(x, output_dim,  activation_fn=None)

                else : 
                    x = slim.fully_connected(h2, output_dim,  activation_fn=None)

                # add random noise to encourge exploration 
                # x += tf.random_uniform(shape = tf.shape(x)) * 0.05
               
            return x 

    # def discriminator(x, is_training, n_hidden=128, output_dim = 1, reuse=False):
    #     with tf.variable_scope("Discriminator", reuse=reuse):
    #         bn_args["is_training"] = is_training
    #         hidden = tf.nn.leaky_relu(linear(x, n_hidden, 'd_hidden1'))
    #         hidden = tf.nn.leaky_relu(linear(hidden, n_hidden, 'd_hidden2'))
    #         out = linear(hidden, 1 , 'd_out')
    #         # out = tf.nn.softplus()

    #     return out , hidden
    def discriminator(x,  is_training, n_hidden=128, n_layer=2 , output_dim = 1, reuse=False):
            with tf.variable_scope("Discriminator", reuse=reuse):
                bn_args["is_training"] = is_training
                h1 = slim.fully_connected(x, n_hidden,activation_fn=tf.nn.leaky_relu)
                h2 = slim.fully_connected(h1,n_hidden,activation_fn=tf.nn.leaky_relu)

                log_d = slim.fully_connected(h2, 1 , activation_fn=None)
            return log_d , h2

          

    z_dim = gan_args['args']['z_dim']
    batch_size = gan_args['args']['batch_size']
    discretize = True if gan_args['args']['discretize_generator'] == 'True' else False
    
    real_dim = 2 
    z_shape = (None , z_dim)
    real_data_shape = (None , real_dim)
    gen_nonlin = nonlin_dict[gan_args['args']['nonlin_g']]
    disc_nonlin = nonlin_dict[gan_args['args']['nonlin_g']]
    print("###### Z_DIM : {} | discretize : {} | g_nonlin :{} | d_nonlin : {} ".format(z_dim, discretize , gen_nonlin, disc_nonlin))



    is_training = tf.placeholder(tf.bool ,[],  name='is_training')
    input_placeholder_Z = tf.placeholder(tf.float32, list(z_shape),name='z_input')
    input_placeholder_X = tf.placeholder(tf.float32, list(real_data_shape), name='real_input')

    g_sample = generator(input_placeholder_Z , discretize, is_training)
    r_logits , _  = discriminator(input_placeholder_X, is_training)
    f_logits , d_features = discriminator(g_sample ,is_training , reuse=True)

    # Setting up loss

    # Discriminator
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels= tf.ones_like(r_logits)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits,  labels = tf.zeros_like(f_logits)))
    D_loss = -(D_loss_real + D_loss_fake) 

    # Generator
    if gan_args['args']['loss'] == 'NN' :  # Non-Saturating Loss
        G_loss=  - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits, labels = tf.ones_like(f_logits)))
    elif gan_args['args']['loss'] == 'MINMAX'  :# Original MINMAX loss
        G_loss = D_loss_fake # ES will maximize this

    elif gan_args['args']['loss'] == 'RA_LOSS' :  # relativistic gan loss
        g_1_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits , labels=tf.zeros_like(r_logits) ))
        g_2_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits , labels=tf.ones_like(f_logits)) )
        G_loss = -(g_1_real_loss + g_2_fake_loss)

    else : 
        raise NotImplementedError(gan_args['args']['g_loss'])


    return [D_loss,G_loss,g_sample,input_placeholder_X,input_placeholder_Z, d_features,D_loss_real,D_loss_fake,is_training]
