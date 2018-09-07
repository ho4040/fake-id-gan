import os, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
from real_data import RealIds

config = tf.ConfigProto(device_count = {'GPU': 0})
batch_size = 64
z_size = 100
train_epoch = 100
training = True
discriminator_learning_ratio = 0.05  # compare to G
start_learning_rate = 0.001

realIds = RealIds()

w_init = tf.contrib.layers.xavier_initializer()

def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        layer = tf.layers.dense(z, 512, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, 512, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, 512, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, realIds.vectorSize1D, activation=None, kernel_initializer=w_init)
        g_d = tf.nn.relu(layer)
        return g_d

    
def discriminator( x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        layer = tf.layers.dense(x, realIds.vectorSize1D, activation=tf.nn.leaky_relu, kernel_initializer=w_init)        
        
        # Add some constraint
        x_ = tf.reshape(x, [-1, realIds.wordLenLimit, len(realIds.all_chars), 1])
        x_hist = tf.contrib.layers.flatten(tf.reduce_sum(x_, axis=1))# histogram of character
        x_len = tf.contrib.layers.flatten(tf.reduce_sum(x_, axis=2))# density of word
        layer = tf.concat([layer, x_len, x_hist], 1)
        
        layer = tf.layers.dense(layer, 512, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 256, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 64, activation=tf.nn.leaky_relu, kernel_initializer=w_init) 
        
        logits = tf.layers.dense(layer, 1, activation=None, kernel_initializer=w_init)
        d = tf.nn.sigmoid(logits)
        return d, logits

def train():
  z = tf.random_uniform(shape=(batch_size, z_size), dtype=tf.float32)
  x = tf.placeholder(shape=[batch_size, realIds.wordLenLimit, len(realIds.all_chars)], dtype=tf.float32) 
  x_reshaped  = tf.reshape(x, [batch_size, realIds.vectorSize1D])
  x_fake = generator(z, reuse=False)
  y_fake, logits_fake = discriminator(x_fake, False)
  y_real, logits_real = discriminator(x_reshaped, True)
  label_one = tf.ones_like(logits_real)
  label_zero = tf.zeros_like(logits_fake)
  loss_d = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_real)
  loss_g = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_real)


  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):

      d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
      g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 50000, 0.96, staircase=True)
      
      # optimize D
      d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
      d_train = d_opt.minimize(loss_d, var_list=d_vars, global_step=global_step)

      # optimize G
      g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
      g_train = d_opt.minimize(loss_g, var_list=g_vars, global_step=global_step)

  tf.summary.histogram('y_real', y_real)
  tf.summary.histogram('y_fake', y_fake)
  tf.summary.scalar('loss_d', loss_d)
  tf.summary.scalar('loss_g', loss_g)

  with tf.Session(config=config) as sess:
    print("session loaded")
    
    init = tf.global_variables_initializer()
    sess.run(init)
    print("graph variable initialized")
    
    writer = tf.summary.FileWriter('./fakeIdGAN', sess.graph)
    merged = tf.summary.merge_all()
    
    np.random.seed(int(time.time()))
    
    start_time = time.time()
    numRealIds = len(realIds.ids)
    print('training start at {}'.format(start_time))
    print("total real ids:", numRealIds)
    
    for epoch in range(train_epoch):
        # update discriminator
        
        for i in range(numRealIds // batch_size):        
            x_data = realIds.get_batch(batch_size)
            resluts = sess.run([g_train, loss_g, loss_d], {x: x_data})
            
            if np.random.random() < discriminator_learning_ratio: # prevent discriminator growing too fast.
                _ = sess.run([d_train], {x: x_data})
            
        if epoch % 1 == 0:
            _summ = sess.run(merged, {x: x_data})
            writer.add_summary(_summ, epoch)
            resluts = sess.run([x_fake, loss_g, loss_d, learning_rate], {x: x_data})
            _fake = resluts[0]
            _loss_g = resluts[1]
            _loss_d = resluts[2]
            _learning_rate = resluts[3]
            testFakeId = realIds.matrixToWord(_fake[0].reshape([realIds.wordLenLimit, len(realIds.all_chars)]))
            print("epoch: %05d  l_G: %2.3f   l_D: %2.3f   a: %.4f    fake_sample: %s"%(epoch, _loss_g, _loss_d, _learning_rate, testFakeId))


    print("Done!!")


if __name__ == "__main__":
  train()