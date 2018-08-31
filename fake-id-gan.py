import os, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
import csv

config = tf.ConfigProto(device_count = {'GPU': 0})
batch_size = 64
z_size = 100
train_epoch = 10000
training = True
discriminator_learning_ratio = 0.08  # compare to G
start_learning_rate = 0.001

class RealIds():
    
    def __init__(self):
        self.ids = []
        self.cursor = 0
        self.wordLenLimit = 40
        self.load()
        self.vectorSize1D = len(self.all_chars) * self.wordLenLimit
        
    def load(self):
        chars ={}
        reader = csv.reader(open('real_ids.csv', newline=''), delimiter=' ', quotechar='|')
        for row in reader:
            self.ids.append(row[0])
            id = row[0]
            for char in list(id):
                chars[char] = True

        # print(ids)
        self.all_chars = list(chars.keys())
        self.all_chars.sort()        
        self.all_chars = [''] + self.all_chars
        self.iMat = np.identity(len(self.all_chars))
        

    def wordToMatrix(self, word):
        # print("len(self.all_chars)", len(self.all_chars))
        mat = np.zeros([self.wordLenLimit, len(self.all_chars)])
        i = 0
        offset = np.random.randint(0, self.wordLenLimit-len(word)) 
        for char in list(word):
            i = i+1
            idx = self.all_chars.index(char)
            mat[offset+i, : ] = self.iMat[idx]
        return mat
    
    def matrixToWord(self, mat):
        idxses = np.argmax(mat, axis=1)        
        chars = []
        for idx in idxses:            
            chars.append(self.all_chars[idx])
        return ''.join(chars)

    def get_batch(self, num):    
        items = []
        if self.cursor + num > len(self.ids):
            items = self.ids[self.cursor:]
            items = items + self.ids[0 : (self.cursor+num) % len(self.ids)]
            self.cursor = (self.cursor+num) % len(self.ids)
        else:
            items = self.ids[self.cursor:self.cursor+num]
            self.cursor = self.cursor+num
        return np.array([self.wordToMatrix(x) for x in items])
        


# test RealIds class

realIds = RealIds()
matrix = realIds.wordToMatrix('apple')

print('apple as matrix:')
for r in matrix:
    print("".join([str(int(x)) for x in list(r)]))
    
word = realIds.matrixToWord(matrix)
print('matrix as word: ', word)
print("batch taking test")
for i in range(2000):
    v = realIds.get_batch(200)
    if i % 300 == 0:
        print(v.shape)


w_init = tf.contrib.layers.xavier_initializer()

def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        layer = tf.layers.dense(z, 1000, activation=tf.nn.relu, kernel_initializer=w_init)
        #layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, 1000, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, realIds.vectorSize1D, activation=None, kernel_initializer=w_init)
        g_d = tf.nn.tanh(layer)
        return g_d

    
def discriminator( x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        layer = tf.layers.dense(x, realIds.vectorSize1D, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 256, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 64, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        logits = tf.layers.dense(layer, 1, activation=None, kernel_initializer=w_init)
        d = tf.nn.sigmoid(logits)
        return d, logits

#z = tf.random_normal(shape=(batch_size, z_size), mean=0.0, stddev=1.0, dtype=tf.float32)
z = tf.random_uniform(shape=(batch_size, z_size), dtype=tf.float32)
x = tf.placeholder(shape=[batch_size, realIds.wordLenLimit, len(realIds.all_chars)], dtype=tf.float32) # for mnist
x_reshaped  = tf.reshape(x, [batch_size, realIds.vectorSize1D])
x_fake = generator(z, reuse=False)

y_fake, logits_fake = discriminator(x_fake, False)
y_real, logits_real = discriminator(x_reshaped, True)

#loss_d = -(tf.reduce_mean(tf.log(y_real)) + tf.reduce_mean(tf.log(1-y_fake)))
#loss_g = tf.reduce_mean(tf.log(1-y_fake))

# To remove sigmoid from backpropagation process, use logit and sigmoid_cross_entropy

label_one = tf.ones_like(logits_real)
label_zero = tf.zeros_like(logits_fake)
loss_d = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_real)
loss_g = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_real)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

    d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
    
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
            #if np.random.random() < discriminator_learning_ratio:
            _loss_g = resluts[1]
            _loss_d = resluts[2]
            if np.random.random() < discriminator_learning_ratio:
                _ = sess.run([d_train], {x: x_data})
            
        if epoch % 20 == 0:
            _summ = sess.run(merged, {x: x_data})
            writer.add_summary(_summ, epoch)
            resluts = sess.run([x_fake, loss_g, loss_d, learning_rate], {x: x_data})
            _fake = resluts[0]
            _loss_g = resluts[1]
            _loss_d = resluts[2]
            _learning_rate = resluts[3]
            testFakeId = realIds.matrixToWord(_fake[0].reshape([realIds.wordLenLimit, len(realIds.all_chars)]))
            print("epoch: %05d  lossG: %10.3f  lossD: %10.3f  fake id sample: %s"%(epoch, _loss_g, _loss_d, testFakeId))


    print("Done!!")