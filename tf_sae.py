import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Generate training set
batch_xs, batch_ys = mnist.train.next_batch(10000)

crop_size = 8

t_batch_xs = []

for i, imvec in enumerate(batch_xs):
    img = imvec.reshape(28,28)
    crop_x = np.random.randint(0, 28 - crop_size)
    crop_y = np.random.randint(0, 28 - crop_size)
    cimg = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size].flatten()

    t_batch_xs.append(cimg)

t_batch_xs = np.asarray(t_batch_xs)
batch_xs = tf.Variable(t_batch_xs)

print(batch_xs)
# sess = tf.InteractiveSession()