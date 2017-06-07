import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# params
num_samples = 10000
crop_size = 8
learning_rate = 0.01
sparse_reg = 0.01
epochs = 20
batch_size = 32
n_input = crop_size ** 2
n_hidden_1 = 25

We = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
be = tf.Variable(tf.random_normal([n_hidden_1]))

Wd = tf.Variable(tf.random_normal([n_hidden_1, n_input]))
bd = tf.Variable(tf.random_normal([n_input]))

def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,We), be))

	return layer_1

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,Wd), bd))

	return layer_1

def kl_divergence(a):
	s = np.repeat([0.05], n_hidden_1).astype(np.float32)
	return s * tf.log(s) - s * tf.log(a) + (1-s) * tf.log(1 - s) - (1 - s) * tf.log(1 - a)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Generate training set
batch_xs, batch_ys = mnist.train.next_batch(num_samples)

t_batch_xs = []

for i, imvec in enumerate(batch_xs):
    img = imvec.reshape(28,28)
    crop_x = np.random.randint(0, 28 - crop_size)
    crop_y = np.random.randint(0, 28 - crop_size)
    cimg = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size].flatten()

    t_batch_xs.append(cimg)

batch_xs = np.asarray(t_batch_xs)

X = tf.placeholder(tf.float32, [None, n_input])

encode_op = encoder(X)
decode_op = decoder(encode_op)
	
y_pred = decode_op
y_true = X

activation = encode_op
kl_div = kl_divergence(activation)
cost_function = tf.reduce_sum(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	total_batch = int(num_samples / batch_size)
	batch_sequence = np.random.permutation(num_samples)

	for epoch in range(epochs):
		for i in range(total_batch):
			batch_indices = batch_sequence[batch_size * i: batch_size * (i+1)]
			batch_x = batch_xs[batch_indices,:]
			_, c = sess.run([optimizer, cost_function], feed_dict={X: batch_x})

		if epoch % 5 == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

	print("Optimization finished.")

	plt.figure(figsize = (5,5))
	gs = gridspec.GridSpec(5, 5)
	gs.update(wspace = 0.025, hspace=0.025)
	
	for i in range(25):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		plt.imshow(We[:,i].eval().reshape(crop_size, crop_size), cmap='gray', interpolation='none')
	plt.show()