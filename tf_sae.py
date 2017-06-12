import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# params
num_samples = 50000
learning_rate = 0.01
sparse_reg = 3
sparse_param = 0.01
l2_reg = 0.0001
epochs = 20
batch_size = 256
n_input = 28 ** 2
n_hidden_1 = 256

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
	p = np.repeat([sparse_param], n_hidden_1).astype(np.float32)
	phat = tf.reduce_mean(a, 0)
	return tf.reduce_sum(p * tf.log(p) - p * tf.log(phat) + (1-p) * tf.log(1 - p) - (1 - p) * tf.log(1 - phat))

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Generate training set
batch_xs, batch_ys = mnist.train.next_batch(num_samples)

X = tf.placeholder(tf.float32, [None, n_input])

encode_op = encoder(X)
decode_op = decoder(encode_op)
	
y_pred = decode_op
y_true = X

activation = encode_op
kl_div = kl_divergence(activation)
cost_function = tf.reduce_sum(tf.pow(y_true - y_pred, 2)) / num_samples # + sparse_reg * kl_div + l2_reg * (tf.reduce_sum(tf.square(Wd)) + tf.reduce_sum(tf.square(We)))
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

	plt.figure(figsize = (8,8))
	gs = gridspec.GridSpec(8,8)
	gs.update(wspace = 0.025, hspace=0.025)

	for i in range(64):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		plt.imshow(We[:,i].eval().reshape(28, 28), cmap='gray', interpolation='none')
	plt.show()