"""TensorFlow implementation of the Sparse Autoencoder from the UFLDL tutorisals
"""
import os
import numpy as np
import scipy.io
import tensorflow as tf
import math as math
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATCHWIDTH = 28
N_HIDDEN_1 = 128
N_HIDDEN_2 = 64
N_HIDDEN_3 = 32
N_INPUT = PATCHWIDTH**2
N_OUTPUT = N_INPUT
BETA = tf.constant(5.)
LAMBDA = tf.constant(.0001)
RHO = tf.constant(0.01)
EPSILON = .000001


def train():
    sess = tf.InteractiveSession()
    # LOAD image data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    images = mnist.train.images
    NUMSAMPLES = np.size(images, 0)

    def normalize(unNormalizedTensor):
        normalized = unNormalizedTensor - np.mean(unNormalizedTensor)
        pstd = 3 * np.std(normalized)
        normalized = np.maximum(np.minimum(normalized, pstd),-1*pstd)/pstd
        normalized = (normalized +1) *.4 + 0.1
        return normalized
        
    #normalize the data 
    for i in range(NUMSAMPLES):
        images[i,:] = normalize(images[i,:])

    samples = images

    """
    def displayWeights(weights, blocking=False):
        num_tiles = math.sqrt(N_HIDDEN)
        image = np.zeros((int(num_tiles*PATCHWIDTH + num_tiles+1),int(num_tiles*PATCHWIDTH + num_tiles+1)))
        for i in range(N_HIDDEN):
            subWeights = normalize(weights[:,i])
            subWeights = np.reshape(subWeights,(PATCHWIDTH,PATCHWIDTH))
            denom = np.sqrt(np.dot(subWeights,subWeights))
            subWeights = np.divide(subWeights,denom)
            xIndex = i % num_tiles
            yIndex = i // num_tiles
            xStart = int((xIndex+1)+(xIndex*PATCHWIDTH))
            yStart = int((yIndex+1)+(yIndex*PATCHWIDTH)) 
            image[xStart:int(xStart+PATCHWIDTH),yStart:int(yStart+PATCHWIDTH)] = subWeights
            
        plt.figure(1)
        plt.imshow(image,interpolation='none')      
        plt.draw()
        plt.savefig('./sae_weights.png')"""

    # Input placehoolders
    with tf.name_scope('input'):
        #Construct the tensor flow model
        x = tf.placeholder("float", [None, N_INPUT], name='x-input')
        hidden = tf.placeholder("float", [None, N_HIDDEN_1], name='hidden-activation')

    def autoencoder(X, weights, biases):
        encodeLayer_1 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    X, weights['encode_1']
                ),
                biases['encode_1']
            )
        )

        encodeLayer_2 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    encodeLayer_1, weights['encode_2']
                ),
                biases['encode_2']
            )
        )

        encodeLayer_3 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    encodeLayer_2, weights['encode_3']
                ),
                biases['encode_3']
            )
        )

        decodeLayer_1 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    encodeLayer_3, weights['decode_1']
                ), 
                biases['decode_1']
            )
        )

        decodeLayer_2 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    decodeLayer_1, weights['decode_2']
                ), 
                biases['decode_2']
            )
        )

        decodeLayer_3 = tf.sigmoid(
            tf.add(
                tf.matmul(
                    decodeLayer_2, weights['decode_3']
                ), 
                biases['decode_3']
            )
        )
        return {'encode_1': encodeLayer_1, 
                'encode_2': encodeLayer_2,
                'encode_3': encodeLayer_3,
                'decode_1': decodeLayer_1,
                'decode_2': decodeLayer_2,
                'decode_3': decodeLayer_3}
          
    weights = {
        'encode_1': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1])),
        'encode_2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2])),
        'encode_3': tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_3])),
        'decode_1': tf.Variable(tf.random_normal([N_HIDDEN_3, N_HIDDEN_2])),
        'decode_2': tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_1])),
        'decode_3': tf.Variable(tf.random_normal([N_HIDDEN_1, N_INPUT]))
    }
    
    biases = {
        'encode_1': tf.Variable(tf.random_normal([N_HIDDEN_1])),
        'encode_2': tf.Variable(tf.random_normal([N_HIDDEN_2])),
        'encode_3': tf.Variable(tf.random_normal([N_HIDDEN_3])),
        'decode_1': tf.Variable(tf.random_normal([N_HIDDEN_2])),
        'decode_2': tf.Variable(tf.random_normal([N_HIDDEN_1])),
        'decode_3': tf.Variable(tf.random_normal([N_INPUT]))
    }

    pred = autoencoder(x, weights, biases)

    #Construct cost

    diff = tf.subtract(pred['decode_3'], x)
    
    cost_J = tf.div(tf.nn.l2_loss(diff ),tf.constant(float(NUMSAMPLES)))
    
    costs = {
        'enc1': tf.nn.l2_loss(weights['encode_1']),
        'enc2': tf.nn.l2_loss(weights['encode_2']),
        'enc3': tf.nn.l2_loss(weights['encode_3']),
        'dec1': tf.nn.l2_loss(weights['decode_1']),
        'dec2': tf.nn.l2_loss(weights['decode_2']),
        'dec3': tf.nn.l2_loss(weights['decode_3'])
    }

    cost_reg = tf.multiply(
        LAMBDA , tf.add(tf.add(tf.add(costs['enc1'], costs['enc2']), tf.add(costs['enc3'], costs['dec1'])), tf.add(costs['dec2'], costs['dec3']))
    )
        
    cost = tf.add(cost_J , cost_reg )

    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    sess.run(init)
    plt.gray()
    vizImage = np.random.randint(NUMSAMPLES)
    
    # Training cycle
    c = 0.
    c_old = 1.
    i = 0
    while np.abs(c - c_old) > EPSILON :
        sess.run([optimizer], feed_dict={x: samples})
        if i % 100 == 0:
            c_old = c
            c,j,reg = sess.run([cost,cost_J,cost_reg], feed_dict={x: samples})
            print ("EPOCH %d: COST = %f, LOSS = %f, REG_PENALTY = %f" %(i,c,j,reg))

            saver = tf.train.Saver()
            save_path = saver.save(sess, './model.ckpt')

        i += 1

    print("Optimization Finished!")

def main(_):
    train()
    
if __name__ == '__main__':
  tf.app.run()
        
