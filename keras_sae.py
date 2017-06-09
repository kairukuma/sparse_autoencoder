import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

(x_train, _), (x_test, _) = mnist.load_data()

encode_dim = 32

input_img = Input(shape=(784,))  # adapt this if using `channels_first` image data format

encoded = Dense(encode_dim, activation='relu',
	activity_regularizer=regularizers.l1(1e-4))(input_img)
decoded = Dense(28**2, activation = 'sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encode_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

print(autoencoder.summary())

autoencoder.fit(x_train, x_train,
	epochs=100,
	batch_size=256,
	shuffle=True,
	validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.mean())

enc_W = encoder.get_weights()[0]

n = 32
plt.figure()

for i in range(n):
	ax = plt.subplot(4, 8, i + 1)
	plt.imshow(enc_W[:,i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()