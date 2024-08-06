'''
Adding a sparsity constraint on the encoded representations
In the previous example, the representations were only constrained by the 
size of the hidden layer (32). In such a situation, what typically happens is
that the hidden layer is learning an approximation of PCA (principal component analysis).
But another way to constrain the representations to be compact is to add a sparsity contraint
on the activity of the hidden representations, so fewer units would "fire" at a given time.
In Keras, this can be done by adding an activity_regularizer to our Dense layer:
'''
import keras
from keras import layers
from keras import regularizers
from keras.datasets import mnist

import matplotlib.pyplot as plt

import numpy as np


encoding_dim = 32

input_img = keras.Input(shape=(784,))
# Add a Dense layer with a L1 activity regularizer
encoded = layers.Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_img, decoded)

encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print(encoded_imgs.mean())

# Calculate train loss
train_loss = autoencoder.evaluate(x_train, x_train, verbose=0)
print(f'Train Loss: {train_loss}')

# Calculate test loss
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f'Test Loss: {test_loss}')


'''
They look pretty similar to the previous model, the only significant
difference being the sparsity of the encoded representations. encoded_imgs.mean()
yields a value 3.33 (over our 10,000 test images), whereas with the previous model
the same quantity was 7.30. So our new model yields encoded representations that are
twice sparser.
'''

