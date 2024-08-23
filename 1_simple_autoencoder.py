import keras
from keras import layers
from keras.datasets import mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model # type: ignore


'''ref: https://blog.keras.io/building-autoencoders-in-keras.html
a single fully-connected neural layer as encoder and as decoder
'''
''' Changes Made:
1 - Intermediate Reconstructions: Added code to visualize intermediate outputs from the layer just before the final reconstruction.
2 - Loss Plot: Added a plot to visualize training and validation loss over epochs.
3 - Encoded Representations: If the encoding dimension is 2D or 3D, the code visualizes the encoded data in a scatter plot.
'''


# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats



# This is our input image
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

'''
a separate encoder model
'''

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

'''
a separate decoder model
'''
# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

'''
training our autoencoder to reconstruct MNIST digits
'''
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Plotting loss curves
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Intermediate layer visualization
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.layers[-2].output)  # Outputs from the layer before final layer

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display intermediate reconstruction
    intermediate_output = intermediate_layer_model.predict(np.array([x_test[i]]))
    ax = plt.subplot(3, n, i + 1 + n)
    plt.bar(range(encoding_dim), intermediate_output[0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display final reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Visualizing encoded representations (if encoding_dim is low)
if encoding_dim <= 3:
    plt.figure(figsize=(8, 6))
    if encoding_dim == 2:
        plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c='blue')
        plt.title('2D Encoded Representations')
    elif encoding_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c='blue')
        ax.set_title('3D Encoded Representations')
    plt.show()

print(encoded_imgs.mean())

# Calculate train loss
train_loss = autoencoder.evaluate(x_train, x_train, verbose=0)
print(f'Train Loss: {train_loss}')

# Calculate test loss
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f'Test Loss: {test_loss}')


