# AutoEncoders

This repository serves as a collection of slides, code samples, and resources put together for "Coding Gen AI from scratch - Session 3 : Variational Autoencoders (VAEs)". 
Our goal is to provide a comprehensive set of materials to help group members learn and experiment with different generative methods.

## Contents

1. simple_autoencoder.py
2. sparse_autoencoder.py
3. deep_autoencoder.py
4. CNN_autoencoder.py
5. image_denoising.py
6. VAE.py
7. autoencoder_pytorch.py
8. requirements.txt
9. presentationSlides.pdf  


## Descriptions

1. This code implements a simple autoencoder using Keras, which compresses and reconstructs MNIST digit images. It trains the model, visualizes the original and reconstructed images, and evaluates the reconstruction loss on the training and test datasets.
2. This code demonstrates the addition of a sparsity constraint to an autoencoder's encoded representations by using an L1 activity regularizer in Keras. The model is trained on the MNIST dataset, with a focus on reducing the average activation of the hidden units, thereby producing sparser encoded representations compared to a standard autoencoder.
3. This code implements a deep autoencoder using Keras, consisting of multiple fully-connected layers for both the encoder and decoder. The model compresses input data into a 32-dimensional representation and then reconstructs it. The autoencoder is trained on the MNIST dataset, and the code includes visualization of the original and reconstructed images, as well as evaluation of the model's performance on training and test datasets.
4. This code implements a convolutional autoencoder using Keras, designed to compress and reconstruct images from the MNIST dataset. The model uses convolutional layers for encoding and decoding, with max-pooling and upsampling operations. The TensorBoard callback is included for visualizing training progress and metrics, and the code visualizes both the original and reconstructed images as well as the encoded representations.
5. This code demonstrates the use of a convolutional autoencoder for image denoising. The model is trained to map noisy images of handwritten digits from the MNIST dataset to their clean counterparts. Gaussian noise is added to the training and testing images, and the autoencoder learns to reconstruct the original, noise-free images from these noisy inputs. The code also includes visualization of both the noisy and denoised images.
6. This code implements a Variational Autoencoder (VAE) using TensorFlow and Keras, designed to generate and visualize new samples from the Fashion MNIST dataset. The VAE architecture includes a custom sampling layer for latent space representation and separate encoder and decoder networks. The model is trained to reconstruct images while minimizing both reconstruction loss and KL divergence. It includes functions for visualizing the latent space and clustering images based on their latent representations.
7. This code defines a simple autoencoder using PyTorch to compress and reconstruct images from the MNIST dataset. The model consists of a fully connected encoder and decoder, with the encoder reducing the dimensionality to 9 and the decoder reconstructing the original 784-dimensional images. The training process minimizes the mean squared error between the input and the reconstructed images, and the code includes visualization of both the loss over time and the original versus reconstructed images.
8. This file lists the dependencies needed for our project, including Keras, Matplotlib, NumPy, TensorFlow (version 2.0 or higher), PyTorch, TorchVision, and scikit-learn.

## Getting Started

To get started, clone this repository and explore the slides and code samples.
Feel free to explore, modify, and experiment with the code.

`git clone https://github.com/miladtajvidi/AutoEncoders.git`


## Additional Resources

1. Intro to Deep Learning and Generative Models Course by Sebastian Raschka
URL: https://youtube.com/playlist?list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&si=f6eyKXaq8Oo7glna


## References

1. Generative Deep Learning with TensorFlow by DeepLearning.AI
URL: https://www.coursera.org/learn/generative-deep-learning-with-tensorflow

2. Building Autoencoders in Keras
URL: https://blog.keras.io/building-autoencoders-in-keras.html 

3. Variational Autoencoders by GeeksforGeeks
URL: https://www.geeksforgeeks.org/variational-autoencoders/



