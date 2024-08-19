<div style="display: flex; justify-content: center;">
  <img src="./images/DL_Logo.png" alt="Alt text 1" width="100" height="100">
  <img src="./images/g4g_Logo.png" alt="Alt text 2" width="100" height="100">
  <img src="./images/keras_Logo.png" alt="Alt text 3" width="100" height="100">
</div>


# AutoEncoders (from simple to variational)

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
9. presentationSlides2.pdf
10. VAE_dist.py
11. utils_VAE.py   


## Descriptions

1. This code implements a simple autoencoder using Keras, which compresses and reconstructs MNIST digit images. It trains the model, visualizes the original and reconstructed images, and evaluates the reconstruction loss on the training and test datasets.
------------------------------------------------------------------------------------
2. This code demonstrates the addition of a sparsity constraint to an autoencoder's encoded representations by using an L1 activity regularizer in Keras. The model is trained on the MNIST dataset, with a focus on reducing the average activation of the hidden units, thereby producing sparser encoded representations compared to a standard autoencoder.
------------------------------------------------------------------------------------
3. This code implements a deep autoencoder using Keras, consisting of multiple fully-connected layers for both the encoder and decoder. The model compresses input data into a 32-dimensional representation and then reconstructs it. The autoencoder is trained on the MNIST dataset, and the code includes visualization of the original and reconstructed images, as well as evaluation of the model's performance on training and test datasets.
------------------------------------------------------------------------------------
4. This code implements a convolutional autoencoder using Keras, designed to compress and reconstruct images from the MNIST dataset. The model uses convolutional layers for encoding and decoding, with max-pooling and upsampling operations. The TensorBoard callback is included for visualizing training progress and metrics, and the code visualizes both the original and reconstructed images as well as the encoded representations.
------------------------------------------------------------------------------------
5. This code demonstrates the use of a convolutional autoencoder for image denoising. The model is trained to map noisy images of handwritten digits from the MNIST dataset to their clean counterparts. Gaussian noise is added to the training and testing images, and the autoencoder learns to reconstruct the original, noise-free images from these noisy inputs. The code also includes visualization of both the noisy and denoised images.
------------------------------------------------------------------------------------
6. This code implements a Variational Autoencoder (VAE) using TensorFlow and Keras, designed to generate and visualize new samples from the Fashion MNIST dataset. The VAE architecture includes a custom sampling layer for latent space representation and separate encoder and decoder networks. The model is trained to reconstruct images while minimizing both reconstruction loss and KL divergence. It includes functions for visualizing the latent space and clustering images based on their latent representations.
------------------------------------------------------------------------------------
7. This code defines a simple autoencoder using PyTorch to compress and reconstruct images from the MNIST dataset. The model consists of a fully connected encoder and decoder, with the encoder reducing the dimensionality to 9 and the decoder reconstructing the original 784-dimensional images. The training process minimizes the mean squared error between the input and the reconstructed images, and the code includes visualization of both the loss over time and the original versus reconstructed images.
------------------------------------------------------------------------------------
8. This file lists the dependencies needed for our project, including Keras, Matplotlib, NumPy, TensorFlow (version 2.0 or higher), PyTorch, TorchVision, and scikit-learn.
------------------------------------------------------------------------------------
9. Presentation Slides
------------------------------------------------------------------------------------
10. This code implements a Variational Autoencoder (VAE) using PyTorch, designed to model a custom probability distribution and generate new samples. The VAE architecture consists of an encoder that maps input data to a latent space, and a decoder that reconstructs the input from the latent variables. The reparameterization trick is used to sample from the latent space during training. The model is trained by minimizing a combination of reconstruction loss (MSE) and KL divergence. The training loop iterates over a custom dataset of randomly generated numbers, and the code includes functions to visualize the distribution of both original and generated samples.
------------------------------------------------------------------------------------
11. This code generates random samples from a custom probability distribution function (PDF) and visualizes the resulting data using Matplotlib. The custom PDF is a mixture of three Gaussian distributions with added sinusoidal modulation. The code includes functions to normalize the PDF, generate random numbers using rejection sampling, and plot the generated samples against the PDF. It is designed to create and display the distribution of the generated data alongside the theoretical PDF to assess the accuracy of the sampling process.

## Getting Started

To get started, clone this repository and explore the slides and code samples.
Feel free to explore, modify, and experiment with the code.

`git clone https://github.com/miladtajvidi/AutoEncoders.git`


## Additional Resources

1. [Youtube Tutorials] Intro to Deep Learning and Generative Models Course by Sebastian Raschka (chapters L16.0 - L17.7)<br>
URL: https://youtube.com/playlist?list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&si=f6eyKXaq8Oo7glna<br>
Source Codes URL: https://github.com/rasbt/stat453-deep-learning-ss21

2. [Coursera Guided Project] Image Compression and Generation using Variational Autoencoders in Python<br>
URL: https://www.coursera.org/projects/image-compression-generation-vae

3. [TowardsDataScience Article] Intuitively Understanding Variational Autoencoders<br>
URL: https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

4. [Textbook] Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. MIT Press, 2016<br>
chapter 14: Autoencoders <br>
chapter 20: Deep Generative Models


## References

1. Generative Deep Learning with TensorFlow by DeepLearning.AI<br>
URL: https://www.coursera.org/learn/generative-deep-learning-with-tensorflow

2. Building Autoencoders in Keras<br>
URL: https://blog.keras.io/building-autoencoders-in-keras.html 

3. Variational Autoencoders by GeeksforGeeks<br>
URL: https://www.geeksforgeeks.org/variational-autoencoders/

4. Variational Autoencoders by Paul Hand
URL: https://www.youtube.com/watch?v=c27SHdQr4lw



