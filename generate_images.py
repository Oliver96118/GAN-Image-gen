
import numpy as np
import tensorflow as tf


# This script is used for generating a new series of images
# Before running this script, ensure that a GAN model (specifically
# a generator model) is saved under models/landgan_model/g_model


# Defining generate_latent_points, used to generate the latent vector (input to the generator)
def generate_latent_points(latent_dim, n_samples):
    # Generate n_samples in the latent_dim dimensional latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # Reshape this into a matrix
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# Defining our function for generating fake image samples, which is used by
# both the generator and discriminator
def generate_fake_samples(g_model, latent_dim, n_samples):
    # First, generate n_samples in the latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Use the generator to synthesize a new image given x_input
    X = g_model.predict(x_input)
    # Since these are fake samples, we will label all of these examples with label 0
    y = np.zeros((n_samples, 1))
    return X, y


# Defining our function for generating a new set of images
# The number of images that will be generated is given by n_samples
def process_images(g_model, latent_dim, n_samples, savepath):
    images_generated_X, images_generated_y = generate_fake_samples(g_model, latent_dim, n_samples)
    for i in range(0, n_samples):
        tf.keras.preprocessing.image.save_img(savepath + str(i+1) + '.jpg', images_generated_X[i])


# Load in the existing generator model, and generate new images
# Ensure that latent_dim is the same as that used for training the GAN
latent_dim = 4
g_model = tf.keras.models.load_model('models/landgan_model/g_model')
process_images(g_model, latent_dim, n_samples=200, savepath='generatedimages/')
