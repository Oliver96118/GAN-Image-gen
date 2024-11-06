
import tensorflow as tf
import numpy as np
import pathlib


# This script is used for training the GAN for generating the images
# The images that the GAN generates are 192x108


# Defining preprocess_images, used to convert images from integers to floating points
def preprocess_images(dataset):
    dataset = (dataset - 127.5) / 127.5
    return dataset


# Defining process_images, uses the generator model of the GAN
# to synthesize new images
def process_images(g_model, latent_dim, n_samples, savepath):
    images_generated_X, images_generated_y = generate_fake_samples(g_model, latent_dim, n_samples)
    for i in range(0, n_samples):
        tf.keras.preprocessing.image.save_img(savepath + str(i+1) + '.jpg', images_generated_X[i])


# Defining generate_real_samples, which randomly selects from the
# training images dataset (all samples) for the discriminator model to train on
def generate_real_samples(dataset, n_samples):
    # Generate an iterator over n_samples within the dataset
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # Store these randomly selected samples in X
    X = dataset[ix]
    # Since these are real samples, we will label all of these examples with label 1
    y = np.ones((n_samples, 1))
    return X, y


# Defining generate_latent_samples, used for making the latent vector (for
# generating fake samples)
def generate_latent_points(latent_dim, n_samples):
    # Generate n_samples in the latent_dim dimensional latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # Reshape this into a matrix
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# Defining generate_fake_samples, used for generating fake images with the generator model
def generate_fake_samples(g_model, latent_dim, n_samples):
    # First, generate n_samples in the latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Use the generator to synthesize a new image given x_input
    X = g_model.predict(x_input)
    # Since these are fake samples, we will label all of these examples with label 0
    y = np.zeros((n_samples, 1))
    return X, y


# Defining make_discriminator, which defines our discriminator model
# Input is a 192x108 image (made by the generator), and output is single value
def make_discriminator(in_shape=(108, 192, 3)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', input_shape=in_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Conv2D(256, (6, 6), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Conv2D(256, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Defining make_generator, which defines our generator model
# input is the latent dimension, and output is a 192x108 image
def make_generator(latent_dim):
    n_nodes = 4096 * 27 * 48
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Reshape((27, 48, 4096)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                              padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                              padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# Defining make_GAN, which defines our GAN model
# This model combines our discriminator and generator models in the
# context of a GAN network
def make_GAN(g_model, d_model):
    # Turn off modification of discriminator weights while in GAN model
    # This is so that the discriminator doesn't overtrain on fake examples
    d_model.trainable = False
    model = tf.keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# Defining train, which is used for training our GAN model
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=8000, n_batch=10):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # Loop over n_epochs of the dataset
    for i in range(n_epochs):
        # Loop over batches within the dataset
        for j in range(bat_per_epo):
            # Get randomly selected real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # Train the discriminator on the real samples
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # Generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Train the discriminator on the fake samples
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # Prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # The labels for the GAN data points will be all 1 to
            # try and trick the discriminator
            y_gan = np.ones((n_batch, 1))
            # Train the generator on what the discriminator observed
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # Evaluate the model performance after 10 epochs
        if (i+1) % 10 == 0:
            summarize_performance(i+1, g_model, d_model, dataset, latent_dim)


# Defining summarize_performance, used to print model performance to the user during training
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=5):
    # Prepare real samples from the training dataset
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # Evaluate the discriminator performance on the real dataset
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # Generate some fake samples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # Evaluate the discriminator performance on the fake dataset
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%, epoch: %d' % (acc_real*100, acc_fake*100, epoch))


# Defining the latent dimension and our models
latent_dim = 4
d_model = make_discriminator()
g_model = make_generator(latent_dim)
gan_model = make_GAN(g_model, d_model)
d_model.summary()
g_model.summary()

# Counting the number of images in sampleimages_lowres
# These images are used as real examples for training the GAN
count = 0
for path in pathlib.Path("sampleimages_lowres").iterdir():
    if path.is_file():
        count += 1
print(count)

# Loading in the images from the sampleimages_lowres directory
training_images = []
training_images_list = []
for i in range(1, count+1):
    current_image = tf.keras.preprocessing.image.load_img("sampleimages_lowres/" + str(i) + ".jpg")
    current_image_array = tf.keras.preprocessing.image.img_to_array(current_image)
    training_images.append(current_image)
    training_images_list.append(current_image_array)
training_images_array = np.asarray(training_images_list)
training_images_array = preprocess_images(dataset=training_images_array)

# Training the models
train(g_model, d_model, gan_model, training_images_array, latent_dim)
# Saving the trained models
g_model.save('models/landgan_model/g_model')
d_model.save('models/landgan_model/d_model')
gan_model.save('models/landgan_model/gan_model')
