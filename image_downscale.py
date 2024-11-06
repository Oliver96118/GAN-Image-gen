
import tensorflow as tf
import numpy as np
import pathlib


# This script is used for downscaling the 1920x1080 images in the
# sampleimages_highres directory to 768x432 and 192x108, which are stored
# in the sampleimages_midres and sampleimages_lowres directories


# Defining make_downscaler_midres, for returning a NN that downscales
# 1920x1080 images to 768x432 images
def make_downscaler_midres(in_shape=(1080, 1920, 3)):
    layer1 = tf.keras.layers.Input(shape=(1080, 1920, 3))
    layer2 = tf.keras.layers.experimental.preprocessing.Resizing(height=432,
                                                                 width=768,
                                                                 interpolation='mitchellcubic',
                                                                 input_shape=in_shape)(layer1)
    model = tf.keras.Model(inputs=layer1, outputs=layer2)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.95)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


# Defining make_downscaler_lowres, for returning a NN that downscales
# 768x432 images to 192x108 images
def make_downscaler_lowres(in_shape=(432, 768, 3)):
    layer1 = tf.keras.layers.Input(shape=(432, 768, 3))
    layer2 = tf.keras.layers.experimental.preprocessing.Resizing(height=108,
                                                                 width=192,
                                                                 interpolation='mitchellcubic',
                                                                 input_shape=in_shape)(layer1)
    model = tf.keras.Model(inputs=layer1, outputs=layer2)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.95)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


# Defining downscale_images_midres, which is used for downscaling the input images
# (loaded from sampleimages_highres) and saving them in sampleimages_midres
def downscale_images_midres(dataset_hr_array, model, count):
    images_dnsc_midres = model.predict(dataset_hr_array)
    for i in range(0, count):
        tf.keras.preprocessing.image.save_img('sampleimages_midres/' + str(i+1) + '.jpg', images_dnsc_midres[i])


# Defining downscale_images_lowres, which is used for downscaling the input images
# (loaded from sampleimages_midres) and saving them in sampleimages_lowres
def downscale_images_lowres(dataset_mr_array, model, count):
    images_dnsc_lowres = model.predict(dataset_mr_array)
    for i in range(0, count):
        tf.keras.preprocessing.image.save_img('sampleimages_lowres/' + str(i+1) + '.jpg', images_dnsc_lowres[i])


# Defining preprocess_images, used to convert images from integers to floating points
def preprocess_images(dataset):
    dataset = (dataset - 127.5) / 127.5
    return dataset


# Counting the number of files in the samppleimages_highres directory
count_hr = 0
for path in pathlib.Path("sampleimages_highres").iterdir():
    if path.is_file():
        count_hr += 1
print(count_hr)

# Loading in the images from the sampleimages_highres directory
training_images_hr_list = []
for i in range(1, count_hr+1):
    current_image_hr = tf.keras.preprocessing.image.load_img("sampleimages_highres/" + str(i) + ".jpg")
    current_image_hr_array = tf.keras.preprocessing.image.img_to_array(current_image_hr)
    training_images_hr_list.append(current_image_hr_array)
training_images_hr_array = np.asarray(training_images_hr_list)
training_images_hr_array = preprocess_images(dataset=training_images_hr_array)

# Make the downscaler model, and downscale the highres images to midres
dnsc_model_midres_model = make_downscaler_midres()
downscale_images_midres(training_images_hr_array, dnsc_model_midres_model, count_hr)

# Counting the number of files in the samppleimages_midres directory
count_mr = 0
for path in pathlib.Path("sampleimages_midres").iterdir():
    if path.is_file():
        count_mr += 1
print(count_mr)

# Loading in the images from the sampleimages_midres directory
training_images_mr_list = []
for i in range(1, count_mr+1):
    current_image_mr = tf.keras.preprocessing.image.load_img("sampleimages_midres/" + str(i) + ".jpg")
    current_image_mr_array = tf.keras.preprocessing.image.img_to_array(current_image_mr)
    training_images_mr_list.append(current_image_mr_array)
training_images_mr_array = np.asarray(training_images_mr_list)
training_images_mr_array = preprocess_images(dataset=training_images_mr_array)

# Make the downscaler model, and downscale the midres images to lowres
dnsc_model_lowres_model = make_downscaler_lowres()
downscale_images_lowres(training_images_mr_array, dnsc_model_lowres_model, count_mr)
