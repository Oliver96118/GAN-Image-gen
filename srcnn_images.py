
import numpy as np
import tensorflow as tf
import pathlib


# This script is used for applying SRCNN to our upscaled generated images
# Before running this script, ensure that an SRCNN model is saved
# under models/srcnn_model/model1


# Defining preprocess_images, used to convert images from integers to floating points
def preprocess_images(dataset):
    dataset = (dataset - 127.5) / 127.5
    return dataset


# Defining process_images, used for applying the SRCNN to the images
def process_images(dataset_lr_array, model, count, savepath):
    images_srcnn = model.predict(dataset_lr_array)
    for i in range(0, count):
        tf.keras.preprocessing.image.save_img(savepath + str(i+1) + '.jpg', images_srcnn[i])


# Count the number of files in the generatedimages_upscaled directory
count_lr = 0
for path in pathlib.Path("generatedimages_upscaled").iterdir():
    if path.is_file():
        count_lr += 1
print("Number of generated and upscaled images: " + str(count_lr))

# Loading in the images from the generatedimages_upscaled directory
training_images_lr = []
training_images_lr_list = []
for i in range(1, count_lr+1):
    current_image_lr = tf.keras.preprocessing.image.load_img("generatedimages_upscaled/" + str(i) + ".jpg")
    current_image_lr_array = tf.keras.preprocessing.image.img_to_array(current_image_lr)
    training_images_lr.append(current_image_lr)
    training_images_lr_list.append(current_image_lr_array)
training_images_lr_array = np.asarray(training_images_lr_list)
training_images_lr_array = preprocess_images(dataset=training_images_lr_array)

# Loading in the SRCNN model
srcnn_model = tf.keras.models.load_model('models/srcnn_model/model1')
# Processing the images
process_images(training_images_lr_array, srcnn_model, count_lr, savepath='generatedimages_srcnn/')
