import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm, trange

TRAIN_SIZES = (100, 1_000, 2_500, 10_000)
VAL_SIZE = 100
TEST_SIZE = 100


def greyscale_tf(image):
    """Converts RGB image to greyscale"""
    return tf.tensordot(image[..., :3], [0.299, 0.587, 0.114], axes=1)


# Load datasets
ds_train = tfds.load('celeb_a', split='train')
ds_val = tfds.load('celeb_a', split='validation')
ds_test = tfds.load('celeb_a', split='test')

# Only select the image feature, reducing the dataset size
ds_train = ds_train.map(lambda x: x['image'])
ds_val = ds_val.map(lambda x: x['image'])
ds_test = ds_test.map(lambda x: x['image'])

# Convert to float32 and rescale images to [0, 1]
ds_train = ds_train.map(lambda x: tf.cast(x, tf.float32) / 255.0)
ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32) / 255.0)
ds_test = ds_test.map(lambda x: tf.cast(x, tf.float32) / 255.0)

# Convert test dataset to greyscale
ds_train_grey_np = ds_train.map(lambda x: greyscale_tf(x))
ds_val_grey_np = ds_val.map(lambda x: greyscale_tf(x))
ds_test_grey_np = ds_test.map(lambda x: greyscale_tf(x))

# Combine original with greyscaled images
# This is to synchronise the images
ds_train_combined = tf.data.Dataset.zip((ds_train, ds_train_grey_np))
ds_val_combined = tf.data.Dataset.zip((ds_val, ds_val_grey_np))
ds_test_combined = tf.data.Dataset.zip((ds_test, ds_test_grey_np))

# Plot images
# fig, axis = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True)
# for i, (image, image_grey) in enumerate(ds_test_combined.take(5)):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(image)
#     plt.subplot(2, 5, i+6)
#     plt.imshow(image_grey, cmap='gray')
# plt.show()

# Make missing folders
os.makedirs('./input/218', exist_ok=True)

# Save different sizes of the train dataset
for train_size in tqdm(TRAIN_SIZES, desc='Saving train datasets'):
    images, images_grey = zip(*ds_train_combined.take(train_size).as_numpy_iterator())
    np.save(f'./input/218/train_images_{train_size}.npy', np.array(images))
    np.save(f'./input/218/train_images_grey_{train_size}.npy', np.array(images_grey))

# Save validation and test dataset, always 100 images
images, images_grey = zip(*ds_val_combined.take(VAL_SIZE).as_numpy_iterator())
np.save('./input/218/val_images.npy', np.array(images))
np.save('./input/218/val_images_grey.npy', np.array(images_grey))

images, images_grey = zip(*ds_test_combined.take(TEST_SIZE).as_numpy_iterator())
np.save('./input/218/test_images.npy', np.array(images))
np.save('./input/218/test_images_grey.npy', np.array(images_grey))

# Load npy files as a test
test_images = np.load('./input/218/test_images.npy')
test_images_grey = np.load('./input/218/test_images_grey.npy')

# Save all images, save each train size in seperate folders
for train_size in TRAIN_SIZES:
    train_images = np.load(f'./input/218/train_images_{train_size}.npy')
    train_images_grey = np.load(f'./input/218/train_images_grey_{train_size}.npy')
    # Check if folder exists
    os.makedirs(f'./input/218/train/{train_size}/original', exist_ok=True)
    os.makedirs(f'./input/218/train/{train_size}/grey', exist_ok=True)
    for i in trange(train_size, desc=f'Saving train dataset with size {train_size}'):
        plt.imsave(f'./input/218/train/{train_size}/original/{i}.jpg', train_images[i])
        plt.imsave(f'./input/218/train/{train_size}/grey/{i}.jpg', train_images_grey[i], cmap='gray')

# Save validation and test images
val_images = np.load('./input/218/val_images.npy')
val_images_grey = np.load('./input/218/val_images_grey.npy')
os.makedirs('./input/218/val/original', exist_ok=True)
os.makedirs('./input/218/val/grey', exist_ok=True)
for i in trange(VAL_SIZE, desc='Saving validation dataset'):
    plt.imsave(f'./input/218/val/original/{i}.jpg', val_images[i])
    plt.imsave(f'./input/218/val/grey/{i}.jpg', val_images_grey[i], cmap='gray')

test_images = np.load('./input/218/test_images.npy')
test_images_grey = np.load('./input/218/test_images_grey.npy')
os.makedirs('./input/218/test/original', exist_ok=True)
os.makedirs('./input/218/test/grey', exist_ok=True)
for i in trange(TEST_SIZE, desc='Saving test dataset'):
    plt.imsave(f'./input/218/test/original/{i}.jpg', test_images[i])
    plt.imsave(f'./input/218/test/grey/{i}.jpg', test_images_grey[i], cmap='gray')

# Plot npy images to show that they are still correct
# fig, axis = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True)
# for i in range(5):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(test_images[i])
#     plt.subplot(2, 5, i+6)
#     plt.imshow(test_images_grey[i], cmap='gray')
# plt.show()
