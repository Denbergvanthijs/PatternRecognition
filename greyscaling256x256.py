import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


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

# Reshape images to 256x256
# This is needed when evaluation on Places365 with the GAN model
# See https://github.com/ImagingLab/Colorizing-with-GANs/issues/4#issuecomment-407769051
ds_train = ds_train.map(lambda x: tf.image.resize_with_pad(x, 256, 256))
ds_val = ds_val.map(lambda x: tf.image.resize_with_pad(x, 256, 256))
ds_test = ds_test.map(lambda x: tf.image.resize_with_pad(x, 256, 256))

# Convert test dataset to greyscale
ds_train_grey_np = ds_train.map(lambda x: greyscale_tf(x))
ds_val_grey_np = ds_val.map(lambda x: greyscale_tf(x))
ds_test_grey_np = ds_test.map(lambda x: greyscale_tf(x))

# Combine original with greyscaled images
# This is to synchronise the images
ds_train_combined = tf.data.Dataset.zip((ds_train, ds_train_grey_np))
ds_val_combined = tf.data.Dataset.zip((ds_val, ds_val_grey_np))
ds_test_combined = tf.data.Dataset.zip((ds_test, ds_test_grey_np))

# Make dataset smaller due to computational limitations
ds_train_combined = ds_train_combined.take(100)
ds_val_combined = ds_val_combined.take(100)
ds_test_combined = ds_test_combined.take(100)

# Plot images
fig, axis = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True)
for i, (image, image_grey) in enumerate(ds_test_combined.take(5)):
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.subplot(2, 5, i+6)
    plt.imshow(image_grey, cmap='gray')
plt.show()

# Save dataset to two seperate numpy arrays
# This is to save time when running the eventual colorisation scripts
if not os.path.exists('./data/256'):
    os.makedirs('./data/256')
for name, dataset in zip(("train", "val", "test"), (ds_train_combined, ds_val_combined, ds_test_combined)):
    images, images_grey = zip(*dataset.as_numpy_iterator())
    np.save(f'./data/256/{name}_images_256x256.npy', np.array(images))
    np.save(f'./data/256/{name}_images_grey_256x256.npy', np.array(images_grey))

# Load npy files as a test
test_images = np.load('./data/256/test_images_256x256.npy')
test_images_grey = np.load('./data/256/test_images_grey_256x256.npy')

# Make missing folders
for path in ['./input/256/train/original', './input/256/train/grey',
             './input/256/val/original', './input/256/val/grey',
             './input/256/test/original', './input/256/test/grey']:
    if not os.path.exists(path):
        os.makedirs(path)

# Save all images to input folder
for i in range(len(test_images)):
    plt.imsave(f'./input/256/test/original/{i}.jpg', test_images[i])
    plt.imsave(f'./input/256/test/grey/{i}_grey.jpg', test_images_grey[i], cmap='gray')
