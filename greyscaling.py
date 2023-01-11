import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load datasets
ds_train = tfds.load('celeb_a', split='train')
ds_val = tfds.load('celeb_a', split='validation')
ds_test = tfds.load('celeb_a', split='test')

# Only select the image feature, reducing the dataset size
ds_train = ds_train.map(lambda x: x['image'])
ds_val = ds_val.map(lambda x: x['image'])
ds_test = ds_test.map(lambda x: x['image'])


def greyscale_tf(image):
    """Converts RGB image to greyscale"""
    image = tf.cast(image, tf.float32)
    return tf.tensordot(image[..., :3], [0.299, 0.587, 0.114], axes=1)


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
for name, dataset in zip(("train", "val", "test"), (ds_train_combined, ds_val_combined, ds_test_combined)):
    images, images_grey = zip(*dataset.as_numpy_iterator())
    np.save(f'./data/{name}_images.npy', np.array(images))
    np.save(f'./data/{name}_images_grey.npy', np.array(images_grey))

# Load npy files
test_images = np.load('test_images.npy')
test_images_grey = np.load('test_images_grey.npy')

# Plot images
fig, axis = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True)
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i])
    plt.subplot(2, 5, i+6)
    plt.imshow(test_images_grey[i], cmap='gray')
plt.show()
