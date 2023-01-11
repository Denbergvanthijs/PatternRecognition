import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load datasets
# ds_train = tfds.load('celeb_a', split='train')
# ds_val = tfds.load('celeb_a', split='validation')
ds_test = tfds.load('celeb_a', split='test')


def greyscale_tf(image):
    """Converts RGB image to greyscale"""
    image = tf.cast(image, tf.float32)
    return tf.tensordot(image[..., :3], [0.299, 0.587, 0.114], axes=1)


# Convert test dataset to greyscale
ds_test_grey_np = ds_test.map(lambda x: greyscale_tf(x['image']))

# Print shape of next image in dataset
image = next(iter(ds_test_grey_np))
print(image.shape)
