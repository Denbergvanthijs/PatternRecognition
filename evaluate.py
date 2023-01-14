# Calculates FID score for a given folder of images compared to a given folder of images
# Usage: python evaluate.py --path-predicted <path to predicted images> --path-ground-truth <path to ground truth images>
# Example: python evaluate.py --path-predicted ./unpadded --path-ground-truth ./input/218/test/original

import argparse
import os

import numpy as np
from PIL import Image
from scipy import linalg
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--path-predicted', type=str, help='Path to folder of predicted images', default='./unpadded')
parser.add_argument('--path-ground-truth', type=str, help='Path to folder of ground truth images', default='./input/218/test/original')
args = parser.parse_args()


def calculate_fid(images_original, images_predicted):
    """Based on:
    https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
    """
    # Calculate mean and covariance statistics
    mu1 = images_original.mean(axis=0)
    sigma1 = np.cov(images_original, rowvar=False)
    mu2 = images_predicted.mean(axis=0)
    sigma2 = np.cov(images_predicted, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Calculate square root of product between cov
    dot_product = sigma1.dot(sigma2)
    covmean = linalg.sqrtm(dot_product)

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def load_images(path, extention='.jpg'):
    files = os.listdir(path)
    files = [f for f in files if f.endswith(extention)]

    images = []
    for file in files:
        img = np.array(Image.open(os.path.join(path, file)))
        images.append(img)
    return np.array(images)


if __name__ == "__main__":
    N_BATCHES = 20  # Number of batches to divide data into due to memory constraints
    N_FEATURES = 128  # Number of features to use from each image due to memory constraints
    # The more features used, the more accurate the FID score will be
    # However, the more features used, the more memory will be required
    # Thus the more batches will be required
    # N_BATCHES cannot be greater than the number of images in the dataset

    # Load ground truth and predicted images
    images_original = load_images(args.path_ground_truth)
    images_predicted = load_images(args.path_predicted)

    # Check if shape of images is the same
    assert images_original.shape == images_predicted.shape, "Shape of images is not the same, have you restored the original resolution?"

    # Flatten channels and reshape to 2D array of (N, 3*H*W)
    images_original = images_original.reshape(images_original.shape[0], -1)
    images_predicted = images_predicted.reshape(images_predicted.shape[0], -1)

    # Only keep first few values of each row due to memory constraints
    images_original = images_original[:, :N_FEATURES]
    images_predicted = images_predicted[:, :N_FEATURES]

    # Normalise images
    images_original = images_original / 255
    images_predicted = images_predicted / 255

    # Divide data into batches
    images_original = np.array_split(images_original, N_BATCHES)
    images_predicted = np.array_split(images_predicted, N_BATCHES)

    fids = []
    for batch in trange(N_BATCHES, desc="Calculating FID score"):
        fid_current = calculate_fid(np.array(images_original[batch]), np.array(images_predicted[batch]))
        fids.append(fid_current)

    print(f"FID score: {np.mean(fids):.2f}; Std: {np.std(fids):.2f}; Min: {np.min(fids):.2f}; Max: {np.max(fids):.2f}")
