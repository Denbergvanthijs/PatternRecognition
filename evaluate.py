# Calculates FID score for a given folder of images compared to a given folder of images
# Usage: python evaluate.py --path-predicted <path to predicted images> --path-ground-truth <path to ground truth images>
# Example: python evaluate.py --path-predicted ./unpadded --path-ground-truth ./input/218/test/original

import argparse
import os

import numpy as np
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--path-predicted', type=str, help='Path to folder of predicted images', default='./unpadded')
parser.add_argument('--path-ground-truth', type=str, help='Path to folder of ground truth images', default='./input/218/test/original')
parser.add_argument('--n-batches', type=int, help='Number of batches to divide data into due to memory constraints', default=20)
parser.add_argument('--n-features', type=int, help='Number of features to use from each image due to memory constraints', default=128)
args = parser.parse_args()


def calculate_fid(images_original, images_predicted):
    """Calculates FID score between two batches of images. Based on:
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


def l2_norm(image1, image2):
    """Calculate L2 norm between two images per channel. Assumes channels are the last dimension."""
    l2_norms = []
    for i in range(image1.shape[-1]):
        l2_norms.append(np.linalg.norm(image1[..., i] - image2[..., i]))
    return l2_norms


def cosine_similarity(image1, image2):
    """Calculate cosine similarity between two images."""
    return np.dot(image1, image2) / (np.linalg.norm(image1) * np.linalg.norm(image2))


def load_images(path, extention='.jpg'):
    """Load images from a given path."""
    files = os.listdir(path)
    files = [f for f in files if f.endswith(extention)]

    images = []
    for file in files:
        img = np.array(Image.open(os.path.join(path, file)))
        images.append(img)
    return np.array(images)


if __name__ == "__main__":
    N_BATCHES = args.n_batches  # Number of batches to divide data into due to memory constraints
    N_FEATURES = args.n_features  # Number of features to use from each image due to memory constraints
    # The more features used, the more accurate the FID score will be
    # However, the more features used, the more memory will be required
    # Thus the more batches will be required
    # N_BATCHES cannot be greater than the number of images in the dataset

    # Load ground truth and predicted images
    images_original = load_images(args.path_ground_truth)
    images_predicted = load_images(args.path_predicted)

    # Check if shape of images is the same
    assert images_original.shape == images_predicted.shape, "Shape of images is not the same, have you restored the original resolution?"

    # Normalise images
    images_original = images_original / 255
    images_predicted = images_predicted / 255

    # Calculate L2 norm for each channel
    l2_norms = []
    for i in trange(images_original.shape[0], desc="Calculating L2 norm"):
        l2_norms.append(l2_norm(images_original[i], images_predicted[i]))

    print(
        f"L2 norm (average over all channels): {np.mean(l2_norms):.2f}; Std: {np.std(l2_norms):.2f}; Min: {np.min(l2_norms):.2f}; Max: {np.max(l2_norms):.2f}")
    print(
        f"L2 norm per channel: {np.mean(l2_norms, axis=0).round(2)}; Std: {np.std(l2_norms, axis=0).round(2)}; Min: {np.min(l2_norms, axis=0).round(2)}; Max: {np.max(l2_norms, axis=0).round(2)}")

    # Calculate cosine similarity
    cosine_similarities = []
    for i in trange(images_original.shape[0], desc="Calculating cosine similarity"):
        cosine_similarities.append(cosine_similarity(images_original[i].flatten(), images_predicted[i].flatten()))

    print(
        f"Cosine similarity: {np.mean(cosine_similarities):.2f}; Std: {np.std(cosine_similarities):.2f}; Min: {np.min(cosine_similarities):.2f}; Max: {np.max(cosine_similarities):.2f}")

    # Calculate Structural Similarity Index (SSIM)
    ssims = []
    for i in trange(images_original.shape[0], desc="Calculating SSIM"):
        ssims.append(ssim(images_original[i], images_predicted[i], channel_axis=2, data_range=1.0))

    print(f"SSIM: {np.mean(ssims):.2f}; Std: {np.std(ssims):.2f}; Min: {np.min(ssims):.2f}; Max: {np.max(ssims):.2f}")
    print(f"Indexes of 5 highest SSIM scores: {np.argsort(ssims)[-5:]}")
    print(f"SSIM scores of 5 highest SSIM scores: {np.sort(ssims)[-5:]}")

    # Flatten channels and reshape to 2D array of (N, 3*H*W)
    images_original = images_original.reshape(images_original.shape[0], -1)
    images_predicted = images_predicted.reshape(images_predicted.shape[0], -1)
    feature_percentage = N_FEATURES / images_original[0].shape[-1] * 100

    # Select N_FEATURES random features
    random_indexes = np.random.choice(images_original.shape[1], N_FEATURES, replace=False)
    images_original = images_original[:, random_indexes]
    images_predicted = images_predicted[:, random_indexes]
    images_per_batch = images_original.shape[0] // N_BATCHES

    # Divide data into batches
    images_original = np.array_split(images_original, N_BATCHES)
    images_predicted = np.array_split(images_predicted, N_BATCHES)

    fids = []
    for batch in trange(N_BATCHES, desc=f"Calculating FID score based on {N_FEATURES} features ({feature_percentage:.2f}%)"):
        fid_current = calculate_fid(np.array(images_original[batch]), np.array(images_predicted[batch]))
        fids.append(fid_current)

    # Calculate mean, std, min and max of FID scores
    # Divide by images_per_batch to get average FID score per image
    results = {"mean": np.mean(fids) / images_per_batch, "std": np.std(fids) / images_per_batch,
               "min": np.min(fids) / images_per_batch, "max": np.max(fids) / images_per_batch}

    print(f"Batch size: {images_per_batch} images; Number of batches: {N_BATCHES}; Number of features: {N_FEATURES} ({feature_percentage:.2f}%)")
    print(f"FID score per image: {results['mean']:.2f}; Std: {results['std']:.2f}; Min: {results['min']:.2f}; Max: {results['max']:.2f}")
