# Calculates FID score for a given folder of images compared to a given folder of images
# Usage: python evaluate.py --path-predicted <path to predicted images> --path-ground-truth <path to ground truth images>
# Example: python evaluate.py --path-predicted ./unpadded --path-ground-truth ./input/218/test/original

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
from scipy import linalg
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--path-predicted', type=str, help='Path to folder of predicted images', default='./unpadded/gan/output_2500')
parser.add_argument('--path-ground-truth', type=str, help='Path to folder of ground truth images', default='./input/218/test/original')
parser.add_argument('--n-batches', type=int, help='Number of batches to divide data into due to memory constraints', default=20)
parser.add_argument('--n-features', type=int, help='Number of features to use from each image due to memory constraints', default=2048)
parser.add_argument('--output-path', type=str, help='Path to output file to save results', default='./results/gan/2500.json')
parser.add_argument('--decimals', type=int, help='Number of decimals to round results to', default=4)
parser.add_argument('--seed', type=int, help='', default=42)
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


def calculate_statistics(array: np.ndarray, decimals: int = args.decimals) -> dict:
    """Calculate mean, standard deviation, min and max of an array."""
    return {"mean": np.mean(array).round(decimals), "std": np.std(array).round(decimals),
            "min": np.min(array).round(decimals), "max": np.max(array).round(decimals)}


def print_statistics(stats: dict, name: str = "") -> None:
    """Print statistics of an array."""
    print(f"{name} Mean: {stats['mean']}; Std: {stats['std']}; Min: {stats['min']}; Max: {stats['max']}")


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

    scores = {"L2": calculate_statistics(l2_norms)}
    print_statistics(scores["L2"], "L2")

    # Calculate cosine similarity
    cosine_similarities = []
    for i in trange(images_original.shape[0], desc="Calculating cosine similarity"):
        cosine_similarities.append(cosine_similarity(images_original[i].flatten(), images_predicted[i].flatten()))

    scores["cosine_similarity"] = calculate_statistics(cosine_similarities)
    print_statistics(scores["cosine_similarity"], "Cosine similarity")

    # Calculate Structural Similarity Index (SSIM)
    # Change axis order to (N, C, H, W) and change to torch tensor
    images_original_pt = torch.from_numpy(np.moveaxis(images_original, -1, 1))
    images_predicted_pt = torch.from_numpy(np.moveaxis(images_predicted, -1, 1))

    ssims = ssim(images_original_pt, images_predicted_pt, data_range=1.0, size_average=False, nonnegative_ssim=True).numpy()
    ms_ssims = ms_ssim(images_original_pt, images_predicted_pt, data_range=1.0, size_average=False).numpy()

    scores["SSIM"] = calculate_statistics(ssims)
    scores["MS-SSIM"] = calculate_statistics(ms_ssims)
    print_statistics(scores["SSIM"], "SSIM")
    print_statistics(scores["MS-SSIM"], "MS-SSIM")

    # Flatten channels and reshape to 2D array of (N, 3*H*W)
    images_original = images_original.reshape(images_original.shape[0], -1)
    images_predicted = images_predicted.reshape(images_predicted.shape[0], -1)
    feature_percentage = N_FEATURES / images_original[0].shape[-1] * 100

    # Select N_FEATURES random features
    np.random.seed(args.seed)
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
    scores["FID"] = calculate_statistics(fids)
    for key, value in scores["FID"].items():
        # Divide by images_per_batch to get average FID score per image
        scores["FID"][key] = (value / images_per_batch).round(args.decimals)
    print_statistics(scores["FID"], "FID")

    print("Saving scores to file...")
    fp_out = os.path.split(args.output_path)[0]
    os.makedirs(fp_out, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(scores, f, indent=4)
