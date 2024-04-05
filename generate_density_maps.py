import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from scipy import io
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KDTree
from tqdm import tqdm

from constants import *


def generate_density_map(img, points, k=4, beta=0.3, verbose=True):
    """
    Computes density map given an image and point annotations of figures via k-nearest neighbours lookup.

    Inputs:
        -img: Specified image (expected shape: (W,H,C)). The number of channels C will be ignored.
        -points: A 2D numpy array of point annotations of the form [[col, row],[col, row],...].
        -k (int=4): Number of nearest neighbours to use for computation (including the point itself)
        -beta (float=0.3): Scale parameter for the average nearest neighbour distances.

    Returns:
        -density: Density map of the image, of the same shape (without channels).
    """

    img_shape = (img.shape[0], img.shape[1])
    gt_count = len(points)
    if verbose:
        print(f"Image shape: {img_shape}, generating {gt_count} Gaussian kernels")
    density = np.zeros(img_shape, dtype=np.float32)
    if gt_count == 0:
        return density

    if k > len(points):
        warnings.warn(
            "Fewer number of figures than specified `k`, using total number of figures as k",
            stacklevel=2,
        )
        k = len(points)

    # Build KD-tree
    ## for leaf_size tradeoff, see https://stackoverflow.com/questions/65003877/understanding-leafsize-in-scipy-spatial-kdtree
    tree = KDTree(points, leaf_size=2048)  ## points.copy()
    distances, locations = tree.query(points, k=k)
    ## query, for each point, the distances and indices of its three closest nearest neighbours (four includes itself)

    for i, pt in enumerate(points):

        ## Build delta/indicator functino
        pt2d = np.zeros(img_shape, dtype=np.float32)
        ## if annotated figure is entirely within image, flag its loc
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.0
        else:
            continue

        ## Compute Gaussian kernel std as the scaled average nearest-neighbour distance
        if gt_count > 1:
            sigma = (
                np.sum(distances[i]) / (k - 1) * beta
            )  ## closest neighbour (itself) has distance 0 anyway
        else:
            sigma = 25  ### not a good method tbh. this implies that if there is only one figure,
            # the density map plots a single dot whose head size we randomly fix.
            # in reality the size of this figure in the image could widely vary from a tiny figure far off in the distance
            # to a close-up headshot.

            ## for the purposes of this project this does not matter because there are no/few such photos in the dataset with
            ## only one figure

        # Convolve
        density += gaussian_filter(pt2d, sigma, mode="constant")

    return density


if __name__ == "__main__":

    splits = ["train_data", "test_data"]
    for split in splits:

        img_paths = sorted(os.listdir(f"{data_path}/{split}/images"))

        gt_folder = f"{data_path}/{split}/gt_maps"
        if not os.path.exists(gt_folder):
            os.makedirs(gt_folder)

        for path in tqdm(img_paths):
            gt_path = f"GT_{path}".replace(".jpg", ".mat")

            gt_mat = io.loadmat(f"{data_path}/{split}/ground_truth/{gt_path}")
            points = gt_mat["image_info"][0, 0]["location"][0, 0]
            img = plt.imread(f"{data_path}/{split}/images/{path}")
            density = generate_density_map(img, points, k=4, beta=0.3, verbose=False)

            base_path = os.path.basename(path).split(".")[0]

            np.save(f"{data_path}/{split}/gt_maps/GT_{base_path}.npy", density)

### NOTES: Summing this ground truth density map typically leads to a fewer number of people than the ground truth label, e.g. 2-5 less
### at times up to 15 less. This is likely intended behaviour (figures who are partially occluded count for less than 1)
