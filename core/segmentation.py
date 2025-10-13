"""Core logic for the segmentation application."""
from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import cv2
import numpy as np
import skfuzzy as fuzz
from skimage import io
from sklearn.mixture import GaussianMixture

class Config:
    SUPPORTED_FORMATS = [".jpg", ".png", ".tiff", ".bmp"]
    OUTPUT_DIR = "output"
    SETTINGS_ORG = "MicroscopicApp"
    SETTINGS_APP = "ImageSegmentation"

class Loader:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        _, ext = os.path.splitext(path)
        if ext.lower() not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        try:
            image = io.imread(path)  # loaded as RGB
        except Exception as e:
            logging.exception("Failed to load image.")
            raise e
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in ['true', '1']
    return bool(val)

#####################################
# 2. SEGMENTATION FUNCTIONS
#####################################

class Preprocessor:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i/255.0)**inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        if len(image.shape)==2:
            return cv2.equalizeHist(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def noise_reduction(image: np.ndarray, method: str, ksize: int = 5) -> np.ndarray:
        if method == "Gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif method == "Median":
            return cv2.medianBlur(image, ksize)
        elif method == "Bilateral":
            return cv2.bilateralFilter(image, ksize, 75, 75)
        return image

    @staticmethod
    def threshold_custom(image: np.ndarray, min_thresh: int, max_thresh: int) -> np.ndarray:
        gray = Preprocessor.to_grayscale(image)
        ret, thresh = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def sharpen(image: np.ndarray, strength: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0,0), sigmaX=3)
        return cv2.addWeighted(image, 1+strength, blurred, -strength, 0)

class Detector:
    @staticmethod
    def adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
        gray = Preprocessor.to_grayscale(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def watershed_segmentation(image: np.ndarray, kernel_size: int = 3, opening_iterations: int = 2,
                               dilation_iterations: int = 3, distance_threshold_factor: float = 0.7) -> np.ndarray:
        gray = Preprocessor.to_grayscale(image)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
        sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iterations)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, distance_threshold_factor*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        markers = cv2.watershed(image, markers)
        annotated = image.copy()
        annotated[markers==-1] = [0,0,255]
        return annotated

    @staticmethod
    def edge_based_segmentation(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150,
                                aperture_size: int = 3) -> np.ndarray:
        gray = Preprocessor.to_grayscale(image)
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(edges, kernel, iterations=1)

    @staticmethod
    def kmeans_segmentation(image: np.ndarray, K: int = 2, seed: int = 42) -> np.ndarray:
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.setRNGSeed(seed)
        Z = image.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label_arr, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label_arr.flatten()]
        segmented = res.reshape(image.shape)
        gray = Preprocessor.to_grayscale(segmented)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

def global_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def sobel_operator(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(grad, 0, 255))

def prewitt_operator(image: np.ndarray) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    grad_x = cv2.filter2D(gray, -1, kernelx)
    grad_y = cv2.filter2D(gray, -1, kernely)
    grad = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
    return np.uint8(np.clip(grad, 0, 255))

def laplacian_operator(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return np.uint8(np.clip(np.abs(lap), 0, 255))

def region_growing(image: np.ndarray, seed: Tuple[int,int], tolerance: int = 10) -> np.ndarray:
    img = Preprocessor.to_grayscale(image).copy()
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    cv2.floodFill(img, mask, seedPoint=seed, newVal=255, loDiff=tolerance, upDiff=tolerance)
    return img

def region_splitting_merging(image: np.ndarray, min_size: int = 16, std_thresh: float = 10) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image).astype(np.float32)
    h, w = gray.shape
    seg = np.zeros_like(gray)
    def split_region(x, y, w, h):
        region = gray[y:y+h, x:x+w]
        if w <= min_size or h <= min_size or np.std(region) < std_thresh:
            seg[y:y+h, x:x+w] = np.mean(region)
        else:
            half_w = w//2
            half_h = h//2
            split_region(x, y, half_w, half_h)
            split_region(x+half_w, y, w-half_w, half_h)
            split_region(x, y+half_h, half_w, h-half_h)
            split_region(x+half_w, y+half_h, w-half_w, h-half_h)
    split_region(0, 0, w, h)
    return np.uint8(seg)

def fuzzy_c_means(image: np.ndarray, K: int = 2, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    gray = Preprocessor.to_grayscale(image).astype(np.float32)
    flat = gray.flatten()
    norm = flat / 255.0
    data = np.expand_dims(norm, 0)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, c=K, m=2, error=0.005, maxiter=1000, init=None)
    cluster_labels = np.argmax(u, axis=0)
    centers = (cntr * 255).flatten()
    segmented = centers[cluster_labels]
    segmented = segmented.reshape(gray.shape)
    ret, thresh = cv2.threshold(np.uint8(segmented), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def mean_shift_segmentation(image: np.ndarray, spatial_radius: int = 20, color_radius: int = 30) -> np.ndarray:
    shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)
    gray = Preprocessor.to_grayscale(shifted)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def gmm_segmentation(image: np.ndarray, components: int = 2, seed: int = 42) -> np.ndarray:
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w, c = image.shape
    X = image.reshape(-1, c)
    gmm = GaussianMixture(n_components=components, random_state=seed)
    gmm.fit(X)
    labels = gmm.predict(X)
    cluster_means = []
    for i in range(components):
        cluster_pixels = X[labels==i]
        if len(cluster_pixels) > 0:
            mean_color = np.mean(cluster_pixels, axis=0)
            gray_val = 0.114*mean_color[0] + 0.587*mean_color[1] + 0.299*mean_color[2]
        else:
            gray_val = 0
        cluster_means.append(gray_val)
    segmented = np.array([cluster_means[label] for label in labels])
    segmented = segmented.reshape(h, w)
    ret, thresh = cv2.threshold(segmented.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def graph_cuts(image: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = image * mask2[:,:,np.newaxis]
    gray = Preprocessor.to_grayscale(result)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def active_contour(image: np.ndarray, iterations: int = 250, alpha: float = 0.015, beta: float = 10, gamma: float = 0.001) -> np.ndarray:
    from skimage.segmentation import active_contour
    from skimage.filters import gaussian
    gray = Preprocessor.to_grayscale(image)
    s = np.linspace(0, 2*np.pi, 400)
    x = image.shape[1]/2 + (image.shape[1]/4)*np.cos(s)
    y = image.shape[0]/2 + (image.shape[0]/4)*np.sin(s)
    init = np.array([x, y]).T
    snake = active_contour(gaussian(gray, 3), init, alpha=alpha, beta=beta, gamma=gamma)
    contour_img = image.copy()
    cv2.polylines(contour_img, [np.int32(snake)], isClosed=True, color=(0,255,0), thickness=2)
    return contour_img

# Morphological Operations

def morphological_opening(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int) -> np.ndarray:
    ks_lower = kernel_shape.lower()
    if ks_lower == "rectangular":
        shape = cv2.MORPH_RECT
    elif ks_lower == "elliptical":
        shape = cv2.MORPH_ELLIPSE
    elif ks_lower == "cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def morphological_closing(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int) -> np.ndarray:
    ks_lower = kernel_shape.lower()
    if ks_lower == "rectangular":
        shape = cv2.MORPH_RECT
    elif ks_lower == "elliptical":
        shape = cv2.MORPH_ELLIPSE
    elif ks_lower == "cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def morphological_dilation(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int) -> np.ndarray:
    ks_lower = kernel_shape.lower()
    if ks_lower == "rectangular":
        shape = cv2.MORPH_RECT
    elif ks_lower == "elliptical":
        shape = cv2.MORPH_ELLIPSE
    elif ks_lower == "cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=iterations)

def morphological_erosion(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int) -> np.ndarray:
    ks_lower = kernel_shape.lower()
    if ks_lower == "rectangular":
        shape = cv2.MORPH_RECT
    elif ks_lower == "elliptical":
        shape = cv2.MORPH_ELLIPSE
    elif ks_lower == "cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.erode(image, kernel, iterations=iterations)

def remove_border_regions(image: np.ndarray, border_distance: int) -> np.ndarray:
    # Modified: Set border region pixels to 0 (black)
    mask = np.ones(image.shape[:2], dtype=np.uint8)*255
    mask[border_distance:-border_distance, border_distance:-border_distance] = 0
    result = image.copy()
    if len(image.shape)==2:
        result[mask==255] = 0
    else:
        result[mask==255] = [0,0,0]
    return result

#####################################
# 3. PIPELINE MANAGEMENT
#####################################

