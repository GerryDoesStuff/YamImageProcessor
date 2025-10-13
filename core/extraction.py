"""Core logic for the feature extraction application."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage import io
from skimage.feature import hog, local_binary_pattern
from skimage.measure import label, regionprops

class Config:
    SUPPORTED_FORMATS = [".jpg", ".png", ".tiff", ".bmp", ".npy"]
    OUTPUT_DIR = "output"
    SETTINGS_ORG = "MicroscopicApp"
    SETTINGS_APP = "ImageProcessor"

class Loader:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        image = io.imread(path)  # loaded as RGB
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in ['true', '1']
    return bool(val)

#####################################
# 2. PREPROCESSOR (for extraction functions that need grayscale)
#####################################

class Preprocessor:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

#####################################
# 3. FEATURE EXTRACTION FUNCTIONS
#####################################

# (A) Region Properties
from skimage.measure import label, regionprops

def region_properties_extraction(image: np.ndarray) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labeled = label(binary)
    props = regionprops(labeled)
    annotated = image.copy()
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (0,255,0), 2)
        cy, cx = prop.centroid
        cv2.circle(annotated, (int(cx), int(cy)), 3, (0,0,255), -1)
    return annotated

def region_properties_data(image: np.ndarray) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labeled = label(binary)
    props = regionprops(labeled)
    data = []
    for i, prop in enumerate(props):
        data.append({
            "region_index": i+1,
            "area": prop.area,
            "perimeter": prop.perimeter,
            "centroid": prop.centroid,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "extent": prop.extent,
            "orientation": prop.orientation
        })
    return pd.DataFrame(data)

# (B) Hu Moments
def hu_moments_extraction(image: np.ndarray) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    annotated = image.copy()
    text = "Hu Moments: " + ", ".join([f"{h:.2e}" for h in hu])
    cv2.putText(annotated, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return annotated

def hu_moments_data(image: np.ndarray) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    return pd.DataFrame([hu], columns=[f"hu_{i+1}" for i in range(len(hu))])

# (C) Local Binary Patterns (LBP)
def lbp_extraction(image: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    lbp = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6))
    return lbp

def lbp_data(image: np.ndarray, P: int = 8, R: float = 1.0) -> pd.DataFrame:
    lbp_img = lbp_extraction(image, P, R)
    hist, bin_edges = np.histogram(lbp_img, bins=256, range=(0,255))
    return pd.DataFrame({"bin": bin_edges[:-1], "count": hist})

# (D) Haralick (Custom GLCM)
def my_greycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True):
    image = np.asarray(image, dtype=np.uint8)
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    rows, cols = image.shape
    for i, d in enumerate(distances):
        for j, angle in enumerate(angles):
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))
            for r in range(rows):
                for c in range(cols):
                    r2 = r + dy
                    c2 = c + dx
                    if 0 <= r2 < rows and 0 <= c2 < cols:
                        i_val = image[r, c]
                        j_val = image[r2, c2]
                        glcm[i_val, j_val, i, j] += 1
                        if symmetric:
                            glcm[j_val, i_val, i, j] += 1
    if normed:
        glcm_sum = glcm.sum(axis=(0,1), keepdims=True)
        glcm = glcm / (glcm_sum + 1e-10)
    return glcm

def my_greycoprops(glcm, prop):
    props = np.zeros(glcm.shape[2:], dtype=np.float64)
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            P = glcm[:, :, i, j]
            I, J = np.indices(P.shape)
            if prop == 'contrast':
                props[i, j] = np.sum(P * (I - J) ** 2)
            elif prop == 'correlation':
                mu_i = np.sum(I * P)
                mu_j = np.sum(J * P)
                sigma_i = np.sqrt(np.sum(((I - mu_i) ** 2) * P))
                sigma_j = np.sqrt(np.sum(((J - mu_j) ** 2) * P))
                if sigma_i * sigma_j == 0:
                    props[i, j] = 1
                else:
                    props[i, j] = np.sum(((I - mu_i) * (J - mu_j) * P) / (sigma_i * sigma_j))
            elif prop == 'energy':
                props[i, j] = np.sum(P ** 2)
            elif prop == 'homogeneity':
                props[i, j] = np.sum(P / (1.0 + (I - J) ** 2))
            else:
                props[i, j] = 0
    return props

def haralick_extraction(image: np.ndarray, distance: int = 1, angle: float = 0.0) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    glcm = my_greycomatrix(gray, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
    contrast = my_greycoprops(glcm, 'contrast')[0, 0]
    correlation = my_greycoprops(glcm, 'correlation')[0, 0]
    energy = my_greycoprops(glcm, 'energy')[0, 0]
    homogeneity = my_greycoprops(glcm, 'homogeneity')[0, 0]
    annotated = image.copy()
    text = f"Haralick: Contrast={contrast:.2f}, Corr={correlation:.2f}, Energy={energy:.2f}, Homog={homogeneity:.2f}"
    cv2.putText(annotated, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return annotated

def haralick_data(image: np.ndarray, distance: int = 1, angle: float = 0.0) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    glcm = my_greycomatrix(gray, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
    contrast = my_greycoprops(glcm, 'contrast')[0, 0]
    correlation = my_greycoprops(glcm, 'correlation')[0, 0]
    energy = my_greycoprops(glcm, 'energy')[0, 0]
    homogeneity = my_greycoprops(glcm, 'homogeneity')[0, 0]
    return pd.DataFrame([{"contrast": contrast, "correlation": correlation, "energy": energy, "homogeneity": homogeneity}])

# (E) Gabor Filter
def gabor_extraction(image: np.ndarray, ksize: int = 21, sigma: float = 5.0, theta: float = 0.0,
                     lambd: float = 10.0, gamma: float = 0.5, psi: float = 0.0) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    return filtered

def gabor_data(image: np.ndarray, ksize: int = 21, sigma: float = 5.0, theta: float = 0.0,
               lambd: float = 10.0, gamma: float = 0.5, psi: float = 0.0) -> pd.DataFrame:
    filtered = gabor_extraction(image, ksize, sigma, theta, lambd, gamma, psi)
    return pd.DataFrame([{"mean": float(np.mean(filtered)), "std": float(np.std(filtered))}])

# (F) Fourier Descriptors
def fourier_descriptors_extraction(image: np.ndarray, num_coeff: int = 10) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    contour_complex = np.array([pt[0][0] + 1j * pt[0][1] for pt in largest])
    fourier_result = np.fft.fft(contour_complex)
    n = len(fourier_result)
    filtered = np.zeros(n, dtype=complex)
    filtered[:num_coeff] = fourier_result[:num_coeff]
    filtered[-num_coeff:] = fourier_result[-num_coeff:]
    contour_reconstructed = np.fft.ifft(filtered)
    annotated = image.copy()
    pts = np.array([[int(pt.real), int(pt.imag)] for pt in contour_reconstructed])
    cv2.drawContours(annotated, [pts], -1, (0,255,255), 2)
    return annotated

def fourier_data(image: np.ndarray, num_coeff: int = 10) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pd.DataFrame()
    largest = max(contours, key=cv2.contourArea)
    contour_complex = np.array([pt[0][0] + 1j * pt[0][1] for pt in largest])
    fourier_result = np.fft.fft(contour_complex)
    n = len(fourier_result)
    filtered = np.zeros(n, dtype=complex)
    filtered[:num_coeff] = fourier_result[:num_coeff]
    filtered[-num_coeff:] = fourier_result[-num_coeff:]
    contour_reconstructed = np.fft.ifft(filtered)
    polygon = np.array([[int(pt.real), int(pt.imag)] for pt in contour_reconstructed])
    area = cv2.contourArea(polygon)
    perimeter = cv2.arcLength(polygon, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    data = {"num_coeff": num_coeff, "area": area, "perimeter": perimeter, "circularity": circularity}
    for i, coeff in enumerate(np.concatenate([fourier_result[:num_coeff], fourier_result[-num_coeff:]])):
        data[f"coeff_{i}_real"] = coeff.real
        data[f"coeff_{i}_imag"] = coeff.imag
    return pd.DataFrame([data])

# (G) HOG
def hog_extraction(image: np.ndarray, orientations: int = 9, pixels_per_cell: Tuple[int,int]=(8,8),
                   cells_per_block: Tuple[int,int]=(3,3)) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    features, hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, visualize=True, block_norm='L2-Hys')
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min() + 1e-6)
    hog_image = np.uint8(255 * hog_image)
    return hog_image

def hog_data(image: np.ndarray, orientations: int = 9, pixels_per_cell: Tuple[int,int]=(8,8),
             cells_per_block: Tuple[int,int]=(3,3)) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    features, _ = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, visualize=True, feature_vector=True, block_norm='L2-Hys')
    return pd.DataFrame([features])

# (H) Histogram Statistics
def histogram_stats_extraction(image: np.ndarray) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    total = np.sum(hist) if np.sum(hist) != 0 else 1
    pixels = np.arange(256)
    mean_val = np.sum(pixels * hist) / total
    variance_val = np.sum(((pixels - mean_val) ** 2) * hist) / total
    data = np.repeat(pixels, hist.astype(int))
    skew_val = skew(data) if len(data) > 0 else 0
    kurt_val = kurtosis(data) if len(data) > 0 else 0
    annotated = image.copy()
    text = f"Hist: Mean={mean_val:.2f}, Var={variance_val:.2f}, Skew={skew_val:.2f}, Kurt={kurt_val:.2f}"
    cv2.putText(annotated, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return annotated

def histogram_data(image: np.ndarray) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    total = np.sum(hist) if np.sum(hist) != 0 else 1
    pixels = np.arange(256)
    mean_val = np.sum(pixels * hist) / total
    variance_val = np.sum(((pixels - mean_val) ** 2) * hist) / total
    data = np.repeat(pixels, hist.astype(int))
    skew_val = skew(data) if len(data) > 0 else 0
    kurt_val = kurtosis(data) if len(data) > 0 else 0
    return pd.DataFrame([{"mean": mean_val, "variance": variance_val, "skewness": skew_val, "kurtosis": kurt_val}])

# (I) Fractal Dimension (Box-counting)
def fractal_dimension_extraction(image: np.ndarray, min_box_size: int = 2) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                             np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])
    sizes = []
    counts = []
    p = min(binary.shape)
    k = min_box_size
    while k <= p:
        sizes.append(k)
        counts.append(boxcount(binary, k))
        k *= 2
    logs = np.log(sizes)
    logc = np.log(counts)
    coeffs = np.polyfit(logs, logc, 1)
    fractal_dim = -coeffs[0]
    annotated = image.copy()
    text = f"Fractal Dim: {fractal_dim:.2f}"
    cv2.putText(annotated, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    return annotated

def fractal_data(image: np.ndarray, min_box_size: int = 2) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                             np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])
    sizes = []
    counts = []
    p = min(binary.shape)
    k = min_box_size
    while k <= p:
        sizes.append(k)
        counts.append(boxcount(binary, k))
        k *= 2
    logs = np.log(sizes)
    logc = np.log(counts)
    coeffs = np.polyfit(logs, logc, 1)
    fractal_dim = -coeffs[0]
    return pd.DataFrame([{"fractal_dimension": fractal_dim}])

# (J) Approximate Shape Extraction
def optimize_epsilon_for_contour(cnt, error_threshold=1.0) -> Tuple[float, np.ndarray]:
    arc_len = cv2.arcLength(cnt, True)
    factors = np.arange(0.005, 0.101, 0.005)
    best_factor = None
    best_approx = None
    for factor in factors:
        epsilon = factor * arc_len
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        errors = [abs(cv2.pointPolygonTest(approx, (float(point[0][0]), float(point[0][1])), True))
                  for point in cnt]
        avg_error = np.mean(errors)
        if avg_error <= error_threshold:
            best_factor = factor
            best_approx = approx
            break
    if best_factor is None:
        best_error = float('inf')
        for factor in factors:
            epsilon = factor * arc_len
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            errors = [abs(cv2.pointPolygonTest(approx, (float(point[0][0]), float(point[0][1])), True))
                      for point in cnt]
            avg_error = np.mean(errors)
            if avg_error < best_error:
                best_error = avg_error
                best_factor = factor
                best_approx = approx
    return best_factor, best_approx

def approximate_shape_extraction(image: np.ndarray, error_threshold: float = 1.0) -> np.ndarray:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        best_factor, approx = optimize_epsilon_for_contour(cnt, error_threshold=error_threshold)
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        vertices = approx.reshape(-1, 2)
        num_vertices = len(vertices)
        edge_lengths = []
        for i in range(num_vertices):
            pt1 = vertices[i]
            pt2 = vertices[(i+1) % num_vertices]
            length = np.linalg.norm(pt2 - pt1)
            edge_lengths.append(f"{length:.4f}")
        cv2.polylines(annotated, [approx], True, (0,255,255), 2)
        x, y, w, h = cv2.boundingRect(approx)
        info = f"A:{area:.2f} P:{perimeter:.2f} V:{num_vertices}"
        cv2.putText(annotated, info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    return annotated

def approximate_shape_data(image: np.ndarray, error_threshold: float = 1.0) -> pd.DataFrame:
    gray = Preprocessor.to_grayscale(image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records = []
    region_index = 1
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        best_factor, approx = optimize_epsilon_for_contour(cnt, error_threshold=error_threshold)
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        vertices = approx.reshape(-1, 2)
        num_vertices = len(vertices)
        edge_lengths = []
        for i in range(num_vertices):
            pt1 = vertices[i]
            pt2 = vertices[(i+1) % num_vertices]
            length = np.linalg.norm(pt2 - pt1)
            edge_lengths.append(f"{length:.4f}")
        records.append({
            "region_index": region_index,
            "area": area,
            "perimeter": perimeter,
            "vertices": num_vertices,
            "edge_lengths": ",".join(edge_lengths)
        })
        region_index += 1
    return pd.DataFrame(records)

# (K) Export Segmented Regions
def export_segmented_regions(original_image: np.ndarray, image_path: str) -> int:
    gray = Preprocessor.to_grayscale(original_image)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No segmented regions found.")
    base_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    regions_folder = os.path.join(base_dir, base_name + "_regions")
    os.makedirs(regions_folder, exist_ok=True)
    count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:
            continue
        region = original_image[y:y+h, x:x+w]
        region_filename = os.path.join(regions_folder, f"{base_name}_region_{i+1}.png")
        cv2.imwrite(region_filename, region)
        count += 1
    return count

#####################################
# 4. PIPELINE CLASSES
#####################################

