Design Document: Laser-Induced Metal Dewetting Analysis Module (LIMDA)
1. Purpose and Scope

The LIMDA module quantifies morphological and optical parameters in images of laser-processed metal films, especially when circular features (droplets, pits, or voids) coexist with striated or textured backgrounds.

It provides automated extraction of:

Particle-level morphology (size, circularity, rim contrast, etc.)

Ensemble statistics (density, area fraction, uniformity)

Spatial organization (clustering metrics)

Optical contrast (color and intensity analysis)

Background texturing (striation direction, frequency, anisotropy)

The design assumes optical microscopy images with a scale bar (converted to µm per pixel).

2. Inputs and Preprocessing

Input types:

.png, .tif, .jpg, .bmp images

Optional metadata file (e.g., pixel size, magnification)

User-specified or auto-detected scale bar

Preprocessing pipeline:

Scale calibration – manual or automatic detection of scale bar (e.g., using color/contrast heuristics) → µm/pixel.

Region of Interest (ROI) selection – optionally mask out scale bar and text.

Illumination correction – background flattening using Gaussian blur subtraction or rolling-ball method.

Noise filtering – bilateral or non-local means denoising to preserve edges.

Color space conversion – RGB → LAB or grayscale, depending on analysis path.

3. Module Architecture
3.1 Core Components

ParticleAnalyzer – segmentation, morphology, and optical feature extraction.

ClusterAnalyzer – spatial distribution and clustering statistics.

TextureAnalyzer – Fourier- and GLCM-based striation characterization.

OpticalAnalyzer – color contrast and reflectance-like metrics.

Visualizer – annotated overlays, histograms, and frequency maps.

ReportGenerator – produces summarized tables, JSON, and optional Excel reports.

Each analyzer operates as a modular class with a consistent API (analyze(image, mask, config) → returns dictionary of results + debug layers).

4. Algorithms and Metrics
4.1 Particle Analysis

Segmentation strategy:

Adaptive thresholding (Otsu or Sauvola) on smoothed grayscale.

Morphological opening to remove debris.

Watershed or distance transform for touching particles.

Contour extraction via OpenCV.

Extracted metrics:

Equivalent diameter (µm)

Area (µm²)

Circularity (4πA/P²)

Aspect ratio

Dark rim/total area ratio (via inner vs outer intensity profile)

Mean intensity (core, rim, background)

Rim-to-core intensity ratio (Iᵣ / Iₚ)

Color difference (ΔE) vs background

Centroid coordinates

Outputs:

Size histogram

Circularity vs size scatter

Annotated image (overlayed contours and IDs)

4.2 Ensemble and Statistical Metrics

Computed after particle extraction:

Number density: N / area

Total area fraction (coverage)

Coefficient of variation for size and circularity

Distribution fit (log-normal, Gaussian)

Bimodality coefficient

4.3 Spatial Organization (Clustering)

Using centroid coordinates:

Nearest-neighbor distance (NND): mean, σ, histogram.

Ripley’s K / L-function: test for clustering vs random distribution.

Voronoi tessellation: area variance as local uniformity measure.

Moran’s I or Getis-Ord G: spatial autocorrelation coefficient.

Fractal dimension (box-counting): measures hierarchical aggregation.

Outputs:

Pair correlation function plot g(r)

Spatial uniformity heatmap

4.4 Optical and Color Contrast Analysis

Color evaluation in CIE LAB space:

Mean LAB values for background, rim, core.

ΔE (CIEDE2000) between rim and background.

Relative luminance (Y channel) and contrast ratios.

Hue and saturation variance maps.

Outputs:

ΔE histograms and false-color maps showing high-contrast zones.

4.5 Background Striation Analysis

Fourier-domain analysis:

Compute 2D FFT of background (droplet-masked image).

Find peak direction (θ) and dominant spatial frequency (fₘₐₓ).

Convert to striation spacing λ = 1/fₘₐₓ (µm).

Compute anisotropy ratio = power(θₘₐₓ) / mean isotropic power.

Spatial-domain texture metrics:

GLCM features: contrast, homogeneity, entropy.

Gradient directionality histograms.

Local Binary Pattern (LBP) entropy as microtexture descriptor.

Background roughness index: std. of local gradient magnitudes.

Outputs:

Orientation histogram.

Striation spacing distribution.

Anisotropy heatmap.

5. Outputs and Reports

Data outputs:

.json file with all computed metrics (hierarchical structure)

.xlsx with:

Per-particle data

Summary statistics

Color/contrast tables

Spatial distribution summary

Striation analysis table

Visual outputs:

Annotated images (particle boundaries, ID labels)

Histograms: size, circularity, contrast

FFT power spectrum image

Cluster density heatmap

Striation orientation map

6. User Interface Concepts

For your modular microscopy app (Python-based):

GUI panel: “Laser Dewetting Analysis” tab.

Adjustable parameters:

Segmentation threshold sensitivity.

Minimum particle size.

Color model (grayscale or LAB).

FFT window size and mask radius.

Checkboxes for:

Include striation analysis

Include clustering metrics

Export Excel / JSON / overlay image

Live preview of detected particles and striation axes.

7. Implementation Dependencies

Python libraries:

numpy, scipy, pandas

opencv-python

scikit-image

matplotlib

Pillow

colormath (for ΔE)

openpyxl (for Excel export)

pyfftw (optional FFT acceleration)

Optional:

PyQt5 or tkinter for GUI

Plotly for interactive visualizations

8. Validation and Calibration

To ensure reliability:

Validate scale calibration using test grids.

Compare particle segmentation vs manual annotation (IoU > 0.8 target).

Verify FFT striation direction against known laser scan direction.

Perform repeatability test on multiple micrographs from the same sample.

9. Extensibility and Future Work

Possible expansions:

Integrate multi-wavelength or dark-field microscopy data for optical depth mapping.

Add machine learning classifier for feature type (droplet vs pore).

Include time-series mode for laser parameter optimization experiments.

Add 3D data fusion if confocal depth slices become available.

10. Expected Deliverables

Source module: limda.py

Configuration template: limda_config.json

Example dataset + analysis notebook

User documentation: parameter definitions, workflow examples

Optional: GUI integration hooks for your main image-processing environment
