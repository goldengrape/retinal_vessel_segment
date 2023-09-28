# Required Libraries
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
from skimage.morphology import skeletonize

# Preprocessing Function
import numpy as np
import cv2

def preprocess_retinal_image(original_image: np.ndarray, fov_threshold: float = 0.3, clahe_clip_limit: float = 2.0, clahe_grid_size: tuple = (8, 8)) -> tuple:
    """
    Preprocesses a retinal image by applying a series of image processing techniques.

    Args:
        original_image (np.ndarray): The original retinal image as a numpy array.
        fov_threshold (float, optional): The threshold for the field of view mask. Defaults to 0.3.
        clahe_clip_limit (float, optional): The clip limit for the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm. Defaults to 2.0.
        clahe_grid_size (tuple, optional): The grid size for the CLAHE algorithm. Defaults to (8, 8).

    Returns:
        tuple: A tuple containing the preprocessed image as a numpy array and the field of view mask as a numpy array.
    """
    original_image_np = np.array(original_image)
    green_channel = original_image_np[:, :, 1]
    cielab_image = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2Lab)
    l_channel = cielab_image[:, :, 0]
    _, further_adjusted_fov_mask = cv2.threshold(l_channel, fov_threshold * np.mean(l_channel), 255, cv2.THRESH_BINARY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))

    clahe_green_channel = clahe.apply(green_channel)
    preprocessed_image = cv2.bitwise_and(clahe_green_channel, clahe_green_channel, mask=further_adjusted_fov_mask)
    return preprocessed_image, further_adjusted_fov_mask

# DoG Filter Function
def generate_DoG_filter(size: int, sigma: float) -> np.ndarray:
    """
    Generates a Difference of Gaussian (DoG) filter.

    Args:
        size (int): The size of the filter.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: A numpy array representing the DoG filter.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    gaussian1 = (1 / (2 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian2 = (1 / (2 * np.pi * (0.5 * sigma)**2)) * np.exp(-(xx**2 + yy**2) / (2 * (0.5 * sigma)**2))
    DoG = gaussian1 - gaussian2
    return DoG

# Half-wave Rectification Function
def half_wave_rectification(image: np.ndarray) -> np.ndarray:
    """
    Applies half-wave rectification to an image.

    Args:
        image (np.ndarray): The input image as a numpy array.

    Returns:
        np.ndarray: The rectified image as a numpy array.
    """
    return np.maximum(image, 0)

# Function for the blurring operation
def perform_blurring(rectified_response: np.ndarray, sigma_0: float = 1.0, alpha: float = 0.5, rho_sample: float = 1.0) -> np.ndarray:
    """
    Performs Gaussian blurring on a rectified response.

    Args:
        rectified_response (np.ndarray): The rectified response as a numpy array.
        sigma_0 (float, optional): The initial standard deviation. Defaults to 1.0.
        alpha (float, optional): The scaling factor for the standard deviation. Defaults to 0.5.
        rho_sample (float, optional): The sampling density. Defaults to 1.0.

    Returns:
        np.ndarray: The blurred response as a numpy array.
    """
    # Calculate the blurred standard deviation
    sigma_blurred = sigma_0 + alpha * rho_sample
    # Perform Gaussian blurring
    blurred_response = gaussian_filter(rectified_response, sigma=sigma_blurred)
    return blurred_response

# Function to perform shifting of the blurred response
def perform_shifting(blurred_response: np.ndarray, rho: float, phi: float) -> np.ndarray:
    """
    Performs a shift on a blurred response.

    Args:
        blurred_response (np.ndarray): The blurred response as a numpy array.
        rho (float): The shift distance.
        phi (float): The shift angle in radians.

    Returns:
        np.ndarray: The shifted response as a numpy array.
    """
    # Calculate the x and y shifts
    delta_x = rho * np.cos(phi)
    delta_y = rho * np.sin(phi)
    
    # Perform the shift
    shifted_response = shift(blurred_response, (delta_y, delta_x))
    
    return shifted_response

def calculate_geometric_mean(shifted_responses: list[np.ndarray], sigma: float, rho_values: np.ndarray, t: float = 1) -> np.ndarray:
    """
    Calculates the geometric mean of a list of shifted responses.

    Args:
        shifted_responses (list[np.ndarray]): A list of shifted responses as numpy arrays.
        sigma (float): The standard deviation for the weighting function.
        rho_values (np.ndarray): An array of shift distances.
        t (float, optional): The exponent for the geometric mean. Defaults to 1.

    Returns:
        np.ndarray: The geometric mean response as a numpy array.
    """
    # Initialize variables
    sum_of_weights = 0
    product_of_responses = np.ones_like(shifted_responses[0])

    # Calculate the geometric mean
    for i, shifted_response in enumerate(shifted_responses):
        weight = np.exp(-rho_values[i]**2 / (2 * sigma**2))
        sum_of_weights += weight
        # Make sure the shifted_response values are positive and non-zero
        shifted_response = np.maximum(shifted_response, np.finfo(float).eps)
        product_of_responses *= shifted_response ** weight

    geometric_mean_response = (product_of_responses ** (1 / sum_of_weights))
    geometric_mean_response **= t

    return geometric_mean_response

# Morphological Filtering to Remove Noise
def apply_morphological_filtering(geometric_mean_response: np.ndarray, structuring_element_size: int = 3) -> np.ndarray:
    """
    Applies morphological filtering to a geometric mean response.

    Args:
        geometric_mean_response (np.ndarray): The geometric mean response as a numpy array.
        structuring_element_size (int, optional): The size of the structural element for morphological operation. Defaults to 3.

    Returns:
        np.ndarray: The filtered response as a numpy array.
    """
    # Define the structural element for morphological operation (a square)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))
    
    # Apply top-hat transform
    top_hat_transform = cv2.morphologyEx(geometric_mean_response, cv2.MORPH_TOPHAT, kernel)
    
    return top_hat_transform;

# Binary Thresholding for Vessel Segmentation
def apply_binary_thresholding(morph_filtered_image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Applies binary thresholding to a morphologically filtered image.

    Args:
        morph_filtered_image (np.ndarray): The morphologically filtered image as a numpy array.
        threshold (float, optional): The threshold value. Defaults to 0.3.

    Returns:
        np.ndarray: The binary thresholded image as a numpy array.
    """
    # Normalize the image to [0, 1] range before thresholding
    normalized_image = morph_filtered_image / np.max(morph_filtered_image)
    
    # Apply manual binary thresholding
    _, binary_image = cv2.threshold(normalized_image, threshold, 1, cv2.THRESH_BINARY)
    
    return binary_image

# Post-processing
def apply_post_processing(binary_image: np.ndarray, gap: int = 5) -> np.ndarray:
    """
    Applies post-processing to a binary image to improve vessel connectivity.

    Args:
        binary_image (np.ndarray): The binary image as a numpy array.
        gap (int, optional): The minimum size of regions to remove. Defaults to 5.

    Returns:
        np.ndarray: The post-processed binary image as a numpy array.
    """
    # Step 1: Thinning the binary image using skimage's skeletonize
    thinned_image = skeletonize(binary_image).astype(np.uint8)
    
    # Step 2: Repeat Step 1 to get more complete vessel connectivity
    thinned_image = skeletonize(thinned_image).astype(np.uint8)
    
    # Step 3: Get the intersection between the original and the repeated thinned image
    intersection = np.bitwise_and(binary_image.astype(np.uint8), thinned_image)
    
    # Step 4: Find regions to fill in the original binary image
    filling_regions = np.bitwise_xor(thinned_image, intersection)
    
    # Step 5: Fill the regions in the original binary image
    filled_image = np.bitwise_or(binary_image.astype(np.uint8), filling_regions)
    
    # Step 6: Remove regions smaller than gap pixels
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(filled_image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] >= gap:
            filled_image[output == i] = 1
        else:
            filled_image[output == i] = 0

    return filled_image

def calculate_shifted_and_blurred_DoG_vectorized(C_sigma: np.ndarray, sigma_i: float, rho_i: float, phi_i: float) -> np.ndarray:
    """
    Calculates the shifted and blurred Difference of Gaussian (DoG) vectorized.

    Args:
        C_sigma (np.ndarray): The input image as a numpy array.
        sigma_i (float): The standard deviation for the Gaussian filter.
        rho_i (float): The shift distance.
        phi_i (float): The shift angle in radians.

    Returns:
        np.ndarray: The shifted and blurred DoG response as a numpy array.
    """
    # Calculate the x and y shifts
    delta_x_i = int(round(rho_i * np.cos(phi_i)))
    delta_y_i = int(round(rho_i * np.sin(phi_i)))
    
    # Apply Gaussian blurring to C_sigma
    C_sigma_blurred = gaussian_filter(C_sigma, sigma=sigma_i)
    
    # Perform the shift
    rows, cols = C_sigma.shape
    S_sigma_rho_phi = np.zeros_like(C_sigma)
    S_sigma_rho_phi[max(0, delta_x_i):min(rows, rows + delta_x_i), max(0, delta_y_i):min(cols, cols + delta_y_i)] = \
        C_sigma_blurred[max(0, -delta_x_i):min(rows, rows - delta_x_i), max(0, -delta_y_i):min(cols, cols - delta_y_i)]
    
    return S_sigma_rho_phi

# Function to extract retinal vessels
def extract_vessle(
        original_image: np.ndarray, fov_threshold: float, clahe_clip_limit: float, clahe_grid_size: tuple[int, int], DoG_size: int, DoG_sigma: float, blurring_sigma: float, blurring_alpha: float, blurring_rho_sample: float, shifting_phi_sample: float, geometric_mean_sigma: float, geometric_mean_t: float, structuring_element_size: int, binary_threshold: float, post_processing_gap: int, resize_size: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Extracts retinal vessels from an input image.

    Args:
        original_image (np.ndarray): The input image as a numpy array.
        fov_threshold (float): The threshold for field of view detection.
        clahe_clip_limit (float): The clip limit for Contrast Limited Adaptive Histogram Equalization (CLAHE).
        clahe_grid_size (tuple[int, int]): The grid size for CLAHE.
        DoG_size (int): The size of the Difference of Gaussian (DoG) filter.
        DoG_sigma (float): The standard deviation for the DoG filter.
        blurring_sigma (float): The standard deviation for the Gaussian filter used in blurring.
        blurring_alpha (float): The alpha value for the blurring function.
        blurring_rho_sample (float): The sampling distance for blurring.
        shifting_phi_sample (float): The sampling angle for shifting.
        geometric_mean_sigma (float): The standard deviation for the weighting function used in geometric mean calculation.
        geometric_mean_t (float): The exponent for the geometric mean calculation.
        structuring_element_size (int): The size of the structural element for morphological operation.
        binary_threshold (float): The threshold value for binary thresholding.
        post_processing_gap (int): The minimum size of regions to remove in post-processing.
        resize_size (int): The size to resize the image to.

    Returns:
        tuple[list[np.ndarray], tuple[int, int]]: A tuple containing a list of images as numpy arrays and the original image size as a tuple.
    """
    # Read the input image using cv2 and convert from BGR to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Resize the image if necessary
    original_size = original_image.shape
    if resize_size > 0:
        original_image = cv2.resize(original_image, (resize_size, resize_size))

    # Preprocess the image
    preprocessed_img, adjusted_fov_mask = preprocess_retinal_image(original_image, fov_threshold, clahe_clip_limit, clahe_grid_size)

    # Generate DoG filter and apply convolution
    DoG_filter = generate_DoG_filter(DoG_size, DoG_sigma)
    response = convolve2d(preprocessed_img, DoG_filter, mode='same', boundary='symm')

    # Perform half-wave rectification
    rectified_response = half_wave_rectification(response)

    # Perform blurring
    blurred_response = perform_blurring(rectified_response, blurring_sigma, blurring_alpha, blurring_rho_sample)

    # Perform shifting operation
    shifted_response = perform_shifting(blurred_response, blurring_rho_sample, shifting_phi_sample)

    # Calculate shifted and blurred DoG response
    shifted_and_blurred_DoG = calculate_shifted_and_blurred_DoG_vectorized(blurred_response, DoG_sigma, blurring_rho_sample, shifting_phi_sample)

    # Sample shifted responses and rho values
    sample_shifted_responses = [shifted_and_blurred_DoG for _ in range(int(np.pi/shifting_phi_sample+1))]
    sample_rho_values = [blurring_rho_sample for _ in range(int(np.pi/shifting_phi_sample+1))]

    # Calculate the geometric mean of the shifted responses
    geometric_mean_response = calculate_geometric_mean(sample_shifted_responses, geometric_mean_sigma, sample_rho_values, geometric_mean_t)

    # Apply morphological filtering to the geometric mean response image
    morph_filtered_image = apply_morphological_filtering(geometric_mean_response, structuring_element_size)

    # Apply binary thresholding to the morphologically filtered image
    binary_image = apply_binary_thresholding(morph_filtered_image, binary_threshold)

    # Apply post-processing steps to the binary image
    post_processed_image = apply_post_processing(binary_image, post_processing_gap)

    # Mark the vessels on the original image
    marked_image = np.copy(original_image)
    marked_image[post_processed_image == 1] = [0, 255, 0]

    # Convert post_processed_image to RGB
    post_processed_image = cv2.cvtColor(post_processed_image.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

    # Create a list of images for output
    imagelist = [original_image, preprocessed_img, rectified_response, blurred_response, shifted_response, geometric_mean_response, morph_filtered_image, binary_image, post_processed_image, marked_image]

    return imagelist, original_size