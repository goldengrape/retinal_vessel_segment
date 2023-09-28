# Required Libraries
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
from skimage.morphology import skeletonize

# Preprocessing Function
def preprocess_retinal_image(original_image, fov_threshold=0.3, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    original_image_np = np.array(original_image)
    green_channel = original_image_np[:, :, 1]
    cielab_image = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2Lab)
    l_channel = cielab_image[:, :, 0]
    _, further_adjusted_fov_mask = cv2.threshold(l_channel, fov_threshold * np.mean(l_channel), 255, cv2.THRESH_BINARY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    clahe_green_channel = clahe.apply(green_channel)
    preprocessed_image = cv2.bitwise_and(clahe_green_channel, clahe_green_channel, mask=further_adjusted_fov_mask)
    return preprocessed_image, further_adjusted_fov_mask

# DoG Filter Function
def generate_DoG_filter(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    gaussian1 = (1 / (2 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian2 = (1 / (2 * np.pi * (0.5 * sigma)**2)) * np.exp(-(xx**2 + yy**2) / (2 * (0.5 * sigma)**2))
    DoG = gaussian1 - gaussian2
    return DoG

# Half-wave Rectification Function
def half_wave_rectification(image):
    return np.maximum(image, 0)

# Function for the blurring operation
def perform_blurring(rectified_response, sigma_0=1.0, alpha=0.5, rho_sample=1.0):
    # Calculate the blurred standard deviation
    sigma_blurred = sigma_0 + alpha * rho_sample
    # Perform Gaussian blurring
    blurred_response = gaussian_filter(rectified_response, sigma=sigma_blurred)
    return blurred_response

# Function to perform shifting of the blurred response
def perform_shifting(blurred_response, rho, phi):
    # Calculate the x and y shifts
    delta_x = rho * np.cos(phi)
    delta_y = rho * np.sin(phi)
    
    # Perform the shift
    shifted_response = shift(blurred_response, (delta_y, delta_x))
    
    return shifted_response

def calculate_geometric_mean(shifted_responses, sigma, rho_values, t=1):
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
def apply_morphological_filtering(geometric_mean_response, structuring_element_size=3):
    # Define the structural element for morphological operation (a square)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))
    
    # Apply top-hat transform
    top_hat_transform = cv2.morphologyEx(geometric_mean_response, cv2.MORPH_TOPHAT, kernel)
    
    return top_hat_transform

# Binary Thresholding for Vessel Segmentation
def apply_binary_thresholding(morph_filtered_image, threshold=0.3):
    # Normalize the image to [0, 1] range before thresholding
    normalized_image = morph_filtered_image / np.max(morph_filtered_image)
    
    # Apply manual binary thresholding
    _, binary_image = cv2.threshold(normalized_image, threshold, 1, cv2.THRESH_BINARY)
    
    return binary_image

# Post-processing
def apply_post_processing(binary_image,gap=5):
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

def calculate_shifted_and_blurred_DoG_vectorized(C_sigma, sigma_i, rho_i, phi_i):
    # 计算偏移量
    delta_x_i = int(round(rho_i * np.cos(phi_i)))
    delta_y_i = int(round(rho_i * np.sin(phi_i)))
    
    # 首先，使用高斯滤波对C_sigma进行模糊
    C_sigma_blurred = gaussian_filter(C_sigma, sigma=sigma_i)
    
    # 然后，进行偏移
    rows, cols = C_sigma.shape
    S_sigma_rho_phi = np.zeros_like(C_sigma)
    S_sigma_rho_phi[max(0, delta_x_i):min(rows, rows + delta_x_i), max(0, delta_y_i):min(cols, cols + delta_y_i)] = \
        C_sigma_blurred[max(0, -delta_x_i):min(rows, rows - delta_x_i), max(0, -delta_y_i):min(cols, cols - delta_y_i)]
    
    return S_sigma_rho_phi

def extract_vessle(original_image,
                   fov_threshold,clahe_clip_limit,clahe_grid_size,DoG_size,DoG_sigma,blurring_sigma,blurring_alpha,blurring_rho_sample,shifting_phi_sample,geometric_mean_sigma,geometric_mean_t,structuring_element_size,binary_threshold,post_processing_gap,resize_size):

    # 用cv2读取图片, 读取的图片是BGR格式, 需要转换为RGB格式
    # original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # 调整图片大小，以免后续计算量过大
    original_size=original_image.shape
    if resize_size>0:
        original_image = cv2.resize(original_image, (resize_size,resize_size))

    preprocessed_img, adjusted_fov_mask = preprocess_retinal_image(original_image, fov_threshold, clahe_clip_limit, (clahe_grid_size,clahe_grid_size))
    # Generate DoG filter and apply convolution
    DoG_filter = generate_DoG_filter(DoG_size, DoG_sigma)
    response = convolve2d(preprocessed_img, DoG_filter, mode='same', boundary='symm')
    # Perform half-wave rectification
    rectified_response = half_wave_rectification(response)
    # Perform blurring
    blurred_response = perform_blurring(rectified_response, blurring_sigma, blurring_alpha, blurring_rho_sample)

    # Perform shifting operation
    shifted_response = perform_shifting(blurred_response, blurring_rho_sample, shifting_phi_sample)

    shifted_and_blurred_DoG = calculate_shifted_and_blurred_DoG_vectorized(blurred_response, DoG_sigma, blurring_rho_sample, shifting_phi_sample)


    # Sample shifted responses and rho values for this example
    sample_shifted_responses = [shifted_and_blurred_DoG  for _ in range(int(np.pi/shifting_phi_sample+1))]
    # sample_rho_values = [rho_sample for _ in range(int(np.pi/phi_sample+1))]
    sample_rho_values = [blurring_rho_sample for _ in range(int(np.pi/shifting_phi_sample+1))]

    # Calculate the geometric mean of the shifted responses
    geometric_mean_response = calculate_geometric_mean(sample_shifted_responses, geometric_mean_sigma, sample_rho_values, geometric_mean_t)

    # Apply morphological filtering to the geometric mean response image
    morph_filtered_image = apply_morphological_filtering(geometric_mean_response, structuring_element_size)

    # Apply binary thresholding to the morphologically filtered image
    binary_image = apply_binary_thresholding(morph_filtered_image, binary_threshold)

    # Apply post-processing steps to the binary image
    post_processed_image = apply_post_processing(binary_image, post_processing_gap)

    

    marked_image = np.copy(original_image)
    marked_image[post_processed_image == 1] = [0, 255, 0]

    # 将post_processed_image转换为RGB
    post_processed_image = cv2.cvtColor(post_processed_image.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

    imagelist = [original_image, preprocessed_img, rectified_response, blurred_response, shifted_response, geometric_mean_response, morph_filtered_image, binary_image,post_processed_image,marked_image]


    return imagelist, original_size