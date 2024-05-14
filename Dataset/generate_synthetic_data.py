
import os
import argparse
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from noise import pnoise2
from skimage.draw import line_aa
from scipy.ndimage import rotate, gaussian_filter
from skimage.io import imread


# Normalize the pixels [0,1]
def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))
################################################################################################################

## Perlins noise to generate background texture
def generate_texture_patch(width, height, scale=100.0, octaves=1, persistence=0.5, lacunarity=2.0, base=0):
    texture = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            texture[i][j] = pnoise2(i / scale,
                                     j / scale,
                                     octaves=octaves,
                                     persistence=persistence,
                                     lacunarity=lacunarity,
                                     base=base)
    return normalize(texture)

# Color maps for texture
color_maps_set1 = [
    ("dark_green", mcolors.LinearSegmentedColormap.from_list('dark_green', ['#003300', '#004d00', '#006600'], N=256)),
    ("green_yellow", mcolors.LinearSegmentedColormap.from_list('green_yellow', ['#007000', '#b8cc34', '#dfff00'], N=256)),
    ("light_green", mcolors.LinearSegmentedColormap.from_list('light_green', ['#70e000', '#afff50', '#efff81'], N=256))
] ### too much yellow for cracks to be visible, not used for generation

color_maps_set2 = [
    ("dark_green_to_black",mcolors.LinearSegmentedColormap.from_list('darker_dark_green', ['#001a00', '#003300', '#004d00'], N=256)),
    ("mid_tone_green", mcolors.LinearSegmentedColormap.from_list('darker_green_yellow', ['#004d00', '#336600', '#4d7f00'], N=256)),
    ("dark_vibrant_green", mcolors.LinearSegmentedColormap.from_list('darker_light_green', ['#407000', '#70e000', '#afff50'], N=256))
] ### sea green with green shades

color_maps_set3 = [
    ("dark_blue_to_dark_green", mcolors.LinearSegmentedColormap.from_list('dark_blue_to_dark_green', ['#000033', '#002b36', '#004d00'], N=256)),
    ("dark_green_to_green", mcolors.LinearSegmentedColormap.from_list('dark_green_to_green', ['#004d00', '#007000', '#009900'], N=256)),
    ("green_to_light_green", mcolors.LinearSegmentedColormap.from_list('green_to_light_green', ['#009900', '#33cc33', '#66ff66'], N=256))
] ### pure green colour shades

color_maps_set4 = [
    ("dark_blue_to_black", mcolors.LinearSegmentedColormap.from_list('dark_blue_to_black', ['#000033', '#000022', '#000000'], N=256)),
    ("dark_blue_to_dark_green", mcolors.LinearSegmentedColormap.from_list('dark_blue_to_dark_green', ['#000033', '#002200', '#004d00'], N=256)),
        ("green_yellow", mcolors.LinearSegmentedColormap.from_list('green_yellow', ['#007000', '#b8cc34', '#dfff00'], N=256))  
] ### dark blue-black with mild green shades 

color_maps_set5 = [
    ("deep_green_to_olive", mcolors.LinearSegmentedColormap.from_list('deep_green_to_olive', ['#1B3D00', '#345D00', '#597D00'], N=256)),
    ("olive_to_bright_green", mcolors.LinearSegmentedColormap.from_list('olive_to_bright_green', ['#597D00', '#749A00', '#95BF00'], N=256)),
    ("bright_green_to_yellowish_green", mcolors.LinearSegmentedColormap.from_list('bright_green_to_yellowish_green', ['#95BF00', '#AACC00', '#C0E036'], N=256))
] ### yellowish-lemon shade

color_maps_set6 = [
    ("deep_forest_green_to_leaf_green", mcolors.LinearSegmentedColormap.from_list('deep_forest_green_to_leaf_green', ['#0B2800', '#204D00', '#397200'], N=256)),
    ("leaf_green_to_lime_green", mcolors.LinearSegmentedColormap.from_list('leaf_green_to_lime_green', ['#397200', '#539A00', '#6EBF00'], N=256)),
    ("lime_green_to_greenish_yellow", mcolors.LinearSegmentedColormap.from_list('lime_green_to_greenish_yellow', ['#6EBF00', '#84CC00', '#9FDAA3'], N=256))
] ### light green shade with hint of lemon

color_maps_set7 = [
    ("dark_olive_to_dark_green", mcolors.LinearSegmentedColormap.from_list('dark_olive_to_dark_green', ['#3B4D00', '#1E6600', '#008000'], N=256)),
    ("dark_green_to_medium_green", mcolors.LinearSegmentedColormap.from_list('dark_green_to_medium_green', ['#008000', '#339933', '#66cc66'], N=256)),
    ("medium_green_to_light_green", mcolors.LinearSegmentedColormap.from_list('medium_green_to_light_green', ['#66cc66', '#99e699', '#ccffcc'], N=256))
] ### greenish-white shades


## To blend cmaps for texture
def blend_colormaps_noise(noise, cmap1, cmap2, cmap3, transition_width=0.1, randomness=0.1):
    if noise < 0.5 - transition_width / 2:
                t = (noise + np.random.uniform(-randomness, randomness)) / (0.5 - transition_width / 2)
                color = cmap1(max(0, min(1, t)))  # Clamp t to the range [0, 1]
    elif noise > 0.5 + transition_width / 2:
                t = (noise - (0.5 + transition_width / 2) + np.random.uniform(-randomness, randomness)) / (0.5 - transition_width / 2)
                color = cmap3(max(0, min(1, t)))
    else:  # In the middle, smoothly blend between the first and third color maps
                t = (noise - (0.5 - transition_width / 2) + np.random.uniform(-randomness, randomness)) / transition_width
                color = cmap2(max(0, min(1, t)))
    return color

## To smoothly overlap the patches
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define values and probabilities for each parameter to generate texture (patches)
scale_options = [lambda: np.random.randint(8, 12), lambda: np.random.randint(12, 18), lambda: np.random.randint(18, 25)]
scale_probabilities = [0.85, 0.10, 0.05]

octaves_options = [2, 5, 10, 20, 30]
octaves_probabilities = [0.05, 0.1, 0.1, 0.7, 0.05]

persistence_options = [lambda: np.random.uniform(0.47, 0.49), lambda: np.random.uniform(0.49, 0.51), lambda: np.random.uniform(0.51, 0.53)]
persistence_probabilities = [0.1, 0.8, 0.1]

lacunarity_options = [1.8, 2.0, 2.2]
lacunarity_probabilities = [0.05, 0.9, 0.05]

base_options = [lambda: np.random.randint(0, 100), lambda: np.random.randint(100, 200)]
base_probabilities = [1, 0.0]

## To create patches
def create_texture(final_width, final_height, patch_size, overlap,color_maps):
    # Calculate the number of patches needed
    num_patches_x = int(np.ceil(final_width / (patch_size - overlap)))
    num_patches_y = int(np.ceil(final_height / (patch_size - overlap)))

    # Create an empty canvas
    full_texture = np.zeros((final_height, final_width))

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Generate a random patch
            scale = random.choices(scale_options, scale_probabilities)[0]()
            octaves = random.choices(octaves_options, octaves_probabilities)[0]
            persistence = random.choices(persistence_options, persistence_probabilities)[0]()
            lacunarity = random.choices(lacunarity_options, lacunarity_probabilities)[0]
            base = random.choices(base_options, base_probabilities)[0]()

            # Generate a patch with the selected parameters
            #print(patch_size, patch_size, scale, octaves, persistence, lacunarity, base)
            patch = generate_texture_patch(patch_size, patch_size, scale, octaves, persistence, lacunarity, base)
            
            # Calculate where to place the patch (with overlap)
            x_start = j * (patch_size - overlap)
            y_start = i * (patch_size - overlap)

            # Blend the patch into the canvas
            for y in range(patch_size):
                for x in range(patch_size):
                    if 0 <= y_start + y < final_height and 0 <= x_start + x < final_width:
                        alpha = 1.0  # No blending on the first patch
                        if x < overlap and x_start > 0:  # Blend horizontally
                            alpha = x / overlap
                        if y < overlap and y_start > 0:  # Blend vertically
                            alpha = y / overlap
                        full_texture[y_start + y, x_start + x] = (alpha * patch[y, x] +
                                                                  (1 - alpha) * full_texture[y_start + y, x_start + x])
            for y in range(patch_size):
                for x in range(patch_size):
                    if 0 <= y_start + y < final_height and 0 <= x_start + x < final_width:
                        alpha_x = alpha_y = 1.0  # No blending on the first patch
                        if x_start > 0:  # Blend horizontally
                            alpha_x = sigmoid((x / overlap - 0.5) * 10)  # Adjust the 10 to control the steepness of the transition
                        if y_start > 0:  # Blend vertically
                            alpha_y = sigmoid((y / overlap - 0.5) * 10)  # Adjust the 10 to control the steepness of the transition
                        alpha = alpha_x * alpha_y  # Combine horizontal and vertical alphas
                        full_texture[y_start + y, x_start + x] = (alpha * patch[y, x] +
                                                      (1 - alpha) * full_texture[y_start + y, x_start + x])

    # Apply color map to the full texture
    #print(full_texture.shape)
    colored_texture = np.zeros((final_height, final_width, 3))
    for i in range(final_height):
        for j in range(final_width):
            colored_texture[i, j] = blend_colormaps_noise(full_texture[i, j],color_maps[0][1], color_maps[1][1], color_maps[2][1])[:3]

    return colored_texture


##########################################################################################################################
# Create crack image
#def generate_crack_path(width, height, max_step=2, stability_span=10):
#    x = np.arange(width)
#    y = np.full(width, np.random.randint(0, height))
#    last_large_change_index = -stability_span  # Initialize to a value outside of the array index
#    for i in range(1, width):
#        if i - last_large_change_index < stability_span:
#            step_options = np.arange(-max_step, max_step)
#            step_options = step_options[step_options != -max_step]
#            step_options = step_options[step_options != max_step]
#            variation = np.random.choice(step_options)
#        else:
#            variation = np.random.randint(-max_step, max_step + 1)
#        y_proposed = y[i-1] + variation
#        if abs(variation) == max_step:
#            last_large_change_index = i
#        y[i] = np.clip(y_proposed, 0, height - 1)
#    return x, y

#def generate_crack_path(width, height, max_step=1, stability_span=200):
#    x = np.arange(width)
#    y = np.full(width, np.random.randint(0, height))
    
#    # Smoothness parameters
#    smoothness = max_step * 20  # The higher, the smoother the crack
#    curve_variation = stability_span  # The higher, the less frequent the direction changes
    
#    direction = 0  # Current direction of the crack
#    for i in range(1, width):
#        # Randomly change direction, but keep changes small for smoothness
#        direction += np.random.uniform(-max_step/smoothness, max_step/smoothness)
#        # Clamp the direction change to maintain stability and less vertical movement
#        direction = np.clip(direction, -max_step/curve_variation, max_step/curve_variation)
#        y_proposed = y[i-1] - direction

        # If y_proposed is at the edge, stop the crack generation
#        if y_proposed <= 0 or y_proposed >= height - 1:
#            x = x[:i]  # Trim the x array to the current length
#            y[i] = np.clip(y_proposed, 0, height - 1)  # Ensure y stays within bounds
##            break  # Exit the loop as we've hit the boundary
#       else:
#            y[i] = y_proposed
#
#    return x, y

def generate_crack_path(width, height, max_step=1, stability_span=200):
    x = np.arange(width)
    y_start = np.random.choice([height//32,height//16,height//8,height//4],p=[0,0.35,0.35,0.3])
    #print(y_start)
    y = np.full(width, np.random.randint(y_start, height))
    
    # Smoothness parameters
    max_step = np.random.choice([1,4,10],p=[0.7,0.28,0.02])
    smoothness = max_step * 20  # The higher, the smoother the crack
    curve_variation = stability_span  # The higher, the less frequent the direction changes

    direction = 0  # Current direction of the crack
    for i in range(1, width):
        # Randomly change direction, but keep changes small for smoothness
        direction += np.random.uniform(-max_step/smoothness, max_step/smoothness)
        # Clamp the direction change to maintain stability and less vertical movement
        direction = np.clip(direction, -max_step/curve_variation, max_step/curve_variation)
        y_proposed = y[i-1] - direction

        # If y_proposed is at the edge, stop the crack generation
        if y_proposed <= 0 or y_proposed >= height - 1:
            x = x[:i]  # Trim the x array to the current length
            y[i] = np.clip(y_proposed, 0, height - 1)  # Ensure y stays within bounds
            break  # Exit the loop as we've hit the boundary
        else:
            y[i] = y_proposed

    return x, y


def draw_crack(image, path_func, line_color, rect_region, max_width=2):
    start_x, start_y, end_x, end_y = rect_region
    width = end_x - start_x
    height = end_y - start_y
    x, y = path_func(width, height)

    # Draw crack image
    crack_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, len(x)):
        num_lines = np.random.randint(1, max_width + 1)
        for j in range(-num_lines // 2, num_lines // 2 + 1):
            rr, cc, val = line_aa(y[i-1] + j, x[i-1], y[i] + j, x[i])
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)
            crack_image[rr, cc] = np.maximum(crack_image[rr, cc], val * 255)

    #plt.imshow(crack_image)
    #plt.show()
    
    # Apply padding to the crack image to prevent clipping during rotation
    padding = min(image.shape)  # The padding size can be adjusted if necessary
    padded_crack_image = np.pad(crack_image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Define angle range
    angle_ranges = [
        (150, 160), (330, 340),  # Horizontal ranges
        (0,  75), (75, 150), (160, 330), (340, 360)  # Diagonal ranges 
        ]
    probabilities = [0.4, 0.4,  
        0.05, 0.05, 0.05, 0.05]

    probabilities = [p / sum(probabilities) for p in probabilities]
    selected_range = np.random.choice(len(angle_ranges), p=probabilities)
    angle = np.random.uniform(angle_ranges[selected_range][0], angle_ranges[selected_range][1])
    #print(angle)
    # Rotate the image with expansion allowed, using nearest-neighbor interpolation
    rotate_crack = np.random.choice([True, False], p=[0.9, 0.1])  # 90% chance to rotate
    if rotate_crack:
        rotated_padded_crack = rotate(padded_crack_image, angle, reshape=True, order=0, mode='constant', cval=0)
    else: 
        rotated_padded_crack = rotate(padded_crack_image, 0, reshape=True, order=0, mode='constant', cval=0)
    #plt.imshow(rotated_padded_crack)
    #plt.show()

    # Find the new bounding box of the rotated crack within the padded image
    non_zero_coords = np.argwhere(rotated_padded_crack > 0)
    if non_zero_coords.size == 0:
        return image,()
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    rotated_crack_cropped = rotated_padded_crack[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]
    #plt.imshow(rotated_crack_cropped)
    #plt.show()

    # Determine placement of the cropped rotated crack
    # Center of the rectangle region in the original image
    #rect_center_y, rect_center_x = (start_y + end_y) // 2, (start_x + end_x) // 2
    # Center of the cropped rotated crack
    #crack_center_y, crack_center_x = rotated_crack_cropped.shape[0] // 2, rotated_crack_cropped.shape[1] // 2
    # Calculate top-left coordinates for placing the cropped rotated crack
    #top_left_y = rect_center_y - crack_center_y
    #top_left_x = rect_center_x - crack_center_x

    # Calculate random top-left coordinates for placing the cropped rotated crack
    max_top_left_y = image.shape[0] - rotated_crack_cropped.shape[0]
    max_top_left_x = image.shape[1] - rotated_crack_cropped.shape[1]

    top_left_y = np.random.randint(0, max(1, max_top_left_y))
    top_left_x = np.random.randint(0, max(1, max_top_left_x))

    # Place the cropped rotated crack onto the original image
    for r in range(rotated_crack_cropped.shape[0]):
        for c in range(rotated_crack_cropped.shape[1]):
            if rotated_crack_cropped[r, c] > 0:
                pos_r = np.clip(top_left_y + r, 0, image.shape[0]-1)
                pos_c = np.clip(top_left_x + c, 0, image.shape[1]-1)
                intensity_factor = rotated_crack_cropped[r, c] / 255
                image[pos_r, pos_c, :] = (1 - intensity_factor) * image[pos_r, pos_c, :] + intensity_factor * line_color

    # Find non-zero pixels in the final image
    non_zero_coords = np.argwhere(image[:, :, 0] != 255)  # Assuming the background is white
    if non_zero_coords.size == 0:
        return image,()
    
    # Get the extents of the non-zero regions
    min_y, min_x = non_zero_coords.min(axis=0)
    max_y, max_x = non_zero_coords.max(axis=0)

    # Compute the coordinates for the bounding box
    bbox_top_left = (min_x, min_y)
    bbox_bottom_right = (max_x, max_y)

    # Adjust the bounding box to be inside the actual edge by one pixel if it's aligning with the main image edges
    if bbox_top_left[0] == 0:
        bbox_top_left = (1, bbox_top_left[1])
    if bbox_top_left[1] == 0:
        bbox_top_left = (bbox_top_left[0], 1)
    if bbox_bottom_right[0] == image.shape[1] - 1:
        bbox_bottom_right = (image.shape[1] - 2, bbox_bottom_right[1])
    if bbox_bottom_right[1] == image.shape[0] - 1:
        bbox_bottom_right = (bbox_bottom_right[0], image.shape[0] - 2)

    # Calculate the normalized bounding box coordinates for YOLOv5
    bbox_width = (max_x - min_x)
    bbox_height = (max_y - min_y)
    bbox_x_center = (min_x + bbox_width / 2) 
    bbox_y_center = (min_y + bbox_height / 2)

    # Class label for cracks is assumed to be 0
    class_label = 0

    # YOLOv5 annotation for the object
    yolov5_annotation = (class_label, bbox_x_center, bbox_y_center, bbox_width, bbox_height)
    bbox_width = bbox_width / image.shape[1]
    bbox_height = bbox_height/ image.shape[0]
    bbox_x_center = bbox_x_center / image.shape[1]
    bbox_y_center = bbox_y_center/ image.shape[0]
    yolov5_annotation_norm = (class_label, bbox_x_center, bbox_y_center, bbox_width, bbox_height)

    # Convert to string format for printing
    #annotation_str = ' '.join(map(str, yolov5_annotation))
    #print(annotation_str)

    return image, yolov5_annotation_norm

####################################################################################################

## Directly add crack to texture
def blend_images_direct(texture_path, crack_image, crack_color):

    # Load the texture image
    texture = imread(texture_path)

    # Ensure the texture is in RGB format
    if texture.ndim == 2:
        texture = np.stack((texture,) * 3, axis=-1)
    elif texture.shape[2] > 3:
        texture = texture[:, :, :3]  # Drop alpha channel if it exists

    # Find where the crack is drawn by checking where the crack image is not white
    crack_mask = np.any(crack_image != [255, 255, 255], axis=-1)

    # Apply the crack color to these positions on the texture
    texture[crack_mask] = crack_color

    return texture


## Add the crack in a processed manner
def blend_images_processed(texture_path, crack_image, crack_color,sigma=1.0):
    """
    Blends the crack image with the texture based on the crack's presence, modulating the crack color intensity randomly.

    Parameters:
    - texture_path: Path to the texture image file.
    - crack_image: The numpy array of the crack image.
    - crack_color: The RGB color used for the crack.

    Returns:
    - A numpy array of the blended image.
    """
    # Load the texture image
    texture = imread(texture_path)
    
    # Ensure the texture is in RGB format
    if texture.ndim == 2:
        texture = np.stack((texture,) * 3, axis=-1)
    elif texture.shape[2] > 3:
        texture = texture[:, :, :3]  # Drop alpha channel if it exists

    # Find where the crack is drawn by checking where the crack image is not white
    crack_mask = np.any(crack_image != [255, 255, 255], axis=-1)
    #crack_mask = gaussian_filter(crack_mask, sigma)

    # Modulate the crack color randomly before applying it
    modulated_crack_color = np.array([
        np.random.choice([crack_color[0]*0.8, crack_color[0]*0.9, crack_color[0], crack_color[0]*1.1], p=[0.00,0.25,0.75,0.0]),
        np.random.choice([crack_color[1]*0.8, crack_color[1]*0.9, crack_color[1], crack_color[1]*1.1], p=[0.00,0.25,0.75,0.0]),
        np.random.choice([crack_color[2]*0.8, crack_color[2]*0.9, crack_color[2], crack_color[2]*1.1], p=[0.00,0.25,0.75,0.0])
    ]).astype(np.uint8)

     # Create a new image for the crack, initially set to the texture
    crack_applied = texture.copy()

    # Apply the crack color to these positions on the crack_applied image
    crack_applied[crack_mask] = modulated_crack_color

    # Now blur the areas where the cracks are applied
    for i in range(3):  # Apply the Gaussian blur to each channel separately
        crack_applied[:, :, i] = gaussian_filter(crack_applied[:, :, i], sigma)

    # Mix the blurred crack area back into the original texture
    texture[crack_mask] = crack_applied[crack_mask]
    

    return texture

##################################################################################################################

## Add salt and pepper nosie to blended image
def add_salt_pepper_noise(image, amount=0.006, s_vs_p=0.4, mean_rad=1, std_rad=0.5):
    rng = np.random.default_rng()
    rgb_image = np.copy(image)

    # Aspect ratio of the image for elliptical noise
    aspect_ratio = image.shape[1] / image.shape[0]

    def get_axis(is_width):
        rad = max(1, int(round(rng.normal(mean_rad, std_rad))))
        return int(rad * aspect_ratio) if is_width else rad

    # Colors for salt and pepper noise
    salt_color = (173, 255, 47)  # Bright yellow as a placeholder for salt
    #pepper_color = (10, 10, 10)  # Dark grey as a placeholder for pepper
    pepper_color = (0, 51, 102) #dark blue peppers

    # Adding Salt noise
    num_salt = np.ceil(amount * image.shape[0] * image.shape[1] * s_vs_p).astype(int)
    salt_coords_x = rng.integers(0, rgb_image.shape[1], num_salt)
    salt_coords_y = rng.integers(0, rgb_image.shape[0], num_salt)
    for x, y in zip(salt_coords_x, salt_coords_y):
        rad_x, rad_y = get_axis(True), get_axis(False)
        cv2.ellipse(rgb_image, (x, y), (rad_x, rad_y), 
                    angle=rng.uniform(0, 360), startAngle=0, endAngle=360, 
                    color=salt_color, thickness=-1)

    # Adding Pepper noise
    num_pepper = np.ceil(amount * image.shape[0] * image.shape[1] * (1 - s_vs_p)).astype(int)
    pepper_coords_x = rng.integers(0, rgb_image.shape[1], num_pepper)
    pepper_coords_y = rng.integers(0, rgb_image.shape[0], num_pepper)
    for x, y in zip(pepper_coords_x, pepper_coords_y):
        rad_x, rad_y = get_axis(True), get_axis(False)
        cv2.ellipse(rgb_image, (x, y), (rad_x, rad_y), 
                    angle=rng.uniform(0, 360), startAngle=0, endAngle=360, 
                    color=pepper_color, thickness=-1)

    return rgb_image

###### main ########

def main(args):
    # Create necessary directories
    os.makedirs(os.path.join(args.output_dir, 'textures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'cracks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'blended_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'noisy_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels'), exist_ok=True)
    print("Creating synthetic dataset...")
    # Resume generation
    start_index = 0
    if args.resume:
        generated_files = os.listdir(os.path.join(args.output_dir, 'noisy_images'))
        print(f"Number of samples already generated: {len(generated_files)}")
        if generated_files:
            # Filter to ensure only numeric filenames are considered
            numeric_files = [f for f in generated_files if f.split('.')[0].isdigit() and f.endswith('.jpg')]
            if numeric_files:
                # Get the highest number from the filenames
                highest_number = max(int(f.split('.')[0]) for f in numeric_files)
                start_index = highest_number  # This will start the generation from the next sample
                print(f"Resuming generation from sample number {start_index + 1}")
            else:
                print("No valid files found in 'noisy_images' directory. Starting from scratch.")
        else:
            print("No existing files found in 'noisy_images' directory. Starting from scratch.")

    # Define probabilities for each color map set based on desired ratios
    ratios = [0.2, 0.2, 0.2, 0.1, 0.1, 0.2]  
    all_color_maps = [color_maps_set2, color_maps_set3, color_maps_set4, color_maps_set5, color_maps_set6, color_maps_set7]
    
    # Start generating samples
    for i in tqdm(range(start_index, args.samples), desc="Generating samples..."):
        chosen_set_index = np.random.choice(len(all_color_maps), p=ratios)
        chosen_color_map = all_color_maps[chosen_set_index]
    
        # Generate and save texture
        texture = create_texture(args.image_width, args.image_height, args.patch_size, args.patch_overlap, chosen_color_map)
        texture_file = os.path.join(args.output_dir, 'textures', f'{i + 1}.jpg')
        plt.imsave(texture_file, texture, dpi=600)

        # Generate and save crack image
        image = np.ones((args.image_height, args.image_width, 3), dtype=np.uint8) * 255
        crack_color = np.array([173, 255, 47])  # bright green
        min_rect_size = 50
        max_rect_extra = 200
        rect_start_x = np.random.randint(20, (args.image_width - min_rect_size)//2)
        rect_start_y = np.random.randint(20, (args.image_height - min_rect_size)//2)
        rect_end_x = np.random.randint(rect_start_x + min_rect_size, min(args.image_width, rect_start_x + max_rect_extra))
        rect_end_y = np.random.randint(rect_start_y + min_rect_size, min(args.image_height, rect_start_y + max_rect_extra))
        rect_region = (rect_start_x, rect_start_y, rect_end_x, rect_end_y)
        max_width = np.random.randint(1,3)
        crack_image, annotations = draw_crack(image, generate_crack_path, crack_color,rect_region, max_width)
        crack_file = os.path.join(args.output_dir, 'cracks', f'{i + 1}.jpg')
        plt.imsave(crack_file, crack_image, dpi=600)

        # Save annotations
        annotations_file = os.path.join(args.output_dir, 'labels', f'{i + 1}.txt')
        if annotations:  # This checks if annotations tuple is not empty
            formatted_annotation = " ".join(f"{x:.8f}" if isinstance(x, float) else str(x) for x in annotations)
            with open(annotations_file, 'w') as file:
                file.write(formatted_annotation + '\n')
        else:
            # Create an empty file if there are no annotations
            with open(annotations_file, 'w') as file:
                pass  # Simply pass, resulting in an empty file

        # Blend images and save
        # Blend images
        #print("Blending images...")
        if args.blending_method == "direct":
          blended_image = blend_images_direct(texture_file, crack_image, crack_color)
        else:
          blended_image = blend_images_processed(texture_file, crack_image, crack_color,args.sigma)
        blended_file = os.path.join(args.output_dir, 'blended_images', f'{i + 1}.jpg')
        plt.imsave(blended_file, blended_image, dpi=600)

        # Add noise
        #if args.noise == True:
        salt_image = add_salt_pepper_noise(blended_image)
        salt_file = os.path.join(args.output_dir, 'noisy_images', f'{i + 1}.jpg')
        plt.imsave(salt_file, salt_image, dpi=600)
        

    print("All samples have been generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic textures and cracks")
    parser.add_argument("-sm", "--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory to save generated images")
    parser.add_argument("-i_h","--image_height", type=int, default=640, help="Height of the generated images")
    parser.add_argument("-i_w", "--image_width", type=int, default=1280, help="Width of the generated images")
    parser.add_argument("-p_s","--patch_size", type=int, default=160, help="Patch size to generate final image")
    parser.add_argument("-p_o", "--patch_overlap", type=int, default=100, help="Overlap between the patches")
    parser.add_argument("-blm", "--blending_method", type=str, default="direct", help="Blending method \
                                                              Available: \
                                                              direct: directly add the crack\
                                                              processed: add crack with some processing")
    parser.add_argument("-sg", "--sigma", type=float, default=1, help="Sigma for gaussian blur in processed blending")
    parser.add_argument("-r", "--resume", type=bool, default=False, help="If True then resumes data gen from last sample")
                                                  

    args = parser.parse_args()
    main(args)
