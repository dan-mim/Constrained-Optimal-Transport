

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import matplotlib.pyplot as plt

def generate_image(height, width):
    # Create a blank image of specified size with all pixels initially black
    image = np.zeros((height, width), dtype=np.uint8)

    # Define the bounds of the lower-left quarter to avoid placing spots there
    lower_left_quarter = (width // 2, width, height // 2, height)  # x_min, x_max, y_min, y_max

    # Number of spots to generate
    num_spots = random.randint(50, 150)

    for _ in range(num_spots):
        # Choose a random position for the spot, excluding the lower-left quarter
        x_center = random.randint(0, width - 1)
        y_center = random.randint(0, height - 1)

        while lower_left_quarter[0] <= x_center < lower_left_quarter[1] and lower_left_quarter[2] <= y_center < lower_left_quarter[3]:
            x_center = random.randint(0, width - 1)
            y_center = random.randint(0, height - 1)

        # Randomly choose the radius and shape
        radius = random.randint(height // 100, height // 20)
        shape = random.choice(['circle', 'square'])

        if shape == 'circle':
            # Draw a circular spot
            for x in range(max(0, x_center - radius), min(height, x_center + radius + 1)):
                for y in range(max(0, y_center - radius), min(width, y_center + radius + 1)):
                    if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                        image[x, y] = 255  # Set pixel to white
        elif shape == 'square':
            # Draw a square spot
            for x in range(max(0, x_center - radius), min(height, x_center + radius + 1)):
                for y in range(max(0, y_center - radius), min(width, y_center + radius + 1)):
                    image[x, y] = 255  # Set pixel to white

    # Return the numpy array representing the spots
    return image

def create_image_with_paris_map(spot_array, alpha_value=255):
    height, width = spot_array.shape
    # Load the Paris map and resize it
    paris_map = Image.open('paris_map.png').convert('RGB').resize((width, height))

    # Convert the Paris map to RGBA format for transparency
    paris_rgba = paris_map.convert('RGBA')

    # Apply transparency and red tint to spots in the overlay array
    for x in range(width):
        for y in range(height):
            gray_value = spot_array[y, x]  # Access the numpy array directly
            if gray_value ==  0:  # If there's a spot
                # Adjust red tint by setting red component higher
                r, g, b, _ = paris_rgba.getpixel((x, y))
                new_r = min(255, 200+r)  # Increase the red channel for a reddish tint
                paris_rgba.putpixel((x, y), (new_r, g, b, alpha_value))
            else:
                # Make non-spot areas fully opaque
                paris_rgba.putpixel((x, y), (*paris_rgba.getpixel((x, y))[:3], 255))

    return paris_rgba

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

def create_image_nuanced_with_paris_map_regular_resolution(spot_array, vmax=0.005, alpha=0.5):
    height, width = spot_array.shape

    # Load and resize the Paris map to match the dimensions of spot_array
    paris_map = Image.open('paris_map.png').convert('RGB').resize((width, height))
    paris_map = np.array(paris_map)

    # Normalize spot_array for colormap application
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.hot_r
    spot_colored = cmap(norm(spot_array))  # Returns an RGBA array (NxMx4)

    # Create a mask where spot_array > 0
    mask = spot_array > 0

    # Alpha blending: overlay color-mapped spots on the Paris map
    overlay = paris_map.copy()
    overlay[mask] = (
        alpha * (spot_colored[mask, :3] * 255) + (1 - alpha) * paris_map[mask]
    ).astype(np.uint8)

    # Create a figure for rendering
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.axis('off')

    # Display the blended image
    ax.imshow(overlay)

    # Add a colorbar
    cbar = ColorbarBase(ax=fig.add_axes([0.9, 0.15, 0.02, 0.7]), cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Intensity', rotation=270, labelpad=15)

    # Render the figure to a NumPy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the plot to release memory

    return image_array

def create_image_nuanced_with_paris_map_increased_resolution(spot_array, vmax=0.005, alpha=0.6):
    # Load the Paris map at its original resolution
    paris_map = Image.open('paris_map.png').convert('RGB')
    paris_map = np.array(paris_map)
    height_map, width_map, _ = paris_map.shape

    # Resize the spot array to match the dimensions of the Paris map
    I = spot_array < 10**-4
    spot_array[I] = 0
    spot_array_resized = np.array(Image.fromarray(spot_array).resize((width_map, height_map), Image.BICUBIC))

    # Normalize the resized spot array for colormap application
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.hot_r
    spot_colored = cmap(norm(spot_array_resized))  # Returns an RGBA array (HxWx4)

    # Create a mask where spot_array > 0 (after resizing)
    mask = spot_array_resized > 0

    # Alpha blending: overlay color-mapped spots on the Paris map
    overlay = paris_map.copy()
    overlay[mask] = (
        alpha * (spot_colored[mask, :3] * 255) + (1 - alpha) * paris_map[mask]
    ).astype(np.uint8)

    # Create a figure for rendering
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.axis('off')

    # Display the blended image
    ax.imshow(overlay)

    # Add a colorbar
    cbar = ColorbarBase(ax=fig.add_axes([0.9, 0.15, 0.02, 0.7]), cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Intensity', rotation=270, labelpad=15)

    # Render the figure to a NumPy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the plot to release memory

    return image_array

def display_dataset(dataset):
    # Define grid dimensions
    grid_rows, grid_cols = 3, 3
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
    # Adjust spacing between the images
    plt.subplots_adjust(wspace=0.0, hspace=0.05)  # Adjust these values as needed
    # Plot each image on the grid
    for i in range(grid_rows * grid_cols):
        # Check if we have an image for this grid cell
        if i < len(dataset):
            # Generate the overlaid image with spots on the Paris map
            im = dataset[i % len(dataset)]
            I, J = im>0, im==0
            im[I] = 0
            im[J] = 1
            result_image = create_image_with_paris_map(im) # create_image_with_paris_map(dataset[i % len(dataset)])

            # Display the image in the appropriate grid cell
            ax = axes[i // grid_cols, i % grid_cols]
            ax.imshow(result_image)
            ax.axis('off')
        else:
            # Hide unused grid cells
            axes[i // grid_cols, i % grid_cols].axis('off')

    plt.show()

from PIL import ImageDraw
def generate_rectangles_image(height, width):
    # Create a white background image
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Convert to PIL image for easy drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw = ImageDraw.Draw(pil_image)
    rectangle_size1 = width//10
    rectangle_size2 = [height//9, height//5]
    liste_x0s = [3*width//4, width//4, 7*width//10, 6*width//11, width-rectangle_size1-10, rectangle_size1, 2*width//9, width-rectangle_size1-5]
    list_y0s = [height//9, height//2, 7*height//12, 3*height//4, 8*height//13, 3*height//4, height-rectangle_size2[0]-5, height-rectangle_size2[0]-5]
    num_squares = len(list_y0s)

    for i in range(num_squares):
        # Randomly choose top-left corner for each square, ensuring it fits within the image
        j = 0
        if i == 0:
            j = 1
        rectangle_size = rectangle_size1, rectangle_size2[j]
        x0 = liste_x0s[i]
        y0 = list_y0s[i]
        x1, y1 = x0 + rectangle_size[0], y0 + rectangle_size[1]

        x1, y1 = x0 + rectangle_size[0], y0 + rectangle_size[1]

        # Draw a filled black square
        draw.rectangle([x0, y0, x1, y1], fill=0)


    # Convert back to a numpy array
    final_image = np.array(pil_image)

    return final_image

if __name__ == '__main__':
    height, width = 100,100
    filter = generate_rectangles_image(height, width)
    filter = create_image_with_paris_map(filter)
    # Display image
    plt.imshow(filter)


    remake_dataset = True
    if remake_dataset:
        dataset = []
        for i in range(10):
            spot_array = generate_image(height, width)
            dataset.append(spot_array/np.sum(spot_array))
        with open("dataset/location_demand", 'wb') as f:
            pickle.dump(dataset, f)

        # Load the dataset of 10 spot arrays
        with open("dataset/location_demand", 'rb') as f:
            dataset = pickle.load(f)
        display_dataset(dataset)
    # plt.show()


