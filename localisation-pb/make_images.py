from .utils_images import generate_rectangles_image, create_image_with_paris_map, remake_dataset
import matplotlib.pyplot as plt

height, width = 100, 100
filter = generate_rectangles_image(height, width)
filter = create_image_with_paris_map(filter)
# Display image
plt.imshow(filter)
plt.show()

# remake_dataset()