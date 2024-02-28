import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('/home/hadi/Desktop/GF/DataSet/MyDataset/Dmap.npz')

# Access the depth map array
image_data = data['arr_0']  # Replace 'arr_0' with the key name if it's different
print(image_data.shape)
print (max(image_data[10]))
# image_data = image_data.reshape(1080, 1920, 4)
# Visualize the depth map

plt.imshow(image_data)  # You can change the colormap as needed
plt.colorbar(label='Depth')  # Add a colorbar to indicate depth values
plt.title('Depth Map Visualization')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()  


# from PIL import Image

# def create_stripe_image(width, height, stripe_width, color1, color2):

#     image = Image.new("RGB", (width, height))
#     pixels = image.load()

#     for y in range(height):
#         for x in range(width):
#             if x % (stripe_width * 2) < stripe_width:
#                 pixels[x, y] = color1
#             else:
#                 pixels[x, y] = color2

#     return image

# if __name__ == "__main__":
#     # Define image dimensions and stripe properties
#     width = 800
#     height = 600
#     stripe_width = 10
#     color1 = (255, 255, 255)  # White
#     color2 = (0, 0, 0)         # Black

#     # Create stripe image
#     stripe_image = create_stripe_image(width, height, stripe_width, color1, color2)

#     # Save image as "pattern.jpg"
#     stripe_image.save("/home/hadi/Desktop/pattern.jpg")
