import matplotlib
matplotlib.use("TkAgg")
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

# ** YOUR CODE HERE **
#nine images from first folder
folder = "att_faces/s25"
images = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".pgm")
])
images = images[:9]

fig, axes = plt.subplots(3, 3, figsize = (6,6))

for ax , img_name in zip(axes.ravel(),images):
    path = os.path.join(folder, img_name)

    img = load_img(path, color_mode="grayscale")
    img_array = img_to_array(img)

    img_array = img_array.squeeze()

    ax.imshow(img_array, cmap="gray")
    ax.axis("off")

root_dir ="att_faces"
fig, axes = plt.subplots(3, 3, figsize = (6,6))

#one image from first nine 
for i, ax in enumerate(axes.ravel(),start =1):
    folder = os.path.join(root_dir,f"s{i}")

    images = sorted([f for f in os.listdir(folder) if f.endswith(".pgm")])

    img_path = os.path.join(folder, images[0])

    img = load_img(img_path, color_mode="grayscale")
    img_array = img_to_array(img).squeeze()

    ax.imshow(img_array, cmap="gray")
    ax.set_title(f"s{i}")
    ax.axis("off")

plt.tight_layout()
plt.show

plt.show(block=True)

