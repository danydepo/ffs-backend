from PIL import Image, ImageOps

from pprint import pprint
import numpy as np
import skimage.transform as st
import os
from skimage import io
from skimage import img_as_ubyte






def load_image(infilename) :
    img = Image.open(infilename)
    gray_image = ImageOps.grayscale(img)
    gray_image = ImageOps.invert(gray_image)
    data = np.asarray(gray_image)

    data = st.resize(data, (400, 300))
    print(data)
    return data


labels = []

images = []
imagesG = []
for root, dirs, files in os.walk("images2"):
    for dir in dirs:
        for root, dirs, files in os.walk("images2/" + dir):
            for file in files:
                if not os.path.exists('imagesGrey/' + dir):
                    os.makedirs('imagesGrey/' + dir)
                images.append("images2/" + dir + "/" + file)
                imagesG.append("imagesGrey/" + dir + "/" + file)


numpyarr = []
for i, val in enumerate(images):
    value = imagesG[i].replace('.jpg', '.tiff')
    io.imsave(value, img_as_ubyte(load_image(images[i])))







