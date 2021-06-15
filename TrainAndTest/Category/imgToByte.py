import tensorflow as tf
from PIL import Image, ImageOps

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as st
import os
import os

def load_image(infilename):
    img = Image.open(infilename)
    data = np.asarray(img)
    print(data.shape)
    data = st.resize(data, (40, 30))
    if (data.shape[0] != 40):
        print("NO")
        exit(8)
    if (data.shape[1] != 30):
        print("NO3")
        exit(8)
    return data



labelsA = {}
labels = []

images = []
i = 0
for root, dirs, files in os.walk("imagesGrey"):
    for dir in dirs:
        for root, dirs, files in os.walk("imagesGrey/" + dir):
            for file in files:
                if (dir not in labelsA):
                    labelsA[dir] = i
                    print(dir)
                    i = i + 1

                #print("imagesGrey/" + dir + "/" + file)
                images.append("imagesGrey/" + dir + "/" + file)
                labels.append(labelsA[dir])


print(labelsA)
labels = np.asarray(labels, dtype=np.uint8)
np.save("npdataset/labels.npy", labels)
numpyarr = []
for i in images:

    numpyarr.append(load_image(i))

print(4444)
np.save("npdataset/images.npy", numpyarr)




