import tensorflow as tf
from PIL import Image, ImageOps

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as st
from sklearn.model_selection import KFold
import os

def load_image(infilename):
    img = Image.open(infilename)
    gray_image = ImageOps.grayscale(img)
    gray_image = ImageOps.invert(gray_image)
    data = np.asarray(gray_image)

    data = st.resize(data, (40, 30))
    return data


def save_image(npdata,outfilename):
    img = Image.fromarray(npdata)
    img.save(outfilename)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


inputs = np.load("npdataset/images.npy")
targets = np.load("npdataset/labels.npy")



# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

img_height = 40
img_width = 30


for train, test in kfold.split(inputs, targets):


    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    classNames = ['CAPPOTTO','SHORTS','GIACCA','VESTITO','T-SHIRT','MAGLIONE','BLAZER','GONNA','PANTALONI','CAMICIA','FELPA']

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(img_height, img_width)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(11)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(inputs[train], targets[train], epochs=4)

    model.save('models/category.h5')

    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

    test_loss, test_acc = model.evaluate(inputs[test], targets[test], verbose=2)

    print('\nTest accuracy:', test_acc)
    break

model = tf.keras.models.load_model('models/category.h5')
test = load_image('gonna.jpg')


img = (np.expand_dims(test, 0))

predictions = probability_model.predict(img)

print(predictions)


print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(classNames[np.argmax(predictions[0])], np.max(predictions[0]))
)


