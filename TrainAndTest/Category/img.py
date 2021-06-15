import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image, ImageOps
import tensorflow as tf
import pathlib
import skimage.transform as st

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def save_image(npdata,outfilename) :
    img = Image.fromarray(npdata)
    img.save(outfilename)


def load_image(infilename):
  img = Image.open(infilename)
  data = np.asarray(img)
  data = st.resize(data, (40, 30))
  return data


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


data_dir = 'images2'

batch_size = 128
img_height = 40
img_width = 30

#creo dataset training da cartella
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(train_ds)

#creo dataset testing da cartella
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
print(len(class_names))

#cache() tiene le immafini in memoria dopo la prima epoch

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
# layer di normalizzazione, porta tutti i dati delle immagini in intervallo 0-1
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


num_classes = 10
#layer per aumentare il numero di dati attraverso trasformazioni delle immagini
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# metrics=['accuracy'] serve solo per visulizzare l'addestramento
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


model.save('models/category')


#new_model = tf.keras.models.load_model('model_img_color')



sunflower_path = 'cappo.jpg'


img_array = load_image(sunflower_path)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

for i, val in enumerate(score):
    print(class_names[i] + ':' + str(val * 100))





