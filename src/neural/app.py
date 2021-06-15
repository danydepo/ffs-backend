import sys
sys.path.append("/mnt/pylib")

import numpy as np
import boto3
import os
from collections import Counter, defaultdict
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from colormap import rgb2hex
import cv2
import skimage.transform as st
import os
import tensorflow as tf


s3 = boto3.resource('s3')

tableName = os.environ['DynamoDbName']
bucketName = os.environ['BucketName']

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(tableName)
s3.Bucket(bucketName).download_file('category.h5', '/tmp/category.h5')


def delete(item):
    table.delete_item(Key=item)




def insert(item):
    table.put_item(Item=item)

def load_image(infilename):
    img = Image.open(infilename)
    gray_image = ImageOps.grayscale(img)
    gray_image = ImageOps.invert(gray_image)
    data = np.asarray(gray_image)

    data = st.resize(data, (40, 30))
    return data

def get_category(img):
    model = tf.keras.models.load_model('/tmp/category.h5')
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    img = (np.expand_dims(load_image(img), 0))

    classNames = ['CAPPOTTO', 'SHORTS', 'GIACCA', 'VESTITO', 'T-SHIRT', 'MAGLIONE', 'BLAZER', 'GONNA',
                  'PANTALONI', 'CAMICIA', 'FELPA']

    predictions = probability_model.predict(img)

    return classNames[np.argmax(predictions[0])]

def calculate_color(img):
    newImg = rmv_back_pro(img)
    clusters = 3
    dc = DominantColors(newImg, clusters)
    colors = dc.dominantColors()
    percentage = dc.get_percentage()
    print(percentage)
    col = ""
    for i in range(len(colors)):
        col = col + str(rgb2hex(colors[i][0], colors[i][1], colors[i][2])) + "(" + str(percentage[i]) + "),"
    col = col[:-1]

    print(col)
    return col


class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image

        im = Image.open(self.IMAGE, 'r')
        pixel_values = list(im.getdata())
        pixels = []
        for pv in pixel_values:
            if (pv[3] > 0):
                pixels.append(pv[:-1])

        if len(pixels) == 0:
            pixels.append([0, 0, 0])

        img = self.IMAGE
        # save image after operations
        self.IMAGE = pixels

        # using k-means to cluster pixels
        diff = 0

        done = False

        if len(pixels) < self.CLUSTERS:
            self.IMAGE = []
            for p in pixels:
                for r in range(self.CLUSTERS * 10):
                    self.IMAGE.append(p)

        while not done:
            try:
                kmeans = KMeans(n_clusters=self.CLUSTERS - diff)
                kmeans.fit(self.IMAGE)
                done = True
            except ValueError:
                print("------------------------ERROR---------------------------------------" + str(img))
                diff = diff + 1
                if diff > self.CLUSTERS:
                    break

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def get_percentage(self):
        total = 0
        counter = {}
        c = Counter(self.LABELS)
        for key in sorted(c):
            counter[key] = c[key]

        for k, v in counter.items():
            total = total + v
        percentage = {}
        for k, v in counter.items():
            percentage[k] = v / total * 100

        return percentage


def rmv_back_pro(path):
    # Load the Image
    imgo = cv2.imread(path)
    height, width = imgo.shape[:2]

    # Create a mask holder
    mask = np.zeros(imgo.shape[:2], np.uint8)

    # Grab Cut the object
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Hard Coding the Rect The object must lie within this rect.
    rect = (10, 10, width - 30, height - 30)
    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img1 = imgo * mask[:, :, np.newaxis]

    # Get the background
    background = imgo - img1

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img1

    newPath = path.replace('.jpg', '.png')
    thresholding(final, newPath, 254)

    # To be done - Smoothening the edges
    return newPath


def thresholding(img, newPath, th=None):
    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shape = img.shape

    if th is None:
        blur = cv2.blur(img, (shape[0], shape[1]))
        th = max(cv2.mean(blur))
        if th >= 200 and th < 240:
            th = 240

        if th > 160 and th < 200:
            th = 220

    # threshold input image as mask
    mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]

    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    print(newPath)

    cv2.imwrite(newPath, result)

    return result


def lambda_handler(event, context):
    for record in event['Records']:

        file_key = record['s3']['object']['key']
        user, image = os.path.split(file_key)
        user = user.split('/')[-1]



        item = {
            'userId': user,
            'imgId': image
        }

        trigger = record['eventName']
        print(record)

        if trigger == 'ObjectRemoved:Delete':
            print("Delete")
            delete(item)

        if trigger == 'ObjectCreated:Put':
            print("Put")
            local_img = '/tmp/' + image
            s3.Bucket(bucketName).download_file(file_key, local_img)
            item['category'] = get_category(local_img)
            item['colors'] = calculate_color(local_img)
            insert(item)

    return True



