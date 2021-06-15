import os
from pprint import pprint


import numpy as np
import cv2


def rmv_b(path, newPath):
    # Load the Image
    imgo = cv2.imread(path)
    thresholding(imgo, newPath)

def rmv_back_pro(path, newPath):


    print(path)
    #Load the Image
    imgo = cv2.imread(path)
    thresholding(imgo,newPath)
    return
    height, width = imgo.shape[:2]

    #Create a mask holder
    mask = np.zeros(imgo.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #Hard Coding the Rect The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    img1 = imgo*mask[:,:,np.newaxis]

    #Get the background
    background = imgo - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [255, 255, 255]

    #Add the background and the image
    final = background + img1

    thresholding(final, newPath, 250)

    #To be done - Smoothening the edges
    return True

def thresholding(img, newPath, th = None):

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
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    newPath = newPath.replace('.jpg', '.png')
    cv2.imwrite(newPath, result)

    return result


images = []
newPath = []

if not os.path.isdir("test2"):
    os.mkdir("test2")


for root, dirs, files in os.walk("../images2"):
    for dir in dirs:

        for root, dirs, files in os.walk("../images2/" + dir):

            if not os.path.isdir("test2/" + dir):
                os.mkdir("test2/" + dir)

            for file in files:
                images.append("../images2/" + dir + "/" + file)
                newPath.append("test2/" + dir + "/" + file)



from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(8)
results = pool.starmap(rmv_back_pro, zip(images, newPath))

print(results)

