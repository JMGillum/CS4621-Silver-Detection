import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from sklearn.model_selection import train_test_split

import cv2

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pathlib import Path

import warnings


def getImagePaths(root,folder_names,extensions):
    values = []
    if not folder_names:
        folder_names = ("",)
    for folder_name, extension in zip(folder_names,extensions):
        values.append(glob.glob(str(Path(root) / Path(folder_name) / Path(extension))))
    return tuple(values)


def load_images(image_paths,img_size):
        images = []
        for path in image_paths:
            print(path)
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(img_size,img_size))
            images.append(img)
        return np.array(images,dtype='float32')/255.0


def Start(root_image_path=None,model_path=None,verbose=False):

    # Define which CNNs to make and which classes exist within them
    image_root = Path.cwd() / Path('images') / Path("unknown")

    if root_image_path:
        image_root = Path(root_image_path)

    if model_path:
        model_path = Path(model_path)
    else:
        model_path = Path.cwd() / Path('binary-classification-silver.keras')
    

    model = keras.models.load_model(model_path)


    img_size = 100
    split = 0.2
    epochs = 10
    batch_size = 64
    image_extension = "*.jpg"

    image_paths = getImagePaths(image_root,None,(image_extension,))[0]
    prediction_images = load_images(image_paths,img_size)
    predictions = model.predict(x=prediction_images, steps=len(prediction_images))
    print(np.round(predictions))

        
