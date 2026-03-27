import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from sklearn.model_selection import train_test_split

import cv2
import os

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')


# Define which CNNs to make and which classes exist within them
image_root = Path.cwd() / Path('./images/')

cnns = (
        {
         "name": "binary-classification-silver",
         "root": image_root / Path("dataset"),
         "classes": [
             {
                 'name':'silver',
                 'folder_name':'silver_images',
                 'image_extension':'*.jpg',
                 },
             {
                 'name':'non_silver',
                 'folder_name':'non_silver_images',
                 'image_extension':'*.jpg'
                 }
             ]
         },
        )

def getImagePaths(root,folder_names,extensions):
    values = []
    for folder_name, extension in zip(folder_names,extensions):
        values.append(glob.glob(os.path.join(root,folder_name,extension)))
    return tuple(values)


# Finds all images within the respective folders for each classes
# All images must have the same image extension
for i in range(len(cnns)):
    current_cnn = cnns[i]
    for c in range(len(current_cnn["classes"])):
        current_class = current_cnn["classes"][c]
        images = getImagePaths(current_cnn["root"],(current_class["folder_name"],),(current_class["image_extension"],))[0]
        try:
            cnns[i]["classes"][c]["images"] |= images
        except KeyError:
            cnns[i]["classes"][c]["images"] = images


# Prints out information about the images found
print("\nImages found:\n")
for cnn in cnns:
    print(f'----{cnn["name"]}----')
    for c in range(len(cnn["classes"])):
        cat = cnn["classes"][c]
        print(f"{cat['name']}: {len(cat['images'])} images")

img_size = 100
split = 0.2
epochs = 10
batch_size = 64


# Displays several images of every category found to ensure they were categorized correctly
for cnn in cnns:
    for c in range(len(cnn["classes"])):
        name = cnn["classes"][c]["name"]
        images = cnn["classes"][c]["images"]
    
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Images for {name} category . . . .', fontsize=20)
    
        for i in range(3):
            k = np.random.randint(0, len(images))
            img = np.array(Image.open(images[k]))
            ax[i].imshow(img)
            ax[i].axis('off')
        plt.show()


def load_images(image_paths,img_size):
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(img_size,img_size))
            images.append(img)
        return np.array(images,dtype='float32')/255.0

for cnn in cnns:
    print(f"-----{cnn['name']}-----")
    image_paths = []
    labels = []
    count = 0
    for x in range(len(cnn["classes"])):
        image_paths += cnn["classes"][x]["images"]
        print(x)
        labels += [count] * len(cnn["classes"][x]["images"])
        count = count + 1

    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size = split*2,
        random_state = 42,
        stratify = labels
    )
    
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(
        X_temp_paths,
        y_temp,
        test_size = 0.5,
        random_state = 42,
        stratify = y_temp
    )
    
    
    print("\nData split:")
    print(f"Training samples: {len(X_train_paths)} ({len(X_train_paths)/len(image_paths)*100:.1f}%)")
    print(f"Validation samples: {len(X_val_paths)} ({len(X_val_paths)/len(image_paths)*100:.1f}%)")
    print(f"Test samples: {len(X_test_paths)} ({len(X_test_paths)/len(image_paths)*100:.1f}%)")
    
    print(f"\nTraining set distribution: {np.bincount(y_train)}")
    print(f"Validation set distribution: {np.bincount(y_val)}")
    print(f"Test set distribution: {np.bincount(y_test)}")
    
    cnn["X_train"] = load_images(X_train_paths,img_size)
    cnn["X_val"] = load_images(X_val_paths,img_size)
    cnn["X_test"] = load_images(X_test_paths,img_size)
    
    cnn["y_train"] = np.array(y_train)
    cnn["y_val"] = np.array(y_val)
    cnn["y_test"] = np.array(y_test)
    
    print(f"X_train shape: {cnn['X_train'].shape}")
    print(f"X_val shape: {cnn['X_val'].shape}")
    print(f"X_test shape: {cnn['X_test'].shape}")


for cnn in cnns:
    cnn["model"] = keras.models.Sequential()
    
    #First Layer
    cnn["model"].add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
    cnn["model"].add(MaxPooling2D((2,2)))
    #Second Layer
    cnn["model"].add(Conv2D(64,(3,3),activation='relu'))
    cnn["model"].add(MaxPooling2D((2,2)))
    #Third Layer
    cnn["model"].add(Conv2D(128,(3,3),activation='relu'))
    cnn["model"].add(MaxPooling2D((2,2)))
    #Convert and Output
    cnn["model"].add(Flatten())
    cnn["model"].add(Dense(128,activation='relu'))
    cnn["model"].add(Dense(3,activation='softmax'))
    
    cnn["model"].compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    cnn["model"].summary()

for cnn in cnns:
    print(f"-----{cnn['name']}-----")
    cnn["history"] = cnn["model"].fit(
        cnn["X_train"],
        cnn["y_train"],
        batch_size=16,
        epochs=1,
        validation_data=(cnn["X_val"],cnn["y_val"])
    )
    
    print('Evaluate on Test Data')
    test_loss,test_accuracy = cnn["model"].evaluate(cnn["X_test"],cnn["y_test"],batch_size=128)
    print('Test Loss: ',test_loss)
    print('Test Accuracy: ',test_accuracy)

# Currently doesn't work
"""
for cnn in cnns:
    print(f"-----{cnn['name']}-----")
    # summarize history for accuracy
    plt.plot(cnn["history"].history['accuracy'])
    plt.plot(cnn["history"].history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(cnn["history"].history['loss'])
    plt.plot(cnn["history"].history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
"""
