import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from sklearn.model_selection import train_test_split

import cv2
import os

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import layers
from pathlib import Path

import warnings


def getImagePaths(root,folder_names,extensions):
    values = []
    for folder_name, extension in zip(folder_names,extensions):
        values.append(glob.glob(str(Path(root) / Path(folder_name) / Path(extension))))
    return tuple(values)


def load_images(image_paths,img_size):
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(img_size,img_size))
            images.append(img)
        return np.array(images,dtype='float32')/255.0


def Start(root_image_path=None,save_output=True,output_file=None,verbose=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    warnings.filterwarnings('ignore')


    # Define which CNNs to make and which classes exist within them
    image_root = Path.cwd() / Path('images') / Path("dataset")

    if root_image_path:
        image_root = Path(root_image_path)

    cnns = (
            {
             "name": "binary-classification-silver",
             "root": image_root,
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

    
    categories_with_more_than_zero_images = 0
    # Prints out information about the images found
    print("\nImages found:\n")
    for cnn in cnns:
        print(f'----{cnn["name"]}----')
        for c in range(len(cnn["classes"])):
            cat = cnn["classes"][c]
            print(f"{cat['name']}: {len(cat['images'])} images")
            if len(cat['images']) > 0:
                categories_with_more_than_zero_images += 1

    if categories_with_more_than_zero_images <= 0:
        print("No images found.",flush=True)
        exit(1)

    img_size = 100
    split = 0.2
    epochs = 10
    batch_size = 64


    # Displays several images of every category found to ensure they were categorized correctly
    if verbose:
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
        
        
        if verbose:
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
        
        if verbose:
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

    if save_output:
        if not output_file:
            output_file = Path.cwd() / Path(f"{cnn['name']}.keras")
        cnn["model"].save(output_file)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)



def Start2(root_image_path=None,save_output=True,output_file=None,verbose=False):
    image_size = (256,256)
    batch_size = 32


    # Define which CNNs to make and which classes exist within them
    image_root = Path.cwd() / Path('images') / Path("dataset")

    if root_image_path:
        image_root = Path(root_image_path)

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        image_root,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    if verbose:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(np.array(images[i]).astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
            plt.show()

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)


    epochs = 2

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    if save_output:
        if not output_file:
            output_file = Path.cwd() / Path("model.keras")
        model.save(output_file)

    img = keras.utils.load_img(image_root / Path("silver_images") / Path("6779.jpg"), target_size=image_size)
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% not silver and {100 * score:.2f}% silver.")
