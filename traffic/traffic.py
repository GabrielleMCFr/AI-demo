import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
# normal batch of data
#NUM_CATEGORIES = 43
# variant, small batch of data
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():

    if len(sys.argv) not in [2, 3]:
        sys.exit("Wrong arguments count")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = list()
    labels = list()
    for dirpath, dirs, files in os.walk(data_dir): 
        for filename in files:
            try:
                filepath = os.path.join(dirpath,filename)
                
                image = cv2.imread(filepath)
                
                resized_img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                images.append(resized_img)
                # for normal batch of data
                #label = int(dirpath[6:])
                # for small batch of data
                label = int(dirpath[12:])
                labels.append(label)
            except:
                print(f"Something went wrong with {filename}")
            
               

    return (images, labels)


def get_model():
    """
    Returns a compiled CNN model.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
         # Flatten units
        tf.keras.layers.Flatten(),
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(2000, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        # Add another hidden layer
        tf.keras.layers.Dense(1500, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        # Add an output layer with output units for all categs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # compile model
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()