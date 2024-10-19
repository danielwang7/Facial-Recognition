# Import Standard Dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import uuid library to generate unique image names
import uuid

# Import tensorflow dependencies -- Functional API
# https://www.tensorflow.org/guide/keras/functional_api
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


def start_camera():

    # Establish a connection to the webcam
    cap = cv2.VideoCapture(1)   # You will probably have to adjust this value to a number between 0 and 4, whatever brings up your webcam
    while cap.isOpened():
        ret, frame = cap.read()     # reads capture

        # cut frame down to 250x250px
        # TODO: Adjust frame dimensions to capture your face in the webcam; this may be tedious
        frame = frame[350:350+250,700:700+250, :]

        # Collect anchors
        if cv2.waitKey(1) & 0xFF == ord('a'):
            # Create unique file path
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives
        if cv2.waitKey(1) & 0xFF == ord('p'):
            # Create unique file path
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)    # displays capture

        # Breaking gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


#Scale and resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    # Resize the image to be 100x100
    img = tf.image.resize(img, [100, 100])
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


if __name__ == "__main__":

    # # Make directories
    # os.makedirs(POS_PATH)
    # os.makedirs(NEG_PATH)
    # os.makedirs(ANC_PATH)

    # # Move lfw images to the following repository data/negative
    # # loop through the directories in lfw
    # for directory in os.listdir('lfw'):
    #     for file in os.listdir(os.path.join('lfw', directory)):
    #         EX_PATH = os.path.join('lfw', directory, file) # existing path
    #         NEW_PATH = os.path.join(NEG_PATH, file)
    #         os.replace(EX_PATH, NEW_PATH)

    # start_camera()

    #Get image directories
    anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)

    
    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(3000, dtype=tf.float32)))
    )

    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(3000, dtype=tf.float32)))
    )

    full_data = positives.concatenate(negatives)

    # samples = full_data.as_numpy_iterator() #Turn the dataset into an iterable object so we can grab an example
    # example = samples.next() #Grab a piece of data
    # print(example) #Print the example grabbed

    full_data = full_data.map(preprocess_twin)
    full_data = full_data.cache()
    full_data = full_data.shuffle(buffer_size=10000)

    # Train split
    train_data = full_data.take(round(len(full_data)*.7)) #70% training
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # Test split
    test_data = full_data.skip(round(len(full_data)*.7))
    test_data = train_data.batch(16)
    test_data = train_data.prefetch(8)


    # ---------MODEL TRAINING BEGINS----------
    
    # data should be in 100, 100, and 3, as preprocessing has done
    input_layer = Input(shape=(100, 100, 3), name="input_image")

    # Define the Conv2D layer with 64 filters, kernel size of 10x10, and ReLU activation
    conv_layer = Conv2D(64, (10, 10), activation='relu')(input_layer)

    model = Model(inputs=input_layer, outputs=conv_layer)

    # Print the summary of the conv layer within the model
    model.summary()
   

