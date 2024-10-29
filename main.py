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
from tensorflow.keras.metrics import Precision, Recall
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


# BUILDING THE MODEL: Making the Embedding layer
def make_embedding():
    
    # data should be in 100, 100, and 3, as preprocessing has done
    input_layer = Input(shape=(100, 100, 3), name="input_image")

    # Create a Conv2D layer with 64 filters of size 10x10 and ReLU activation, applied to the input inp
    c1 = Conv2D(64, (10, 10), activation='relu')(input_layer)

    #Create a maxpooling layer with 64 filters of size 2x2 and "same" padding, applied to layer c1
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    #Create a Conv2D layer with 128 filters of size 7x7 and ReLU activation
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)

    #All remaining max pooling layers below are identical to one above, just make sure you are applying it to the correct layer
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    #Create a Conv2D layer with 128 filters of size 4x4 and ReLU activation
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    #Create a Conv2D layer with 256 filters of size 4x4 and ReLU activation
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)

    f1 = Flatten()(c4)

    #Create a dense layer with 4096 output neurons, sigmoid activation, applied to the flattened layer in previous line
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[input_layer], outputs=[d1], name='embedding')

# BUILDING THE MODEL: Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    # similarity calculation
    def call(self, inputs):
         # Unpack the inputs - ensure they are tensors, not lists
        input_embedding, validation_embedding = inputs

        # Ensure the inputs are Keras tensors
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)

        # Calculate L1 distance (absolute difference between embeddings)
        return tf.math.abs(input_embedding - validation_embedding)
    

def make_siamese_model():

    embedding = make_embedding()

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'

    # Embed input layers into feature representations
    # Simply pass input image through the embedding model
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)

    distances = siamese_layer([inp_embedding, val_embedding])

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

@tf.function
def train_step(batch, siamese_model, opt, binary_cross_loss):
    # Record all of our operations
    with tf.GradientTape() as tape:

        # Unpack batch inputs correctly
        input_img, validation_img = batch[:2]

        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model([input_img, validation_img], training=True)

        # LOOK FOR BUGS ON THIS LINE!!!!
        yhat = tf.squeeze(yhat)  # This removes any singleton dimensions if needed
        
        # Calculate loss
        # Ensure shape match before loss calculation
        yhat = tf.squeeze(yhat)  # Adjust if needed to match `y`
        loss = binary_cross_loss(y, yhat)
         
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

def train(data, EPOCHS, model, optimizer, loss_fn, checkpoint, checkpoint_prefix):

    try:
        # Loop through epochs
        for epoch in range(1, EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))

            #Create a Recall object
            r = tf.keras.metrics.Recall()
            #Create a Precision object
            p = tf.keras.metrics.Precision()

            # Loop through each batch
            for idx, batch in enumerate(data):
                #Call train step function and pass in the batch
                loss = train_step(batch, model, optimizer, loss_fn)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx+1)
            print(f"Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

            # Save checkpoints
            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current progress...")
        # Save the current state before exiting
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("Checkpoint saved. Training stopped.")


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

    # MAKE THE FINAL MODEL
    siamese_model = make_siamese_model()
    siamese_model.summary()

    #Setup loss and optimizer

    #Use binary cross entropy loss
    binary_cross_loss = tf.losses.BinaryCrossentropy()

    #Use Adam with a learning rate of 0.0001
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    #Create checkpoints for saving model progress throughout training
    checkpoint_dir = os.path.join(os.getcwd(), "training_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)



    # TRAIN THE MODEL WITH THE FUNCTIONS!
    train(data=train_data, 
      EPOCHS=2, 
      model=siamese_model, 
      optimizer=opt, 
      loss_fn=binary_cross_loss, 
      checkpoint=checkpoint, 
      checkpoint_prefix=checkpoint_prefix)
    




   

