# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pickle
import csv
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from itertools import zip_longest
import tensorflow as tf
import sklearn.metrics as m



print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def data_preparation():
    train = pd.read_csv('digit-recognizer/train.csv')
    global Y_train
    Y_train = train["label"]
    # Drop 'label' column
    global X_train
    X_train = train.drop(labels=["label"], axis=1)
    # Normalize the data
    X_train = X_train / 255.0
    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes=10)

def split_pickle_data():
    print(len(Y_train))
    Y_train_splits= np.split(Y_train,3)
    X_train_splits = np.split(X_train, 3)
    for i in range(3):

        Y_train_split=Y_train_splits[i]
        pickle_out=open("Lables/Y_train_split_"+str(i+1)+".pickle","wb")
        pickle.dump(Y_train_split,pickle_out)
        pickle_out.close()

        X_train_split = X_train_splits[i]
        pickle_out2 = open("Images/X_train_split_" + str(i + 1) + ".pickle", "wb")
        pickle.dump(X_train_split, pickle_out2)
        pickle_out2.close()

def read_split_data():

    for i in range(3):
        pickle_in = open("Lables/Y_train_split_" + str(i + 1) + ".pickle", "rb")
        Y_train_split = pickle.load(pickle_in)
        pickle_in2 = open("Images/X_train_split_" + str(i + 1) + ".pickle", "rb")
        X_train_split = pickle.load(pickle_in2)

        # Set the random seed
        random_seed = 2
        # Split the train and the validation 50% set for the fitting
        X_train, X_val, Y_train, Y_val= train_test_split(X_train_split, Y_train_split, test_size=0.3, random_state=random_seed)
        data_augmentation(X_train, X_val, Y_train, Y_val,i)
    model_eval(X_train, X_val, Y_train, Y_val, i)

def data_augmentation(X_train, X_val, Y_train, Y_val,i):

    i=i
    # With data augmentation to prevent overfitting (accuracy 0.99286)
    X_train, X_val, Y_train, Y_val=X_train, X_val, Y_train, Y_val

    global datagen
    datagen= ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    fit_model(X_train, X_val, Y_train, Y_val,i)

def model_defination():

    # Set the CNN model
    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    model= Sequential()

    model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1,1), padding='Same',
                     activation='relu', kernel_initializer='he_normal' ,input_shape=(28, 28, 1)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),strides=(1,1), padding='Same',
                     activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),strides=(1,1), padding='Same',
                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def fit_model(X_train, X_val, Y_train, Y_val,i):
    i=i
    model = model_defination()
    X_train, X_val, Y_train, Y_val = X_train, X_val, Y_train, Y_val
    print(len(X_train),len(Y_train),len(X_val),len(Y_val))


    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    epochs = 30# Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 100
    # Fit the model

    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(X_val, Y_val),
                                  verbose=2, steps_per_epoch=X_train.shape[0] // batch_size
                                  , callbacks=[learning_rate_reduction])
    model.save("Models/model_"+str(i+1)+"_custom")

def mostCommon(lst):
    val, count = mode(lst, axis=0)
    return val.ravel().tolist()


def model_eval(X_train, X_val, Y_train, Y_val,i):
    i = i
    X_train, X_val, Y_train, Y_val = X_train, X_val, Y_train, Y_val
    model = []
    predictions=[]
    for i in range(3):
        model.append(load_model("Models/model_" + str(i+1) + "_custom"))
        curr_model = model[i]
        Y_pred=curr_model.predict(X_val)
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        predictions.append(Y_pred_classes)

    Y_val_classes = np.argmax(Y_val, axis=1)
    final_prediction=mostCommon(predictions)
    print(m.accuracy_score(final_prediction, Y_val_classes))

def test_model():
    test = pd.read_csv("digit-recognizer/test.csv")
    X_test = test
    # Normalize the data
    X_test = X_test / 255.0
    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_test = X_test.values.reshape(-1, 28, 28, 1)
    model = []
    predictions = []
    for i in range(3):
        model.append(load_model("Models/model_" + str(i + 1) + "_custom"))
        curr_model = model[i]
        Y_pred = curr_model.predict(X_test)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        predictions.append(Y_pred_classes)

    final_prediction = mostCommon(predictions)
    imageID = list(range(1, len(final_prediction)+1))

    data=[imageID,final_prediction]
    export_data = zip_longest(*data, fillvalue='')
    with open('submission.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("ImageId", "Label"))
        wr.writerows(export_data)


    myfile.close()

if __name__ == '__main__':

    data_preparation()
    split_pickle_data()
    read_split_data()
    test_model()