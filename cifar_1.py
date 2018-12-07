'''
One layer only. 10 trials over test data.
CNN for recognizing CIFAR-10
'''

'''
imports ------------------------------------------------------------------------
'''

# math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

# keras / tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    InputLayer, Input, Conv2D, Dense)
from tensorflow.python.keras.layers import (
    Reshape, MaxPooling2D, Dropout, Flatten, BatchNormalization)
from tensorflow.python.keras.optimizers import Adam
from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array, load_img)
from tensorflow.python.keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold

# etc
import os, shutil, time

'''
loading and formatting data ----------------------------------------------------
'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

default_x_train = x_train
default_y_train = y_train

# scale pixel values to range of 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# one-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = 10

# show data
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()


'''
create/load model --------------------------------------------------------------
'''

epochs = 100
batch_size = 64
layer_num = 1

def create_model():
    model = Sequential()

    # block 1
    model.add(Conv2D(64,(3,3), input_shape=(32,32,3),strides=1,
                     padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    # block 2
    model.add(Conv2D(128,(3,3),strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    # block 3
    model.add(Conv2D(256,(3,3),strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    # flatten outputs
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    return model


# def load_trained_model():
#     model = load_model(model_path)
#     return model
'''
datagen ------------------------------------------------------------------------
'''

datagen = ImageDataGenerator(
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

# make sure datagen works
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(32, 32, 3), cmap=plt.get_cmap('gray'))
	# show the plot
	plt.show()
	break

# input training data into generator
train_generator = datagen.flow(
    x=x_train,
    y=y_train,
    batch_size=batch_size)

# create a folder and text file to store this model's metrics
filename = str(layer_num) +'_layers'
folder = "C:/Users/Walter/sci-fair/data/{}".format(str(layer_num) +'_layers')
scores_log = folder + "/{}".format('scores')
if not os.path.exists(folder):
    os.makedirs(folder)

'''
repeated trials ----------------------------------------------------------------
'''

final_scores = []

trials = 10

# model definition, training, logging
for i in range(trials):

    optimizer = Adam(lr=0.001)
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    '''
    creating custom dirs
    '''
    model_name = str(layer_num) + '_model'

    # path for this fold's specific stats
    trial_path = folder + "/{}".format('fold' + str(i))
    # log performance/history
    perf_log = trial_path + "/{}".format(filename + '_perf_log' + str(i))
    csv_log = trial_path + "/{}".format(filename + '_csv_log' + str(i))
    # save model
    save_path = trial_path + "/{}.hdf5".format(model_name)
    # plot paths
    acc_path = trial_path + "/{}.png".format('acc')
    loss_path = trial_path + "/{}.png".format('loss')

    if not os.path.exists(trial_path):
        os.makedirs(trial_path)


    '''
    training
    '''
    # train model & save checkpoints
    checkpointer = ModelCheckpoint(filepath=save_path,
                                verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_log)
    # start timer and train
    start_time = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train) // batch_size,
                                  epochs=epochs, validation_data=(x_test,y_test),
                                  callbacks=[checkpointer,csv_logger])
    # stop training timer
    end_time = str(time.time() - start_time)
    print(end_time)
    # evaluate
    score = model.evaluate(x=x_test, y=y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    final_scores.append(score[1] * 100)


    '''
    logging
    '''
    # plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.axis([0, epochs, 0, 1])
    plt.savefig(acc_path)
    plt.close()

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.axis([0, epochs, 0, 2.25])
    plt.savefig(loss_path)
    plt.close()

    # log data for future reference
    with open(perf_log, 'a') as metrics:
        metrics.write((str(layer_num) + '_layers'))
        metrics.write('\n')
        metrics.write('time: '+ end_time)
        metrics.write('\n')

        zipped_metrics = zip(model.metrics_names, score)
        for name, value in zipped_metrics:
            print(name, value)
            metrics.write(str(name))
            metrics.write(str(value))
            metrics.write('\n')

mean_score = np.mean(final_scores)

# write scores
with open(scores_log, 'a') as scores:
    for i in final_scores:
        scores.write(str(i))
        scores.write('\n')
    scores.write('avg. acc: ' + str(mean_score))

print(final_scores)
