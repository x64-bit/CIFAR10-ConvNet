'''
Creates computational graph of network
'''

'''
imports
-------------------------------------------------------------------------------
'''
# keras / tf
from keras.models import Sequential
from keras.layers import (
    InputLayer, Input, Conv2D, Dense)
from keras.layers import (
    Reshape, MaxPooling2D, Dropout, Flatten, BatchNormalization)
from keras.utils.vis_utils import plot_model

'''
create/load model
important note: dropout, batchNorm, etc. layers will be added with an image
editor to help fit the model architecture on one page
--------------------------------------------------------------------------------
'''

model = Sequential()

model.add(Conv2D(64,(3,3), input_shape=(32,32,3),strides=1,
                 padding='same',activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
# model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
# model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3),strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
# model.add(Dropout(0.2))

model.add(Flatten())
# model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

plot_model(model, to_file='C:/Users/Walter/Desktop/model_plot_V3.png', show_shapes=True)
