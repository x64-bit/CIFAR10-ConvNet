'''
returns random image + augmented image
'''

# math
import matplotlib.pyplot as plt
import math

# keras / tf
from tensorflow.python.keras.datasets import cifar10
from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array, load_img)
from keras.utils import np_utils

'''
loading data
--------------------------------------------------------------------------------
'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# scale pixel values to range of 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# one-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# show data
for i in range(6):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

'''
datagen
--------------------------------------------------------------------------------
'''

datagen = ImageDataGenerator(
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

# check datagen
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=6, shuffle=False):
	# create a grid of 3x3 images
	for i in range(6):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(32, 32, 3), cmap=plt.get_cmap('gray'))
	# show the plot
	plt.show()
	break

# feed training data into generator
train_generator = datagen.flow(
    x=x_train,
    y=y_train,
    batch_size=batch_size)
