# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

import tensorflow as tf

# Rescaling and data augmentation: ImageDataGenerator was removed in Keras 3;
# rescale/shear_range/zoom_range/horizontal_flip are now layers applied to
# each batch instead of parameters of a generator
rescale = tf.keras.layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomShear(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomFlip('horizontal'),
])

training_set = tf.keras.utils.image_dataset_from_directory('dataset/training_set',
                                                            image_size = (64, 64),
                                                            batch_size = 32,
                                                            label_mode = 'binary')
class_names = training_set.class_names  # saved before augmentation: a mapped dataset no longer exposes class_names
training_set = training_set.map(lambda x, y: (data_augmentation(rescale(x), training = True), y))

test_set = tf.keras.utils.image_dataset_from_directory('dataset/test_set',
                                                        image_size = (64, 64),
                                                        batch_size = 32,
                                                        label_mode = 'binary')
test_set = test_set.map(lambda x, y: (rescale(x), y))

classifier.fit(x = training_set,
              validation_data = test_set,
              epochs = 25)

# Part 3 - Making new predictions

import numpy as np
import keras.utils as image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
class_names  # equivalent to the old training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'