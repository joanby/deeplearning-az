# Convolutional Neural Network (Red Neuronal Convolucional)

# Instalación de Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalación de Tensorflow
# pip install tensorflow

# Instalación de Keras
# pip install --upgrade keras

# Parte 1 - Construcción de la CNN

# Importando las librerías de Keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))  # Capa convolucional

# Paso 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))  # Capa de pooling

# Añadiendo una segunda capa convolucional
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))  # Segunda capa convolucional
classifier.add(MaxPooling2D(pool_size = (2, 2)))  # Capa de pooling

# Paso 3 - Aplanamiento (Flattening)
classifier.add(Flatten())  # Aplanando las características para alimentar la red neuronal densa

# Paso 4 - Conexión total (Full Connection)
classifier.add(Dense(units = 128, activation = 'relu'))  # Capa densa con 128 unidades
classifier.add(Dense(units = 1, activation = 'sigmoid'))  # Capa de salida con 1 unidad (binaria)

# Compilando la CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Parte 2 - Ajustando la CNN a las imágenes

import tensorflow as tf

# Reescalado y aumento de datos: en Keras 3, ImageDataGenerator se eliminó;
# rescale/shear_range/zoom_range/horizontal_flip pasan a ser capas que se
# aplican a cada lote, en vez de parámetros de un generador
rescale = tf.keras.layers.Rescaling(1./255)  # Normalizando las imágenes
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomShear(0.2),  # Aplicando transformación de corte
    tf.keras.layers.RandomZoom(0.2),  # Aplicando aumento por zoom
    tf.keras.layers.RandomFlip('horizontal'),  # Aplicando volteo horizontal aleatorio
])

# Cargando el conjunto de datos de entrenamiento
training_set = tf.keras.utils.image_dataset_from_directory('dataset/training_set',  # Ruta del conjunto de entrenamiento
                                                            image_size = (64, 64),  # Tamaño de las imágenes
                                                            batch_size = 32,  # Tamaño de lote
                                                            label_mode = 'binary')  # Clasificación binaria (dos clases)
training_set = training_set.map(lambda x, y: (data_augmentation(rescale(x), training = True), y))  # Solo el entrenamiento lleva aumento

# Cargando el conjunto de datos de prueba
test_set = tf.keras.utils.image_dataset_from_directory('dataset/test_set',  # Ruta del conjunto de prueba
                                                        image_size = (64, 64),  # Tamaño de las imágenes
                                                        batch_size = 32,  # Tamaño de lote
                                                        label_mode = 'binary')  # Clasificación binaria
test_set = test_set.map(lambda x, y: (rescale(x), y))  # La prueba solo se reescala

# Ajustando la CNN al conjunto de entrenamiento
classifier.fit(x = training_set,  # Entrenando la CNN
              validation_data = test_set,  # Conjunto de validación
              epochs = 25)  # Número de épocas
