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

from keras.preprocessing.image import ImageDataGenerator

# Preprocesamiento de las imágenes de entrenamiento
train_datagen = ImageDataGenerator(rescale = 1./255,  # Normalizando las imágenes
                                   shear_range = 0.2,  # Aplicando transformación de corte
                                   zoom_range = 0.2,  # Aplicando aumento por zoom
                                   horizontal_flip = True)  # Aplicando volteo horizontal aleatorio

# Preprocesamiento de las imágenes de prueba
test_datagen = ImageDataGenerator(rescale = 1./255)  # Normalizando las imágenes de prueba

# Cargando el conjunto de datos de entrenamiento
training_set = train_datagen.flow_from_directory('dataset/training_set',  # Ruta del conjunto de entrenamiento
                                                 target_size = (64, 64),  # Tamaño de las imágenes
                                                 batch_size = 32,  # Tamaño de lote
                                                 class_mode = 'binary')  # Clasificación binaria (dos clases)

# Cargando el conjunto de datos de prueba
test_set = test_datagen.flow_from_directory('dataset/test_set',  # Ruta del conjunto de prueba
                                            target_size = (64, 64),  # Tamaño de las imágenes
                                            batch_size = 32,  # Tamaño de lote
                                            class_mode = 'binary')  # Clasificación binaria

# Ajustando la CNN al conjunto de entrenamiento
classifier.fit_generator(training_set,  # Entrenando la CNN
                         steps_per_epoch = 250,  # Número de pasos por cada época steps_per_epoch * batch_size = training_set
                         epochs = 25,  # Número de épocas
                         validation_data = test_set,  # Conjunto de validación
                         validation_steps = 2000)  # Número de pasos de validación
