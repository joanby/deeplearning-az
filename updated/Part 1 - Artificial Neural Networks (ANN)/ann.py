# Artificial Neural Network

"""
Descripción de las actualizaciones:
Codificación de datos categóricos:
Codificamos la columna Geography utilizando LabelEncoder y luego aplicamos OneHotEncoder mediante un ColumnTransformer para evitar la trampa de la variable ficticia.
Se asegura de que la variable Geography sea transformada correctamente y el resultado no tenga la columna redundante que podría causar problemas en el modelo.
Red Neuronal Artificial (ANN):
Estructura de la ANN: Se mantiene la arquitectura básica de 2 capas ocultas con 6 unidades cada una.
Activación: Se usa ReLU para las capas ocultas y Sigmoid para la capa de salida, ya que estamos haciendo una clasificación binaria.
Entrenamiento:
Utilizamos el optimizador Adam y la función de pérdida binary_crossentropy, que son comunes en problemas de clasificación binaria.
El modelo se entrena durante 100 épocas con un tamaño de lote de 10.
Predicción y Evaluación:
Después de hacer las predicciones sobre el conjunto de prueba, convertimos las probabilidades generadas por la red a un valor binario (True o False) mediante un umbral de 0.5.
Finalmente, calculamos la matriz de confusión para evaluar el rendimiento del modelo.
"""


# Instalación de Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalación de Tensorflow
# pip install tensorflow

# Instalación de Keras
# pip install --upgrade keras

# Parte 1 - Preprocesamiento de Datos

# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando el conjunto de datos
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



# Codificación de datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # Codificando la columna 'Geography'
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # Codificando la columna 'Gender'

# Aplicando OneHotEncoder para la columna 'Geography'
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Evitando la trampa de la variable ficticia eliminando la primera columna de la codificación OneHot
X = X[:, 1:]

# Dividiendo el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de características (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Parte 2 - Creación de la Red Neuronal Artificial (ANN)

# Importando las librerías de Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializando la ANN
classifier = Sequential()

# Añadiendo la capa de entrada y la primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Añadiendo la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Añadiendo la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando la ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustando la ANN al conjunto de entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Parte 3 - Realizando predicciones y evaluando el modelo

# Prediciendo los resultados del conjunto de prueba
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # Convertimos la probabilidad en una clasificación binaria


# Calculamos la matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# La imprimimos por pantalla
print(cm)
