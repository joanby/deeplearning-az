# Artificial Neural Network

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
from keras.layers import Dropout

# Inicializando la ANN
classifier = Sequential()

# Añadiendo la capa de entrada y la primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# classifier.add(Dropout(p = 0.1))  # Regularización Dropout si es necesario

# Añadiendo la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))  # Regularización Dropout si es necesario

# Añadiendo la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando la ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustando la ANN al conjunto de entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Parte 3 - Realizando predicciones y evaluando el modelo

# Prediciendo los resultados del conjunto de prueba
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # Convertimos la probabilidad en una clasificación binaria

# Prediciendo una nueva observación
"""Predecir si el cliente con la siguiente información dejará el banco:
Geografía: Francia
Puntuación de Crédito: 600
Género: Masculino
Edad: 40
Antigüedad: 3
Balance: 60000
Número de productos: 2
Tiene tarjeta de crédito: Sí
Es miembro activo: Sí
Salario estimado: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Creando la matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Parte 4 - Evaluando, Mejorando y Ajustando la ANN

# Evaluando la ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Función para construir la ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# Evaluación con validación cruzada
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Mejorando la ANN
# Regularización Dropout para reducir el sobreajuste si es necesario

# Ajustando la ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Función para construir la ANN con optimizador variable
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Definición de los parámetros para la búsqueda en cuadrícula
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

# Mostrando los mejores parámetros y la mejor exactitud
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
