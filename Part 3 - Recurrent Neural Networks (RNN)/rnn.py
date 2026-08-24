# Recurrent Neural Network (Red Neuronal Recurrente)

# Parte 1 - Preprocesamiento de los datos

# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando el conjunto de entrenamiento
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values  # Seleccionando los precios de apertura

# Escalado de las características (normalización)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))  # Escalando los valores entre 0 y 1
training_set_scaled = sc.fit_transform(training_set)

# Creando una estructura de datos con 60 pasos temporales y 1 salida
X_train = []
y_train = []
for i in range(60, 1258):  # Usamos 60 pasos previos para predecir el siguiente valor
    X_train.append(training_set_scaled[i-60:i, 0])  # Datos de entrada para cada instante
    y_train.append(training_set_scaled[i, 0])  # Etiquetas de salida (el valor siguiente)
X_train, y_train = np.array(X_train), np.array(y_train)

# Remodelando las entradas para que tengan la forma requerida por la RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Añadiendo la dimensión de características

# Parte 2 - Construcción de la RNN

# Importando las librerías de Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Inicializando la RNN
regressor = Sequential()

# Añadiendo la primera capa LSTM y regularización con Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))  # Capa LSTM
regressor.add(Dropout(0.2))  # Dropout para evitar el sobreajuste

# Añadiendo una segunda capa LSTM y Dropout
regressor.add(LSTM(units = 50, return_sequences = True))  # Segunda capa LSTM
regressor.add(Dropout(0.2))  # Dropout

# Añadiendo una tercera capa LSTM y Dropout
regressor.add(LSTM(units = 50, return_sequences = True))  # Tercera capa LSTM
regressor.add(Dropout(0.2))  # Dropout

# Añadiendo una cuarta capa LSTM y Dropout
regressor.add(LSTM(units = 50))  # Cuarta capa LSTM
regressor.add(Dropout(0.2))  # Dropout

# Añadiendo la capa de salida
regressor.add(Dense(units = 1))  # Capa densa con una sola unidad de salida

# Compilando la RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # Usando el optimizador Adam y el error cuadrático medio

# Ajustando la RNN al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)  # Entrenando la red con 100 épocas y tamaño de lote de 32

# Parte 3 - Realizando las predicciones y visualizando los resultados

# Obteniendo el precio real de las acciones de Google en 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values  # Precios reales de las acciones

# Obteniendo el precio predicho de las acciones de Google para 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  # Combinando los datos de entrenamiento y prueba
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values  # Seleccionando los últimos 60 valores
inputs = inputs.reshape(-1,1)  # Remodelando los datos de entrada
inputs = sc.transform(inputs)  # Escalando los datos de entrada
X_test = []
for i in range(60, 80):  # Usamos 60 valores previos para hacer la predicción
    X_test.append(inputs[i-60:i, 0])  # Preparando los datos de entrada
X_test = np.array(X_test)  # Convertimos a array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Remodelando para la RNN
predicted_stock_price = regressor.predict(X_test)  # Realizando las predicciones
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # Desescalando las predicciones

# Visualizando los resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio real de las acciones de Google')  # Graficando los precios reales
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio predicho de las acciones de Google')  # Graficando las predicciones
plt.title('Predicción del precio de las acciones de Google')  # Título del gráfico
plt.xlabel('Tiempo')  # Eje X
plt.ylabel('Precio de las acciones de Google')  # Eje Y
plt.legend()  # Leyenda
plt.show()  # Mostrando el gráfico
