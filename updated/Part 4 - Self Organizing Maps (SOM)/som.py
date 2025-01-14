# Self Organizing Map (Mapa Auto-Organizado)

# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando el conjunto de datos
dataset = pd.read_csv('Credit_Card_Applications.csv')  # Cargando el archivo CSV con las aplicaciones de tarjetas de crédito
X = dataset.iloc[:, :-1].values  # Seleccionando todas las columnas excepto la última como características
y = dataset.iloc[:, -1].values  # Seleccionando la última columna como las etiquetas (fraude o no fraude)

# Escalado de las características (normalización)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))  # Escalando los datos entre 0 y 1
X = sc.fit_transform(X)  # Aplicando el escalado a las características

# Entrenando el SOM (Mapa Auto-Organizado)
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)  # Inicializando el SOM
som.random_weights_init(X)  # Inicializando los pesos de manera aleatoria
som.train_random(data = X, num_iteration = 100)  # Entrenando el SOM con 100 iteraciones

# Visualizando los resultados
from pylab import bone, pcolor, colorbar, plot, show
bone()  # Estableciendo el gráfico base
pcolor(som.distance_map().T)  # Mostrando el mapa de distancias del SOM
colorbar()  # Mostrando la barra de colores
markers = ['o', 's']  # Definiendo los tipos de marcadores
colors = ['r', 'g']  # Definiendo los colores (rojo para fraude, verde para no fraude)
for i, x in enumerate(X):  # Iterando sobre todas las muestras
    w = som.winner(x)  # Obteniendo la posición del ganador (coordenadas en la cuadrícula del SOM)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],  # Usando 'o' para no fraude y 's' para fraude
         markeredgecolor = colors[y[i]],  # Usando color rojo para fraude y verde para no fraude
         markerfacecolor = 'None',  # Dejando el marcador vacío por dentro
         markersize = 10,
         markeredgewidth = 2)  # Ajustando el tamaño y grosor del borde del marcador
show()  # Mostrando el gráfico

# Encontrando fraudes
mappings = som.win_map(X)  # Obteniendo los mapas de los valores ganadores
frauds = np.concatenate((mappings[(6,1)], mappings[(7,8)]), axis = 0)  # Concatenando las muestras en las posiciones de fraude
frauds = sc.inverse_transform(frauds)  # Invirtiendo la normalización para obtener los valores originales
