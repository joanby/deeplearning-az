# AutoEncoders con PyTorch

"""
Explicación del código:
Importación de librerías: Se importan las bibliotecas necesarias, como numpy, pandas y torch para el procesamiento de los datos y el entrenamiento de la red neuronal.
Cargar los datos: Se cargan los archivos de datos de MovieLens (movies.dat, users.dat, ratings.dat) y los conjuntos de entrenamiento y prueba (u1.base, u1.test).
Preparación de los datos:
Se transforman los conjuntos de entrenamiento y prueba en matrices donde cada fila representa a un usuario y cada columna a una película.
Se convierten las calificaciones en valores continuos (de 0 a 5) a un rango binario en el caso de las calificaciones (1 para "gustado", 0 para "no gustado").
Definición de la red neuronal:
Se crea una clase SAE (Autoencoder Esquemático), que define la arquitectura de la red con capas lineales y funciones de activación sigmoides.
Entrenamiento:
Se utiliza el optimizador RMSprop y la función de pérdida MSELoss para entrenar el modelo durante 200 épocas.
Se ajustan los pesos de la red para minimizar la diferencia entre las calificaciones predichas y las reales.
Evaluación:
Después del entrenamiento, se evalúa el modelo en el conjunto de prueba y se imprime la pérdida final.
"""

# Importación de las librerías necesarias
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Cargar el conjunto de datos
# Se cargan los datos de películas, usuarios y valoraciones de MovieLens
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparación de los conjuntos de entrenamiento y prueba
# Se leen los conjuntos de entrenamiento y prueba de MovieLens
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Obtener el número de usuarios y películas
# `nb_users` es el número total de usuarios y `nb_movies` el número total de películas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Función para convertir los datos a una matriz donde las filas son usuarios y las columnas son películas
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):  # Iterar sobre los usuarios
        id_movies = data[:, 1][data[:, 0] == id_users]  # Películas vistas por el usuario
        id_ratings = data[:, 2][data[:, 0] == id_users]  # Calificaciones dadas por el usuario
        ratings = np.zeros(nb_movies)  # Inicializar un vector de calificaciones con ceros
        ratings[id_movies - 1] = id_ratings  # Asignar las calificaciones a las películas
        new_data.append(list(ratings))  # Añadir la fila de calificaciones para el usuario
    return new_data

# Convertir los conjuntos de datos de entrenamiento y prueba a matrices
training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos a tensores de PyTorch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Crear la arquitectura del Autoencoder (SAE)
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        # Definir las capas del Autoencoder
        self.fc1 = nn.Linear(nb_movies, 20)  # Capa de entrada
        self.fc2 = nn.Linear(20, 10)  # Capa intermedia
        self.fc3 = nn.Linear(10, 20)  # Capa intermedia
        self.fc4 = nn.Linear(20, nb_movies)  # Capa de salida
        self.activation = nn.Sigmoid()  # Función de activación Sigmoide

    def forward(self, x):
        # Propagación hacia adelante (forward pass)
        x = self.activation(self.fc1(x))  # Aplicar la activación en la primera capa
        x = self.activation(self.fc2(x))  # Aplicar la activación en la segunda capa
        x = self.activation(self.fc3(x))  # Aplicar la activación en la tercera capa
        x = self.fc4(x)  # Capa de salida sin activación (se utiliza MSE)
        return x

# Crear una instancia del Autoencoder
sae = SAE()

# Definir la función de pérdida (Loss Function) y el optimizador
criterion = nn.MSELoss()  # Función de pérdida MSE (Error cuadrático medio)
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)  # Optimizador RMSprop

# Entrenamiento del Autoencoder
nb_epoch = 200  # Número de épocas de entrenamiento

for epoch in range(1, nb_epoch + 1):  # Iterar sobre las épocas
    train_loss = 0  # Inicializar la pérdida de entrenamiento
    s = 0.  # Inicializar el contador de usuarios procesados
    for id_user in range(nb_users):  # Iterar sobre todos los usuarios
        input = Variable(training_set[id_user]).unsqueeze(0)  # Convertir los datos del usuario a tensor
        target = input.clone()  # Clonar la entrada como objetivo
        if torch.sum(target.data > 0) > 0:  # Si el usuario ha dado alguna calificación
            output = sae(input)  # Obtener la salida del Autoencoder
            target.require_grad = False  # Desactivar gradientes para el objetivo
            output[target == 0] = 0  # No calcular la pérdida para las entradas no calificadas
            loss = criterion(output, target)  # Calcular la pérdida
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)  # Corrector para normalizar
            loss.backward()  # Retropropagar el error
            train_loss += np.sqrt(loss.data * mean_corrector)  # Acumular la pérdida
            s += 1.  # Incrementar el contador de usuarios procesados
            optimizer.step()  # Actualizar los pesos de la red

    print(f'Epoca: {epoch} Pérdida de entrenamiento: {train_loss / s}')  # Imprimir la pérdida promedio por época

# Evaluación del modelo en el conjunto de prueba
test_loss = 0  # Inicializar la pérdida de prueba
s = 0.  # Inicializar el contador de usuarios procesados
for id_user in range(nb_users):  # Iterar sobre todos los usuarios
    input = Variable(training_set[id_user]).unsqueeze(0)  # Convertir los datos del usuario a tensor
    target = Variable(test_set[id_user])  # Obtener las calificaciones reales del conjunto de prueba
    if torch.sum(target.data > 0) > 0:  # Si el usuario tiene calificaciones en el conjunto de prueba
        output = sae(input)  # Obtener la salida del Autoencoder
        target.require_grad = False  # Desactivar gradientes para el objetivo
        output[(target == 0).unsqueeze(0)] = 0  # No calcular la pérdida para las entradas no calificadas
        loss = criterion(output, target)  # Calcular la pérdida
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)  # Corrector de normalización
        test_loss += np.sqrt(loss.item() * mean_corrector)  # Acumular la pérdida de prueba
        s += 1.  # Incrementar el contador de usuarios procesados

# Imprimir la pérdida final en el conjunto de prueba
print(f'Pérdida de prueba: {test_loss / s}')
