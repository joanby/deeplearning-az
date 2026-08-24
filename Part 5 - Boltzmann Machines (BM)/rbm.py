# Boltzmann Machines

"""
Comentarios y Descripción:
Cargar y preparar los datos:
Se cargan los datasets de películas, usuarios y valoraciones, y se convierten en matrices adecuadas para el entrenamiento.
Las valoraciones se convierten en binarias (1 si le gusta, 0 si no le gusta).
Red de Máquinas de Boltzmann (RBM):
Se crea una clase RBM, que tiene métodos para calcular las probabilidades de activación de las neuronas ocultas y visibles.
La función train actualiza los parámetros de la red a través de la regla de contraste positivo y negativo.
Entrenamiento:
Se entrena la RBM durante 10 épocas, utilizando un tamaño de batch de 100.
Se calcula y muestra la pérdida (error de reconstrucción) para cada época.
Prueba:
Se evalúa el rendimiento de la RBM en el conjunto de prueba, calculando la pérdida promedio.
"""

# Importación de librerías
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Carga del dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparación del conjunto de entrenamiento y conjunto de prueba
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Número de usuarios y películas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convertir los datos a una matriz con usuarios en filas y películas en columnas
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos a tensores de Torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convertir las valoraciones a valoraciones binarias: 1 (Me gusta) o 0 (No me gusta)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creación de la arquitectura de la Red Neuronal (RBM)
class RBM():
    def __init__(self, nv, nh):
        """
        Inicializa la Red de Máquinas de Boltzmann (RBM).
        
        nv: número de variables visibles (número de películas)
        nh: número de unidades ocultas
        """
        self.W = torch.randn(nh, nv)  # Pesos entre las neuronas visibles y ocultas
        self.a = torch.randn(1, nh)   # Sesgo para las neuronas ocultas
        self.b = torch.randn(1, nv)   # Sesgo para las neuronas visibles
    
    def sample_h(self, x):
        """ 
        Muestra las neuronas ocultas dadas las visibles (cálculo de la probabilidad de activación de las neuronas ocultas).
        """
        wx = torch.mm(x, self.W.t())  # Producto punto entre las entradas visibles y los pesos
        activation = wx + self.a.expand_as(wx)  # Activación de las neuronas ocultas
        p_h_given_v = torch.sigmoid(activation)  # Probabilidad de que las neuronas ocultas se activen
        return p_h_given_v, torch.bernoulli(p_h_given_v)  # Bernoulli para muestreo

    def sample_v(self, y):
        """ 
        Muestra las neuronas visibles dadas las ocultas (cálculo de la probabilidad de activación de las neuronas visibles).
        """
        wy = torch.mm(y, self.W)  # Producto punto entre las entradas ocultas y los pesos
        activation = wy + self.b.expand_as(wy)  # Activación de las neuronas visibles
        p_v_given_h = torch.sigmoid(activation)  # Probabilidad de que las neuronas visibles se activen
        return p_v_given_h, torch.bernoulli(p_v_given_h)  # Bernoulli para muestreo

    def train(self, v0, vk, ph0, phk):
        """ 
        Actualiza los parámetros de la RBM (pesos y sesgos) mediante la regla de contraste positivo y negativo.
        """
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()  # Actualización de los pesos
        self.b += torch.sum((v0 - vk), 0)  # Actualización de los sesgos visibles
        self.a += torch.sum((ph0 - phk), 0)  # Actualización de los sesgos ocultos

# Inicialización de las variables
nv = len(training_set[0])  # Número de variables visibles
nh = 100  # Número de unidades ocultas
batch_size = 100  # Tamaño del batch
rbm = RBM(nv, nh)  # Creación del objeto RBM

# Entrenamiento de la RBM
nb_epoch = 10  # Número de épocas
for epoch in range(1, nb_epoch + 1):
    train_loss = 0  # Pérdida de entrenamiento
    s = 0.  # Variable para contar las iteraciones
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]  # Datos de entrada (batch)
        v0 = training_set[id_user:id_user+batch_size]  # Datos de entrada iniciales
        ph0, _ = rbm.sample_h(v0)  # Muestra las neuronas ocultas
        for k in range(10):
            _, hk = rbm.sample_h(vk)  # Muestra las neuronas ocultas
            _, vk = rbm.sample_v(hk)  # Muestra las neuronas visibles
            vk[v0 < 0] = v0[v0 < 0]  # Mantiene las valoraciones no conocidas
        phk, _ = rbm.sample_h(vk)  # Muestra las neuronas ocultas después de la reconstrucción
        rbm.train(v0, vk, ph0, phk)  # Entrena la RBM usando el contraste positivo y negativo
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))  # Calcula la pérdida
        s += 1.
    print(f'Época: {epoch}, Pérdida: {train_loss/s}')

# Prueba de la RBM
test_loss = 0  # Pérdida de la prueba
s = 0.  # Variable para contar las iteraciones
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]  # Datos de entrada del usuario
    vt = test_set[id_user:id_user+1]  # Datos de prueba del usuario
    if len(vt[vt >= 0]) > 0:  # Si existen datos de prueba
        _, h = rbm.sample_h(v)  # Muestra las neuronas ocultas
        _, v = rbm.sample_v(h)  # Muestra las neuronas visibles
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))  # Calcula la pérdida
        s += 1.
print(f'Pérdida de prueba: {test_loss/s}')
