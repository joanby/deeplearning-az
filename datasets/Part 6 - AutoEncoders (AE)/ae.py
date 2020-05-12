# Auto Encoders
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:51:46 2020

@author: juangabriel
"""

# Importar las librerías
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importar el dataset
movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users  = pd.read_csv("ml-1m/users.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings  = pd.read_csv("ml-1m/ratings.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparar el conjunto de entrenamiento y elconjunto de testing
training_set = pd.read_csv("ml-100k/u1.base", sep = "\t", header = None)
training_set = np.array(training_set, dtype = "int")
test_set = pd.read_csv("ml-100k/u1.test", sep = "\t", header = None)
test_set = np.array(test_set, dtype = "int")

# Obtener el número de usuarios y de películas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convertir los datos en un array X[u,i] con usuarios u en fila y películas i en columna
def convert(data):
    new_data = []
    for id_user in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos a tensores de Torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Crear la arquitectura de la Red Neuronal
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Entrenar el SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            # la media no es sobre todas las películas, sino sobre las que realmente ha valorado
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10) 
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector) ## sum(errors) / n_pelis_valoradas
            s += 1.
            optimizer.step()
    print("Epoch: "+str(epoch)+", Loss: "+str(train_loss/s))

# Evaluar el conjunto de test en nuestro SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        # la media no es sobre todas las películas, sino sobre las que realmente ha valorado
        mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10) 
        test_loss += np.sqrt(loss.data*mean_corrector) ## sum(errors) / n_pelis_valoradas
        s += 1.
print("Test Loss: "+str(test_loss/s))


