import numpy as np
import matplotlib.pyplot as plt
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn
from math import sqrt
from typing import Callable, Optional

"""
    Implementación minimalista de los Mapas Auto-Organizados (Self Organizing Maps, SOM).
"""

def fast_norm(x: np.ndarray) -> float:
    """ 
    Devuelve la norma-2 (distancia euclidiana) de un arreglo unidimensional de NumPy.
    
    * Es más rápida que linalg.norm en caso de arreglos unidimensionales (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


class MiniSom:
    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0, 
                 learning_rate: float = 0.5, decay_function: Optional[Callable] = None, 
                 random_seed: Optional[int] = None):
        """
            Inicializa un Mapa Auto-Organizado (SOM).
            
            x, y - dimensiones del SOM
            input_len - número de elementos de los vectores de entrada
            sigma - dispersión de la función de vecindad (Gaussiana)
            learning_rate - tasa de aprendizaje inicial
            decay_function - función que reduce el learning_rate y sigma en cada iteración
            random_seed - semilla aleatoria para los cálculos
        """
        # Validación de sigma
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Advertencia: sigma es demasiado alto para las dimensiones del mapa.')
        
        # Inicialización del generador de números aleatorios
        self.random_generator = random.RandomState(random_seed) if random_seed else random.RandomState()
        
        # Función de decaimiento por defecto
        self._decay_function = decay_function if decay_function else lambda x, t, max_iter: x / (1 + t / max_iter)
        
        self.learning_rate = learning_rate  # Tasa de aprendizaje
        self.sigma = sigma  # Sigma (dispersion)
        
        # Inicialización aleatoria de los pesos
        self.weights = self.random_generator.rand(x, y, input_len) * 2 - 1
        for i in range(x):
            for j in range(y):
                # Normalización de los pesos
                self.weights[i, j] = self.weights[i, j] / fast_norm(self.weights[i, j])
        
        # Mapa de activación
        self.activation_map = zeros((x, y))
        
        # Definición de las coordenadas para la vecindad
        self.neigx = arange(x)
        self.neigy = arange(y)
        self.neighborhood = self.gaussian

    def _activate(self, x: np.ndarray):
        """ 
        Actualiza el mapa de activación, donde cada elemento i,j es la respuesta del 
        neurona i,j al patrón de entrada x.
        """
        s = subtract(x, self.weights)  # Calculamos la diferencia entre el patrón y los pesos de la red
        for idx in np.ndindex(self.activation_map.shape):
            self.activation_map[idx] = fast_norm(s[idx])  # Calculamos la norma (distancia) entre el patrón y los pesos

    def activate(self, x: np.ndarray) -> np.ndarray:
        """ 
        Devuelve el mapa de activación para el patrón de entrada x.
        """
        self._activate(x)
        return self.activation_map

    def gaussian(self, c: tuple[int, int], sigma: float) -> np.ndarray:
        """ 
        Devuelve una función Gaussiana centrada en c, para simular la vecindad de los mapas.
        """
        d = 2 * pi * sigma * sigma
        ax = exp(-power(self.neigx - c[0], 2) / d)
        ay = exp(-power(self.neigy - c[1], 2) / d)
        return outer(ax, ay)  # El producto externo genera una matriz bidimensional

    def winner(self, x: np.ndarray) -> tuple[int, int]:
        """ 
        Calcula las coordenadas de la neurona ganadora para un patrón de entrada x.
        """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def update(self, x: np.ndarray, win: tuple[int, int], t: int):
        """
        Actualiza los pesos de las neuronas.
        
        x - patrón actual para aprender
        win - posición de la neurona ganadora para x
        t - índice de la iteración
        """
        eta = self._decay_function(self.learning_rate, t, self.T)  # Aprendizaje en la iteración t
        sig = self._decay_function(self.sigma, t, self.T)  # Sigma también disminuye con el tiempo
        g = self.neighborhood(win, sig) * eta  # Mejora el rendimiento al usar vecindad
        for idx in np.ndindex(g.shape):
            self.weights[idx] += g[idx] * (x - self.weights[idx])  # Actualización de los pesos
            self.weights[idx] = self.weights[idx] / fast_norm(self.weights[idx])  # Normalización

    def quantization(self, data: np.ndarray) -> np.ndarray:
        """ 
        Asigna un código de libro (vector de pesos de la neurona ganadora) a cada muestra en data.
        """
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def random_weights_init(self, data: np.ndarray):
        """ 
        Inicializa los pesos del SOM eligiendo muestras aleatorias de los datos.
        """
        for idx in np.ndindex(self.weights.shape[:2]):
            self.weights[idx] = data[self.random_generator.randint(len(data))]
            self.weights[idx] = self.weights[idx] / fast_norm(self.weights[idx])

    def train_random(self, data: np.ndarray, num_iteration: int):
        """ 
        Entrena el SOM eligiendo muestras aleatorias de los datos en cada iteración.
        """
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            rand_i = self.random_generator.randint(len(data))  # Elegimos una muestra aleatoria
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data: np.ndarray, num_iteration: int):
        """ 
        Entrena el SOM utilizando todas las muestras de datos secuencialmente.
        """
        self._init_T(len(data) * num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data) - 1)  # Elegimos el índice dentro de los límites de los datos
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration: int):
        """ 
        Inicializa el parámetro T, que se utiliza para ajustar la tasa de aprendizaje.
        """
        self.T = num_iteration / 2  # Mantiene la tasa de aprendizaje casi constante en la última mitad de las iteraciones

    def distance_map(self) -> np.ndarray:
        """ 
        Devuelve el mapa de distancias de los pesos.
        """
        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        for idx in np.ndindex(um.shape):
            for ii in range(idx[0] - 1, idx[0] + 2):
                for jj in range(idx[1] - 1, idx[1] + 2):
                    if 0 <= ii < self.weights.shape[0] and 0 <= jj < self.weights.shape[1]:
                        um[idx] += fast_norm(self.weights[ii, jj, :] - self.weights[idx])
        um = um / um.max()  # Normalizamos el mapa de distancias
        return um

    def activation_response(self, data: np.ndarray) -> np.ndarray:
        """ 
        Devuelve una matriz en la que cada elemento i,j es el número de veces que la neurona i,j ha sido la ganadora.
        """
        a = zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data: np.ndarray) -> float:
        """ 
        Devuelve el error de cuantización, que es el promedio de las distancias entre cada muestra de entrada
        y su mejor neurona coincidente (BMU).
        """
        error = 0
        for x in data:
            error += fast_norm(x - self.weights[self.winner(x)])
        return error / len(data)

    def win_map(self, data: np.ndarray) -> defaultdict:
        """ 
        Devuelve un diccionario donde cada clave (i,j) tiene una lista con todos los patrones que 
        han sido mapeados en la posición i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap
