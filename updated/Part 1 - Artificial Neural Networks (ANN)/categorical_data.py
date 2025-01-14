# Data Preprocessing

"""
Explicación Detallada:
Importación de bibliotecas:
Se importan las bibliotecas numpy, matplotlib, y pandas para la manipulación de datos y visualización.
Se utilizan funciones específicas de sklearn para el preprocesamiento de los datos.
Cargar el conjunto de datos:
Se carga el archivo CSV Data.csv y se separan las características (X) y la etiqueta dependiente (y). Las características son todas las columnas excepto la última, y la etiqueta es la cuarta columna (y = dataset.iloc[:, 3]).
Tratamiento de valores faltantes:
Imputer de sklearn.preprocessing se utiliza para manejar los valores faltantes en las características.
Se usan los valores medios de las columnas 1 y 2 para rellenar los valores faltantes en esas columnas. Esto se hace con la función fit_transform que ajusta el imputer y luego transforma los datos con los valores de reemplazo.
La línea X[:, 1:3] = imputer.transform(X[:, 1:3]) reemplaza los valores faltantes en las columnas seleccionadas de X.
Codificación de datos categóricos:
Codificación de la variable independiente:
Se utiliza LabelEncoder para convertir la primera columna de X (que contiene datos categóricos) en valores numéricos. LabelEncoder asigna un número entero para cada categoría distinta.
Luego, se aplica OneHotEncoder para convertir esas categorías en variables binarias (0 o 1) usando codificación One-Hot. Esto es necesario para modelos como regresión logística, que no pueden tratar datos categóricos directamente.
El parámetro categorical_features = [0] indica que la primera columna es una variable categórica que será transformada.
Codificación de la variable dependiente:
Se utiliza también LabelEncoder para convertir la variable dependiente (y) en valores numéricos, en caso de que esta también tenga datos categóricos. Esto se realiza para poder utilizarla en modelos que requieren valores numéricos.
Resultados Esperados:
Manejo de valores faltantes: Las columnas con valores faltantes se rellenan con el valor medio de las demás observaciones en esas columnas.
Codificación de características categóricas:
La primera columna de X se convierte de categorías en números enteros, y luego esos valores se transforman en variables binarias (One-Hot).
Codificación de la variable dependiente: Si la variable dependiente (y) también tiene valores categóricos, estos se convierten en números enteros.
La codificación One-Hot crea una columna extra para cada categoría. Si tienes muchas categorías, esto puede aumentar significativamente el tamaño de los datos.
"""

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importación del conjunto de datos
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Características
y = dataset.iloc[:, 3].values    # Etiqueta dependiente

# Tratamiento de valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])  # Ajuste sobre las columnas con valores faltantes
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Rellenado de valores faltantes

# Codificación de datos categóricos
# Codificación de la variable independiente
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # Codificando la primera columna
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()  # Aplicación de codificación OneHot

# Codificación de la variable dependiente
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  # Codificación de la variable dependiente
