# Cambios 2026

Este documento explica qué se ha tocado en el código del curso (Python) para
que vuelva a ejecutarse en 2026, y por qué. No es un diff: es la explicación
de qué decidió cada librería entre 2020 (cuando se grabó el curso, en
Spyder, sobre los ficheros `.py`) y ahora, y qué se ha hecho al respecto.

Todo lo que hay en esta rama son las mismas clases del vídeo. Lo que sigue
cambia cómo se le pide algo a una librería, casi nunca lo que obtienes. Hay
una única sección, la de la CNN, donde el código deja de parecerse línea a
línea a lo que ves en pantalla — está marcada bien claro, y se explica por
qué no había otra forma.

## Versiones con las que se ha verificado

Verificado en **agosto de 2026**, en **macOS con Apple Silicon**:

| Paquete | Versión |
|---|---|
| Python | 3.13.15 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| pytorch | 2.13.0 |
| tensorflow | 2.21.0 |
| keras | 3.15.1 |
| scikeras | 0.13.0 |
| jupyter | 1.1.1 |

`scikeras` no lo instalaba el curso original: es un paquete nuevo, de
terceros, que hizo falta añadir — se explica en la causa 3 de más abajo.

Si usas otro sistema operativo o instalas versiones distintas de estas
librerías, es posible que algo de lo que se cuenta aquí ya no aplique tal
cual, para bien o para mal.

## Lo que rompieron las librerías

### 1. `OneHotEncoder(categorical_features=...)` desapareció de scikit-learn

**Afecta a:** `ann.py`, `ann_homework_solution.py`, `evaluating_improving_tuning.py`
y `categorical_data.py`, los cuatro en `Part 1 - Artificial Neural Networks (ANN)`.

**Qué se rompía.**
`OneHotEncoder(categorical_features = [1])` moría con
`TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'categorical_features'`.

**Por qué lo cambió la librería.** `categorical_features` le decía al
`OneHotEncoder` qué columna concreta codificar dentro de una matriz con
columnas mixtas (texto y números). scikit-learn eliminó ese parámetro hace
varias versiones: hoy esa selección de columna se hace por fuera, envolviendo
el `OneHotEncoder` en un `ColumnTransformer`.

**El detalle que merece contarse aparte.** Dos de estos cuatro ficheros
(`ann.py` y `ann_homework_solution.py`) llevaban, desde una revisión anterior
del curso, un docstring en español que explicaba exactamente este arreglo
—"aplicamos `OneHotEncoder` mediante un `ColumnTransformer` para evitar la
trampa de la variable ficticia"— sin que el código de debajo lo aplicara
nunca. El comentario llevaba tiempo diciendo la verdad y el código llevaba
el mismo tiempo sin hacerle caso. Se ha hecho lo que el docstring ya
prometía. `categorical_data.py` tenía el problema contrario: su comentario
sí describía el código viejo (`categorical_features = [0]`) y se ha quedado
sin actualizar al aplicar este mismo arreglo — ya corregido para que
describa el `ColumnTransformer` de arriba.

```python
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
```

Se comprobó que el resultado es equivalente al de siempre: mismo `shape`,
mismo `dtype` (`float64`), y el orden de columnas que la siguiente línea del
curso ya esperaba (la que evita la trampa de la variable ficticia) se
mantiene igual.

### 2. `Dropout(p=...)` de Keras 3 pide `rate`, no `p`

Keras 3 renombró el argumento del `Dropout`: llamarlo con `p=` revienta con
`TypeError: Dropout.__init__() missing 1 required positional argument: 'rate'`.
Es una rotura real, y la vas a ver documentada si comparas con el curso
hermano *Machine Learning de A a la Z*.

En este curso, sin embargo, no hizo falta tocar nada en esta rama: las dos
llamadas a `Dropout(p = 0.1)` de `evaluating_improving_tuning.py` están
comentadas en el propio fichero (`# classifier.add(Dropout(p = 0.1))`), y
las cuatro llamadas activas de `rnn.py` ya usaban la forma posicional
(`Dropout(0.2)`), compatible con Keras 3 sin cambios. El único sitio del
repositorio donde `Dropout(p=...)` sí estaba activo y rompía era
`original/Part 1/ann.py` — y `original/` no se toca, vive en `master` tal
como se grabó.

### 3. `keras.wrappers.scikit_learn` desapareció — y con él, dos nombres más

**Afecta a:** `evaluating_improving_tuning.py` y `ann.ipynb`, los dos en
`Part 1`.

**Qué se rompía.**
`from keras.wrappers.scikit_learn import KerasClassifier` moría con
`ModuleNotFoundError: No module named 'keras.wrappers.scikit_learn'`.

**Por qué lo cambió la librería.** Ese submódulo envolvía un modelo Keras
como si fuera un estimador de scikit-learn — necesario para usar
`cross_val_score` y `GridSearchCV` sobre una red neuronal. Keras 3 lo
eliminó por completo. Su sustituto, **`scikeras`**, es un paquete
independiente de terceros que el curso nunca instalaba ni mencionaba: no
viene con Keras ni con TensorFlow, hay que añadirlo aparte.

Cambiar de paquete arrastró dos cambios más, porque `scikeras` no es un
sustituto que se comporte exactamente igual:

- `KerasClassifier(build_fn = build_classifier, ...)` pasa a
  `KerasClassifier(model = build_classifier, ...)`. `scikeras` acepta
  `build_fn` como alias con aviso de deprecación, pero se ha usado `model`
  por ser el nombre soportado a futuro.
- En el diccionario `parameters` que alimenta al `GridSearchCV`, la clave
  `'optimizer'` pasa a **`'model__optimizer'`**. Esto no es cosmético:
  `optimizer` es ya un parámetro de primer nivel del propio `KerasClassifier`
  de `scikeras` (con su propio significado), así que no se reenvía sin más a
  tu función `build_classifier(optimizer)` aunque el nombre coincida —
  `scikeras` documenta el prefijo `model__` precisamente para este caso.
  **Es una clave que tecleas en el vídeo** y que aquí verás distinta.
- En `ann.ipynb` (no en el `.py`: el notebook usa todavía la API de Keras 1)
  apareció además `nb_epoch`, el nombre antiguo del parámetro de épocas, que
  `scikeras` no reconoce en absoluto — se ha renombrado a `epochs`, el mismo
  nombre que ya usaba `evaluating_improving_tuning.py`.

**Lo que no se ha probado a esta escala.** El `GridSearchCV` real del curso
prueba hasta 500 épocas con 10 particiones de validación cruzada. Verificar
eso completo son horas de cómputo. Se ha comprobado que el mecanismo
funciona de principio a fin con una rejilla reducida (2 épocas, 2
particiones, un valor por clave) — confirma que `model__optimizer` enruta
correctamente al `build_fn`, no que la rejilla completa termine ni cuánto
tarda.

### 4. `ImageDataGenerator` desapareció con Keras 3 — y con él, una forma entera de hacer las cosas

**Afecta a:** `cnn.py`, `cnn_homework_solution.py` y `CNN.ipynb`, los tres en
`Part 2 - Convolutional Neural Networks (CNN)`. Es el único cambio de este
documento donde el código deja de parecerse línea a línea al vídeo.

**Qué se rompía.**
`from keras.preprocessing.image import ImageDataGenerator` moría
directamente con
`ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image'`.

**Por qué lo cambió la librería, y qué significa de verdad.**
`ImageDataGenerator` era una única clase que hacía cuatro trabajos distintos
sobre las fotos de una carpeta: las cargaba, las redimensionaba, las
reescalaba a valores entre 0 y 1 (`rescale = 1./255`), y les aplicaba
aumento de datos aleatorio en cada pasada —recortes, zooms y giros
(`shear_range`, `zoom_range`, `horizontal_flip`)— para que la red no se
aprendiera de memoria las 8.000 fotos de entrenamiento y generalizara mejor.
Keras 3 eliminó esa clase entera, junto con todo el pipeline que
representaba.

Lo que la sustituye, `image_dataset_from_directory`, solo resuelve **dos**
de esos cuatro trabajos: carga las imágenes y las redimensiona. Ni reescala
ni aumenta datos por sí sola. Esto no es una carencia que se rodee con más
configuración: es un cambio de filosofía deliberado. **El aumento de datos
deja de ser un parámetro de "el objeto que carga fotos" y pasa a ser capas
normales de la red neuronal** (`Rescaling`, `RandomShear`, `RandomZoom`,
`RandomFlip`), que se ejecutan sobre cada lote de imágenes igual que se
ejecuta una capa convolucional. Antes preparabas los datos *antes* de
dárselos a la red; ahora esa preparación es parte del propio grafo de
cómputo de la red. Es la misma solución, con el mismo porqué, que se aplicó
en la CNN de *Machine Learning de A a la Z*: si haces los dos cursos, te vas
a encontrar el mismo patrón en el mismo punto.

**Qué se hace ahora, y qué es distinto de lo que ves en el vídeo.**

- Antes de cargar las imágenes, se declaran las transformaciones como capas:
  `Rescaling(1./255)` para el reescalado, y un `Sequential` con
  `RandomShear`, `RandomZoom` y `RandomFlip` para el aumento de datos.
- `train_datagen.flow_from_directory(...)` pasa a ser
  `image_dataset_from_directory(...)` (mismo directorio, mismo
  `batch_size = 32`; `target_size` ahora se llama `image_size`), seguido de
  un `.map(...)` que aplica el reescalado y el aumento a cada lote. El
  conjunto de test se reescala igual, pero sin aumento de datos — igual que
  en el vídeo.
- `class_indices` (un diccionario) pasa a ser `class_names` (una lista),
  capturado antes del `.map()`: el nuevo pipeline no expone el diccionario
  de siempre. Esto solo afecta a `cnn_homework_solution.py` y `CNN.ipynb`,
  los dos que hacen la predicción sobre una imagen suelta; `cnn.py` no
  llega a esa línea, así que no le toca.
- La arquitectura de la CNN y su compilación no se han tocado. El
  entrenamiento con `epochs = 25` tampoco cambia el resultado, pero **sí
  cambia cómo se teclea**: `classifier.fit_generator(training_set,
  steps_per_epoch = 250, epochs = 25, validation_data = test_set,
  validation_steps = 2000)` pasa a `classifier.fit(x = training_set,
  validation_data = test_set, epochs = 25)`. Keras 3 retiró
  `fit_generator` — `fit` ya sabe distinguir un `tf.data.Dataset` de un
  array — y `steps_per_epoch`/`validation_steps` desaparecen porque ese
  `Dataset` conoce su propia longitud: son dos argumentos menos que
  tecleabas en el vídeo.
- La predicción final sobre una imagen suelta tampoco es línea a línea la
  misma: `from keras.preprocessing import image` pasa a `import keras.utils
  as image`, porque Keras 3 movió `load_img` a ese módulo. Es el import con
  el que arranca la Parte 3 de la sección, justo antes de la predicción.

Dos valores del aumento de datos no tienen un equivalente exacto entre la
clase vieja y las capas nuevas — se ha mantenido el número que pide el
vídeo, pero el efecto visual no es idéntico:

- `shear_range = 0.2` era un ángulo. `RandomShear(0.2)` interpreta ese `0.2`
  como una fracción del ancho de la imagen — la cizalladura que verás es
  notablemente más fuerte que la del original.
- `zoom_range = 0.2` hacía zoom hacia dentro y hacia fuera a la vez.
  `RandomZoom(0.2)` con un solo número solo hace zoom hacia fuera.

`horizontal_flip` y el reescalado sí son exactamente equivalentes.

## Lo que rompió la propia historia del repositorio, no ninguna librería

### 5. Los 7 notebooks llevaban rotos desde agosto de 2023 — también en Colab

**Afecta a:** los 7 `.ipynb` del curso (`ann.ipynb`, `CNN.ipynb`, `RNN.ipynb`,
`som.ipynb`, `mega_case_study.ipynb`, `rbm.ipynb`, `ae.ipynb`).

Estos notebooks no se grabaron en el vídeo — el curso se grabó en Spyder,
sobre los `.py`— se añadieron después como vía alternativa para seguir el
curso en Google Colab, y su primera celda hacía `!git clone
.../deeplearning-az.git` para traerse una carpeta `datasets/` con todos los
CSV. El commit `93369b0` ("Update Agosto 2023") **borró esa carpeta** al
reorganizar el repositorio en `original/`/`updated/`. Desde ese commit, los
7 notebooks clonan el repositorio, buscan una carpeta que ya no existe, y
fallan — en Colab exactamente igual que en local. Nadie los volvió a
ejecutar desde entonces: llevaban rotos tres años.

De propina, la segunda celda de cada notebook hacía
`drive.mount('/content/drive')`, y era **código muerto**: los datos siempre
vinieron del `git clone`, nunca de Google Drive. Se comprobó leyendo los 7
notebooks completos — ninguna lectura de datos usa una ruta
`/content/drive/...`. En `ae.ipynb` la propia salida guardada de esa celda,
de 2020, es un `KeyboardInterrupt`: ni en la grabación original llegó a
montarse Drive con éxito.

**Qué se hizo:**

- Se copiaron los 7 notebooks a la carpeta de su parte correspondiente, al
  lado de los datos que ya usan los `.py` de esa misma parte.
- Se borró la celda de `drive.mount` y el resto de configuración de Colab
  que no aporta nada en local y que además rompería la ejecución si se
  dejara: `!pip install sklearn` (el paquete `sklearn` de PyPI da hoy un
  error de instalación explícito), la instalación de Theano por
  `git://` (protocolo retirado de GitHub, y ningún fichero del curso usa
  Theano), reinstalaciones de `keras`/`tensorflow` que pisarían las
  versiones ya fijadas, y el bloque de utilidades de Google Colab, sin uso
  en ningún sitio. `CNN.ipynb` y `RNN.ipynb` tenían además una celda `pip
  install --upgrade keras` **sin el `!` inicial** — sintácticamente
  inválida como código Python, un bug propio del notebook anterior a
  cualquier problema de Colab.
- La celda del `git clone` no se borró, se condicionó: en local es un
  no-op (no hace ninguna llamada de red), en Colab sigue clonando el
  repositorio para que quien lo use ahí tenga sus datos, exactamente como
  antes.
- Las rutas de datos pasan a ser relativas al propio notebook, en vez de a
  `/content/deeplearning-az/datasets/...`.

Los notebooks también recibieron los arreglos que les tocaban de las causas
de arriba, donde aplicaban: `random_seed` en los dos que usan `MiniSom` (ver
causa 7), y el import a `scikeras` con sus tres claves en `ann.ipynb` (causa
3). `CNN.ipynb` recibió la misma reescritura que su `.py` gemelo (causa 4).

### 6. Faltaban dos CSV en `Part 1`, nunca copiados al reorganizar

`categorical_data.py` pedía `Data.csv` y `classification_template.py` pedía
`Social_Network_Ads.csv`; los dos morían con `FileNotFoundError`. Los dos
son plantillas heredadas de una sección anterior del curso ("Get the
Machine Learning Basics") que una revisión previa copió a `Part 1` sin
copiar también su CSV al lado. No lo rompió ninguna librería: fue un
descuido de esa reorganización. Se han copiado los dos CSV a la carpeta y
ya se ejecutan sin tocar una línea de código.

### 7. `som.py` nunca fue reproducible, ni en 2020

**Afecta a:** `som.py`, `som.ipynb` y `mega_case_study.ipynb`, en `Part 4 -
Self Organizing Maps (SOM)`.

Esta es la más importante de contar bien, porque no es un fallo con
mensaje de error: el fichero se ejecutaba, terminaba con éxito, y a veces
daba un resultado distinto al del vídeo sin avisar de nada.

**Qué pasaba.** La clase enseña a entrenar un mapa autoorganizado sobre
solicitudes de tarjeta de crédito, mirar el mapa resultante, y **elegir a
ojo los cuadrados más blancos** —las neuronas más alejadas de sus vecinas—
porque ahí es donde se concentran las solicitudes atípicas, candidatas a
fraude. El fichero del curso lee luego esas celdas con sus coordenadas
**escritas a mano**: `mappings[(6,1)]` y `mappings[(7,8)]`, las que salieron
el día que se grabó el vídeo.

El problema es que el SOM se entrena con pesos iniciales aleatorios
(`som.random_weights_init(X)` / `som.train_random(...)`) **sin ninguna
semilla fijada**. Cada vez que se ejecuta el fichero, el mapa que sale es
distinto, y las celdas que "ganan" —las que de verdad tienen las muestras
atípicas— cambian con él. Casi siempre `(6,1)` y `(7,8)` seguían teniendo
algo dentro y el fichero terminaba bien, aunque ya mostrando un mapa
distinto al del vídeo. Pero a veces una de las dos quedaba completamente
vacía, y entonces `np.concatenate` revienta porque intenta unir un array
2D con uno vacío de 1D:

```
ValueError: all the input arrays must have same number of dimensions, but
the array at index 0 has 2 dimension(s) and the array at index 1 has 1
dimension(s)
```

Es un fallo intermitente por diseño: unas veces sale, otras no, y el propio
código nunca dio ninguna pista de que dependiera de un sorteo.

**Qué se hace ahora.** Se ha fijado una semilla en el propio constructor de
`MiniSom` —el único lugar donde de verdad se puede controlar el sorteo, ver
más abajo— elegida específicamente para que las celdas que ganan sean
las mismas que el alumno ve señalar en el vídeo:

- `som.py` y `som.ipynb` (mismo dato, es el notebook gemelo):
  `random_seed = 33` (la celda `(6,1)` sale con 13 muestras y la `(7,8)`
  con 19 — margen cómodo, no un empate al límite).
- `mega_case_study.ipynb`: usa las mismas ideas pero con datos y celdas de
  fraude propios de esa lección (`(3,1)`/`(2,5)`), así que necesitaba su
  propia búsqueda. Elegida `random_seed = 17` (18 y 23 muestras
  respectivamente).

Las dos semillas se encontraron probando un barrido de 0 a 99 y quedándose
con la primera que diera margen cómodo en las dos celdas — de 100 semillas
probadas, 39 sirven para `som.py` y 37 para `mega_case_study.ipynb`. Una
vez fijadas, se comprobó que el resultado es determinista de verdad: tres
ejecuciones seguidas, mismo `frauds.shape` y misma suma de contenido las
tres veces, no solo "no lanzó excepción".

Dos avisos técnicos que costó descubrir y que merece la pena dejar
anotados, por si el patrón reaparece en otro curso: un `np.random.seed()`
puesto antes de crear el `MiniSom` **no sirve de nada**, porque
`MiniSom.__init__` crea su propio generador aleatorio interno que no lee el
estado global de NumPy — la única semilla que cuenta es el parámetro
`random_seed` del propio constructor. Y ese parámetro, tal como está escrito
en `minisom.py`, trata `random_seed = 0` como "sin semilla" (`if 0` es
`False` en Python), así que `0` no es una opción válida — de ahí que se
haya elegido `33` y `17`, no `0`.

Es un cambio de una línea (se añade un argumento a una llamada que el
alumno ya teclea) y, al contrario que los demás cambios de este documento,
no corrige algo que rompiera una librería: corrige un problema que el
fichero arrastraba desde 2020, y que solo se nota si te toca la mala suerte
en el sorteo.

## Lo que puede verse distinto al vídeo

- **Codificación categórica** (`ann.py`, `ann_homework_solution.py`,
  `categorical_data.py`): donde el vídeo teclea
  `OneHotEncoder(categorical_features = [1])`, aquí ves un `ColumnTransformer`
  envolviendo el `OneHotEncoder`. El resultado —qué columnas salen y con qué
  valores— es idéntico; solo cambia cómo se lo pides a la librería.
- **`evaluating_improving_tuning.py`**: el import pasa de
  `keras.wrappers.scikit_learn` a `scikeras.wrappers`, `build_fn=` pasa a
  `model=`, y la clave `'optimizer'` del diccionario que rellena el
  `GridSearchCV` pasa a `'model__optimizer'`. Son tres cosas que sí tecleas
  en el vídeo tal como estaban antes.
- **Sección CNN** (`cnn.py`, `cnn_homework_solution.py`, `CNN.ipynb`): como
  se explica en la causa 4, el bloque de carga y aumento de datos está
  reescrito. La cizalladura y el zoom del aumento de datos se ven algo más
  fuertes que en el vídeo, por el motivo técnico ya explicado ahí. Además,
  `fit_generator(...)` pasa a `fit(...)` (con `steps_per_epoch` y
  `validation_steps` fuera) y, en la predicción, `from keras.preprocessing
  import image` pasa a `import keras.utils as image`. Son dos líneas más que
  tecleas en el vídeo y que aquí verás distintas.
- **`som.py`, `som.ipynb` y `mega_case_study.ipynb`**: ahora llevan un
  `random_seed` en la llamada a `MiniSom` que el vídeo no teclea. El efecto
  es justo el contrario de "distinto al vídeo": es lo que hace que el mapa
  señale las mismas celdas de fraude que ves elegir en pantalla, en vez de
  un mapa aleatorio distinto cada vez.
- **Los notebooks en general** (`ann.ipynb`, `CNN.ipynb`, `RNN.ipynb`,
  `som.ipynb`, `mega_case_study.ipynb`, `rbm.ipynb`, `ae.ipynb`): no
  aparecen en ningún vídeo — el curso se grabó en Spyder sobre los `.py`—
  así que no hay una pantalla de referencia con la que compararlos. Se han
  arreglado para que funcionen tanto en local como en Colab, con el mismo
  contenido que su `.py` gemelo cuando existe uno.
- **Al pasar de la Parte 3 a la Parte 5, reinicia la consola de Spyder.**
  Las Partes 1-3 usan Keras/TensorFlow y las Partes 5-6 usan PyTorch, y
  `import torch` + `import tensorflow` en el mismo proceso de Python
  segfaultan (`exit 139`, sin traceback, en cualquier orden de import). No
  es un fallo de tu código: es que las dos librerías no conviven en la
  misma consola. Una consola nueva por parte lo resuelve.

## Lo que NO se ha ejecutado

Esta rama tiene 19 ficheros de Python, pero uno de ellos, `minisom.py`, es
una librería de terceros vendorizada que solo define clases — no es código
del curso. De los **18 ficheros del curso, 11 se ejecutan enteros y 7
quedan omitidos por coste de cómputo** (`minisom.py` se ejecuta sin problema,
como es de esperar en un fichero que solo define clases):

- La **CNN** (`cnn.py`, `cnn_homework_solution.py`, `CNN.ipynb`): 25 épocas
  sobre 8.000 imágenes de entrenamiento.
- El **autocodificador** (`ae.py`, `ae.ipynb`): 200 épocas, entrenadas
  usuario a usuario sobre 6.040 usuarios de MovieLens 1M.
- El **`GridSearchCV`** (`evaluating_improving_tuning.py`, `ann.ipynb`):
  hasta 500 épocas por combinación, con 10 particiones de validación
  cruzada cada una.

Dicho sin maquillar: **sabemos que el pipeline funciona; no hemos
comprobado que reproduzca los números del vídeo.** Los 7 se han verificado
con una versión reducida —una sola época en la CNN y el autocodificador,
una rejilla de un único valor por clave y 2 particiones en el
`GridSearchCV`— para confirmar que el código corre de principio a fin sin
errores. Los números que salen de esas pruebas reducidas (por ejemplo, un
`val_accuracy` de la CNN con 1 época) no son los del vídeo y no se
presentan como tales: con una fracción del entrenamiento no hay manera
honesta de que lo sean.

Esto no es que este curso esté peor verificado que otros: Deep Learning es
caro de ejecutar por naturaleza, mucho más que un modelo de regresión o de
clustering. El curso hermano, *Machine Learning de A a la Z*, tuvo 2
ficheros omitidos de 60 por este mismo motivo (una única CNN); aquí son 7
de 18 porque casi todo el curso son redes que se entrenan durante minutos u
horas por época, no segundos.

## Lo que sigue roto

Nada, con una acotación honesta: de los 18 ficheros del curso, los 11 que
se ejecutan enteros terminan sin error, y los 7 omitidos por coste se han
verificado con una versión reducida que confirma que el código corre de
principio a fin. Queda además
`Part 5 - Boltzmann Machines (BM)/ml-1m/Train_Test_Set_Creation.R`, un
script de R que nadie ha ejecutado como parte de esta actualización — no es
Python y no lo cubre esta rama, pero tampoco queremos decir "no queda nada
sin revisar" y dejarlo fuera.

---

¿Dudas sobre algún cambio? Este documento intenta explicar el porqué, no
solo el qué — si algo no cuadra, dínoslo.
