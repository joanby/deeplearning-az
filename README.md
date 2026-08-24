# 🚀 Curso Completo de Deep Learning: ¡Domina las Redes Neuronales! 🧠

## Creado por [Kirill Eremenko](https://www.udemy.com/user/kirilleremenko/) y [Hadelin de Ponteves](https://www.udemy.com/user/hadelin-de-ponteves/)  
## Traducido al español por [Juan Gabriel Gomila Salas](https://www.udemy.com/user/juangabriel2)

**Última actualización**: Agosto de 2026

---

## 🆕 Esta es la rama `update-2026`

Estás en la versión del código **puesta al día y verificada en agosto de
2026**, con Python 3.13 y las librerías tal como están hoy (numpy 2,
scikit-learn 1.9, TensorFlow/Keras 3...). El curso se grabó en 2020, en
Spyder, sobre los ficheros `.py`; algunas librerías han cambiado de
comportamiento desde entonces, y aquí el código vuelve a ejecutarse de
principio a fin.

**Qué encontrarás — y qué no — en esta rama:**

- El código **Python** (`.py` y `.ipynb`), puesto al día y en la raíz del
  repositorio, organizado por partes del curso igual que siempre
  (`Part 1 - Artificial Neural Networks (ANN)`, `Part 2 - Convolutional
  Neural Networks (CNN)`, etc.).
- **No hay carpetas `original/` ni `updated/`.** Esta rama es plana: lo que
  ves en cada `Part X` es directamente el código que necesitas.
- **`original/` no ha desaparecido: vive en la rama
  [`master`](https://github.com/joanby/deeplearning-az/tree/master)**, tal
  como se grabó el vídeo. Si quieres ver el código exactamente como lo
  teclea el profesor en pantalla, es ahí donde está.
- **Los 7 notebooks de Colab también están aquí, arreglados y
  ejecutables** — tanto en local como en el propio Google Colab. Llevaban
  rotos desde 2023 por un problema del propio repositorio, no del Deep
  Learning; el detalle completo está en `CAMBIOS-2026.md`.
- `additional materials/` (las transparencias de teoría) se mantiene igual
  que en `master`.

**Qué ha cambiado exactamente y por qué:** todo está explicado, causa por
causa y con el porqué de cada librería, en
[`CAMBIOS-2026.md`](CAMBIOS-2026.md). Incluye también qué partes pueden
verse distintas al vídeo y qué no se ha llegado a ejecutar entero (el
entrenamiento completo de la CNN, el autocodificador y la rejilla del
`GridSearchCV`, los tres por coste de cómputo, no por estar rotos). Léelo
antes de reportar como error algo que ya está documentado ahí.

---

## 🐍 Cómo montar el entorno

El curso usa un entorno conda. Con
[Miniforge](https://github.com/conda-forge/miniforge) o Anaconda instalado:

```bash
conda env create -f deeplearningaz.yml
conda activate deeplearningaz
```

Esto te deja Python 3.13 con numpy, pandas, matplotlib, scikit-learn,
Jupyter, Spyder (el mismo editor del curso) y PyTorch, más TensorFlow y
`scikeras` (por pip). `scipy` también acaba instalado, como dependencia de
scikit-learn, aunque el `.yml` no lo pida por nombre. Las versiones exactas
con las que se ha verificado esta rama están en la tabla de
[`CAMBIOS-2026.md`](CAMBIOS-2026.md#versiones-con-las-que-se-ha-verificado).

---

## 🌟 ¿Por qué elegir el curso "Deep Learning de A a la Z"? 🌟

Aquí te presentamos **5 razones clave** por las que este curso es único y te ayudará a dominar el Deep Learning:

### 1️⃣ **Estructura sólida y clara**

El Deep Learning es un campo extenso y complejo, por lo que hemos diseñado una estructura que te guía paso a paso. Dividimos el curso en dos grandes volúmenes: **Deep Learning supervisado** y **Deep Learning no supervisado**, cubriendo tres algoritmos claves en cada uno. ¡Así tendrás una comprensión global y profunda del tema! 📘🔍

### 2️⃣ **¡Intuición primero!**

¿Algún curso te ha saturado con fórmulas y teorías sin explicar el "por qué"? Aquí, te ayudamos a entender la **intuición** detrás de cada algoritmo. Aprenderás no solo a codificar, sino a comprender de forma instintiva lo que estás haciendo. ¡La teoría y la práctica van de la mano! 💡💻

### 3️⃣ **Proyectos del mundo real 💼**

Nada de ejemplos aburridos o conjuntos de datos obsoletos. Trabajarás en desafíos reales como:

- **Redes Neuronales para predicción de pérdida de clientes (Customer Churn)**
- **Redes neuronales convolucionales para clasificación de imágenes**
- **Redes neuronales recurrentes para predicción de precios de acciones**
- ¡Y mucho más! Cada proyecto está diseñado para que puedas aplicar lo aprendido a problemas empresariales reales.

### 4️⃣ **Ejercicios prácticos desde cero 🖥️**

Empezaremos desde un script en blanco para que construyas el código línea por línea. Verás cómo se ensamblan las piezas y tendrás acceso a los códigos desde **GitHub** y **Google Drive**. Además, aprenderás a adaptar los algoritmos a tus propios proyectos y conjuntos de datos. ¡Estarás listo para aplicar lo aprendido en tu trabajo! 🔧⚙️

### 5️⃣ **Asistencia personalizada 💬**

En este curso no estarás solo. Si tienes alguna duda, nuestro equipo de expertos está disponible para ayudarte en un máximo de **48 horas**. ¡Tu éxito es nuestra prioridad! 🚀

---

## 🛠️ Las herramientas que usarás:

Aprenderás a usar **TensorFlow** y **PyTorch**, las dos principales librerías de Deep Learning. Además, exploraremos otras herramientas como **Keras**, **Scikit-learn**, **Numpy**, **Matplotlib**, y **Pandas** para manipular datos y crear modelos potentes y precisos. ¡Todo esto a través de Python! 🐍📊

---

## 🎯 ¿A quién va dirigido este curso?

Este curso es perfecto para:

- **Principiantes:** Si estás comenzando en el Deep Learning, aprenderás de manera estructurada sin ahogarte en complejidades matemáticas.
- **Expertos:** Si ya tienes experiencia, te desafiarás con técnicas avanzadas y aprenderás aplicaciones prácticas que quizás no hayas visto antes.

---

## 📝 Casos prácticos del mundo real

Dominar el Deep Learning no es solo entender la teoría, sino aplicarlo a **problemas reales**. Aquí tienes algunos de los casos prácticos que resolverás durante el curso:

1. **Predicción de pérdida de clientes (Customer Churn)**: Ayudarás a un banco a predecir si los clientes se irán o se quedarán.
2. **Reconocimiento de imágenes**: Construirás un modelo para identificar gatos y perros en imágenes (¡o lo que se te ocurra!).
3. **Predicción de precios de acciones**: Crearás un modelo para predecir los precios de las acciones de Google.
4. **Detección de fraude**: Aplicarás Deep Learning para detectar fraudes en solicitudes de tarjetas de crédito.
5. **Sistemas de recomendación**: Desarrollarás un sistema como el de **Netflix** para recomendar películas basadas en las preferencias de los usuarios.

---

**¡Conviértete en un experto en Deep Learning!** 💥 Domina las técnicas más avanzadas y aplica lo aprendido a problemas del mundo real. ¡Únete al curso [Deep Learning de A a la Z: Redes Neuronales al Completo](https://cursos.frogamesformacion.com/courses/deep-learning-az) y lleva tus habilidades al siguiente nivel! 🚀

---

**Juan Gabriel, Kirill y Hadelin te esperan.** 👨‍🏫👨‍💻
