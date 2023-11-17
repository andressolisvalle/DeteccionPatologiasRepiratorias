# Bibliotecas a emplear
import os
import cv2
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import matplotlib.pyplot as plt

# Definir las rutas donde estan las imagenes
train_folder= '/DataSet/Neumonia_Dataset/train'
val_folder = '/DataSet/Neumonia_Dataset/val'
test_folder = '/DataSet/Neumonia_Dataset/test'


# Red neuronal convolucional
cnn = models.Sequential()

# Capas convolucionales y de pooling
cnn.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
cnn.add(layers.MaxPool2D(pool_size = (2, 2)))
cnn.add(layers.Conv2D(32, (3, 3), activation="relu"))
cnn.add(layers.MaxPool2D(pool_size = (2, 2)))
cnn.add(layers.Conv2D(32, (3, 3), activation="relu"))
cnn.add(layers.MaxPool2D(pool_size = (2, 2)))
cnn.add(layers.Flatten())

# Capas densamente conectadas
cnn.add(layers.Dense(activation = 'relu', units = 128))
cnn.add(layers.Dense(activation = 'sigmoid', units = 1))

# Compilar el modelo neuronal
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Detalle de la red neuronal convolucional
cnn.summary()

# Preprocesamiento de las imagenes
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Normalización de imagenes
test_datagen = ImageDataGenerator(rescale = 1./255)

# Generación de los conjuntos de entrenamiento, validación y prrueba
training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (150, 150),
                                                 batch_size = 20,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(val_folder,
                                                        target_size =(150, 150),
                                                        batch_size = 20,
                                                        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_folder,
                                            target_size = (150, 150),
                                            batch_size = 20,
                                            class_mode = 'binary')

cnn_model = cnn.fit(training_set,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=50)

# Guardar el modelo
cnn.save('modelo_neumonia.h5')


# Predicción sobre una imagen de prueba
img_ori = cv2.imread('/content/Neumonia_Dataset/test/PNEUMONIA/person119_bacteria_566.jpeg')

img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
img = cv2.resize(img_ori, (150, 150), interpolation=cv2.INTER_CUBIC)
imagen_a_probar = np.expand_dims(img,(1,150, 150, 3))
predictions = cnn.predict(imagen_a_probar)
if(predictions == 0):
  print('Persona sin neumonia')
else:
  print('Persona con neumonia')
plt.imshow(img)
plt.show()