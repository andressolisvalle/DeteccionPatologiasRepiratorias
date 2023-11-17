import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.debug import console

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    modelo_cargado = load_model('modelo_neumonia.h5')

    # Obtener la imagen subida desde la solicitud
    imagen_subida = request.files['xray_image']

    # Procesar la imagen para que coincida con las dimensiones esperadas por el modelo
    img = cv2.imdecode(np.fromstring(imagen_subida.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (150, 150))
    img = np.array(img, axis=0)
    img = img / 255.0  # Normalizar la imagen

    resultado = modelo_cargado.predict(img)

    resultado = "Persona con neumonía" if resultado[0][0] > 0.5 else "Persona sin neumonía"

    return render_template('resultado.html', resultado=resultado)



