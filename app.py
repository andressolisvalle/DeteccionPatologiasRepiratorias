from io import BytesIO
#from tkinter import Image
from PIL import Image
import numpy as np
import skimage
from flask import Flask, render_template, request
from keras.models import load_model
from skimage.color import rgb2gray

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        response = request.files['xray_image']
        model = load_model('modelo_neumonia2.1.h5')
        class_names = ['NORMAL', 'NEUMONIA']

        img = Image.open(BytesIO(response.read()))
        np_img = np.array(img)
        bw_image = skimage.transform.resize(np_img, (150, 120, 3), mode='constant', anti_aliasing=True)
        bw_image = rgb2gray(bw_image)
        x = np.zeros((1, 150, 120))
        x[0] = np.array(bw_image)
        x_reshaped = x.reshape(len(x), 150, 120, 1)
        predictions = model.predict(x_reshaped)
    except Exception as e:
        return f"Error: {str(e)}"

    return render_template('resultado.html', predictions=predictions[0], class_names=class_names, zip=zip)



