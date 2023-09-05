import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

model = load_model(r"C:\Users\vignesh\OneDrive\Desktop\Automated Weather Classification using Transfer Learning\project\wcv.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/input')
def input1():
    return render_template("input.html")

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224, 3))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        prediction = np.argmax(model.predict(img_data), axis=1)

        index = ['alien_test', 'cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
        result = index[prediction[0]]

        return render_template('output.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=False)
