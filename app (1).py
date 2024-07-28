
import os
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Directory where templates and static files are located
template_dir = "D:/project run"

app = Flask(__name__, template_folder=template_dir)
model = load_model('D:/project run/coral_classification_model.h5')

UPLOAD_FOLDER = 'D:/project run'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = (224, 224)

def model_predict(img_path):
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        return np.argmax(prediction, axis=1)[0]
    except Exception as e:
        print(f"Error in model_predict: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image part in the request.")
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        print("No selected file.")
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"Saved file: {filepath}")
        
        prediction = model_predict(filepath)
        if prediction is None:
            print("Error in prediction.")
            return "Error in prediction."
        
        print(f"Prediction result: {prediction}")
        return render_template('result.html', filename=file.filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

