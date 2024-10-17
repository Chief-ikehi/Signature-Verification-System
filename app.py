from flask import Flask, request, render_template, flash, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import pickle

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages
model = load_model('signature_verification_model2.h5')

# Directory to store genuine signatures
GENUINE_SIGNATURES_DIR = 'genuine_signatures'

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    image = image / 255.0
    image = image.reshape(1, 128, 128, 3)
    return image

def save_genuine_signature(person_id, image):
    person_dir = os.path.join(GENUINE_SIGNATURES_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)
    image_path = os.path.join(person_dir, 'genuine_signature.png')
    cv2.imwrite(image_path, image)
    # Optionally, save the features extracted by the model
    features = model.predict(preprocess_image(image))
    features_path = os.path.join(person_dir, 'features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

def load_genuine_signature(person_id):
    features_path = os.path.join(GENUINE_SIGNATURES_DIR, person_id, 'features.pkl')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            return pickle.load(f)
    return None

@app.route('/upload_genuine', methods=['GET', 'POST'])
def upload_genuine():
    if request.method == 'POST':
        person_id = request.form['person_id']
        file = request.files['file']
        if file and file.filename != '':
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            save_genuine_signature(person_id, image)
            flash('Genuine signature uploaded successfully.')
            return redirect(url_for('upload_genuine'))
        else:
            flash('No file selected or invalid file format.')
            return redirect(url_for('upload_genuine'))
    return render_template('upload_genuine.html')

@app.route('/verify_signature', methods=['GET', 'POST'])
def verify_signature():
    result = ''
    if request.method == 'POST':
        person_id = request.form['person_id']
        file = request.files['file']
        if file and file.filename != '':
            genuine_features = load_genuine_signature(person_id)
            if genuine_features is not None:
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                uploaded_features = model.predict(preprocess_image(image))
                similarity = np.dot(genuine_features, uploaded_features.T) / (
                    np.linalg.norm(genuine_features) * np.linalg.norm(uploaded_features))
                result = 'Genuine' if similarity > 0.5 else 'Forged'
            else:
                flash('No genuine signature found for this person.')
                return redirect(url_for('verify_signature'))
        else:
            flash('No file selected or invalid file format.')
            return redirect(url_for('verify_signature'))
    return render_template('verify_signature.html', result=result)

if __name__ == '__main__':
    app.run()