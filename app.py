from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os

app = Flask(__name__)

# Load CNN Model

model = load_model('model.h5')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home

@app.route('/')
def home():
    return render_template('index.html')

# Predict

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Create folders

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save image

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # Read image

    img = cv2.imread(filepath)

    if img is None:
        return "Invalid Image"

    # Detect Faces using cvlib

    faces, confidences = cv.detect_face(img)

    face_count = 0

    for face in faces:

        startX, startY, endX, endY = face

        detected_face = img[startY:endY, startX:endX]

        try:

            detected_face = cv2.resize(
                detected_face,
                (128,128)
            )

        except:
            continue

        detected_face = detected_face / 255.0

        detected_face = np.expand_dims(
            detected_face,
            axis=0
        )

        prediction = model.predict(
            detected_face,
            verbose=0
        )

        if prediction[0][0] > 0.5:

            label = "Human Face"

            color = (0,255,127)

            face_count += 1

        else:

            label = "Non Face"

            color = (0,0,255)

        # Draw Rectangle

        cv2.rectangle(
            img,
            (startX,startY),
            (endX,endY),
            color,
            2
        )

        # Label

        cv2.putText(
            img,
            label,
            (startX,startY-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Save Output

    output_path = os.path.join(
        OUTPUT_FOLDER,
        file.filename
    )

    cv2.imwrite(output_path, img)

    return render_template(
        'index.html',
        prediction=f'{face_count} Human Face(s) Detected',
        image=file.filename
    )

if __name__ == "__main__":
    app.run(debug=True)