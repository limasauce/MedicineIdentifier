import os
import cv2
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pytesseract
from werkzeug.utils import secure_filename

app = Flask(__name__)

upload_folder = 'static/uploads'
app.config['upload_folder'] = upload_folder

allowed_extensions = {'png', 'jpg', 'jpeg'}

def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def detect_imprint(img_path):
    image = cv2.imread(img_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray

def detect_color(img_path):
    # Load image
    image = cv2.imread(img_path)
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    colors = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255]),
        'blue': ([100, 100, 100], [140, 255, 255]),
        'pink': ([140, 100, 100], [170, 255, 255])
    }

    detected_colors = []
    for color, (lower, upper) in colors.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        if cv2.countNonZero(mask) > 0:
            detected_colors.append(color)

    return detected_colors

#def detect_input():
    print("Please enter the image you wish to enter:")
    img_path = input()
    return img_path

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_files(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['upload_folder'], filename)
            file.save(file_path)  # Save the uploaded file
            
            # Process the file
            colors_found = detect_color(file_path)
            image_gray = detect_imprint(file_path)

            # Save the grayscale image
            grayscale_path = os.path.join(app.config['upload_folder'], 'gray_' + filename)
            cv2.imwrite(grayscale_path, image_gray)

            return render_template('results.html', colors=colors_found, img_file=filename, gray_img_file='gray_' + filename)
    
    return render_template('index.html')

#img_path = detect_input()
#colors_found = detect_color(img_path)
#image_gray = detect_imprint(img_path)

#print("Detected Colors:", colors_found)
#cv2.imshow('Grayscale Image', image_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
