import os
import numpy as np
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_UPLOADS'] = 'static/uploads/'  # Thư mục lưu ảnh để hiển thị
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_UPLOADS'], exist_ok=True)

print("Loading model...")
model = load_model('best_model.keras')
print("Model loaded successfully")

train_dir = 'data1/train'
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Directory {train_dir} not found")
class_names = sorted(os.listdir(train_dir))
if len(class_names) != 104:
    raise ValueError(f"Expected 104 classes, but found {len(class_names)} in {train_dir}")
class_indices = {i: name for i, name in enumerate(class_names)}
print("Number of classes:", len(class_indices))

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot load image")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            static_file_path = os.path.join(app.config['STATIC_UPLOADS'], filename)
            file.save(file_path)
            
            try:
                img_batch = process_image(file_path)
                prediction = model.predict(img_batch)
                predicted_class = np.argmax(prediction)
                if predicted_class not in class_indices:
                    raise KeyError(f"Predicted class {predicted_class} not found in class_indices")
                predicted_label = class_indices[predicted_class]
                confidence = prediction[0][predicted_class]
                
                # Sao chép ảnh sang static/uploads để hiển thị
                shutil.copy(file_path, static_file_path)
                os.remove(file_path)
                
                return render_template('index.html', prediction=predicted_label, confidence=f"{confidence:.2f}", filename=filename)
            except Exception as e:
                flash(str(e))
                os.remove(file_path)
                return redirect(request.url)
    
    return render_template('index.html', prediction=None, confidence=None, filename=None)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)