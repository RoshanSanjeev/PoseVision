from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from PoseModule import process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    if video.filename == '':
        return "No selected file"

    filename = secure_filename(video.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    video.save(upload_path)
    feedback = process_video(upload_path, processed_path)

    return render_template('processed.html', filename=filename, feedback=feedback)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

