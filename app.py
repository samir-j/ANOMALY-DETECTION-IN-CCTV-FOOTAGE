from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from detect_anomaly import run_anomaly_detection
from flask import send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')

        file.save(input_path)
        run_anomaly_detection(input_path, output_path)

        return jsonify({'output_url': '/static/output.mp4'})
    return "Upload failed", 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
