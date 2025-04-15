from flask import Flask, request, send_file, jsonify
from io import BytesIO
from yolo import run_yolo  # Modified to accept file-like object and return PIL.Image
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    start_time = time.time()
    processed_image = run_yolo(file)  # This should return a PIL image
    end_time = time.time()

    img_io = BytesIO()
    processed_image.save(img_io, 'JPEG')
    img_io.seek(0)

    response_time = end_time - start_time

    return send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='result.jpg',
        headers={
            "X-Response-Time": str(response_time)
        }
    )

@app.route('/')
def health_check():
    return "Cloud server is up ✅", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
