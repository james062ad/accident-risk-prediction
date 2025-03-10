﻿from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from predict import predict_accident_risk

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = predict_accident_risk(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/check-config', methods=['GET'])
def check_config():
    files_to_check = [
        'app.py',
        'Classification.py',
        'check_data.py',
        'visualizations1.py',
        'visualizations.py',
        'processed_accident_data.csv'
    ]
    
    files = []
    for file_name in files_to_check:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        exists = os.path.exists(file_path)
        file_info = {
            'name': file_name,
            'exists': exists
        }
        if exists:
            file_info['size'] = os.path.getsize(file_path)
        files.append(file_info)
    
    return jsonify({'files': files})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
