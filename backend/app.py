from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_accident_risk
import pandas as pd
from visualizations1 import create_policy_visualizations
from test_scenarios import test_cases

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'Region', 'Road Type', 'Time of Day',
            'Weather Condition', 'Speed Limit', 'Number of Vehicles'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        result = predict_accident_risk(data)
        
        return jsonify({
            'input': data,
            'prediction': {
                'risk_level': result['risk_level'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scenarios', methods=['GET'])
def get_scenarios():
    return jsonify({
        'scenarios': test_cases
    })

@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    try:
        df = pd.read_csv('processed_accident_data.csv')
        summary = {
            'total_records': len(df),
            'regions': df['Region'].unique().tolist(),
            'road_types': df['Road Type'].unique().tolist(),
            'weather_conditions': df['Weather Condition'].unique().tolist()
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 