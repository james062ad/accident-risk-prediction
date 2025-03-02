import joblib
import pandas as pd
from validation import validate_input

def prepare_prediction_features(df, encoders):
    """Prepare features the same way as training data"""
    df_encoded = df.copy()
    
    # 1. Speed and Vehicle Features
    df_encoded['speed_vehicle_ratio'] = df_encoded['Speed Limit'] / df_encoded['Number of Vehicles']
    df_encoded['speed_squared'] = df_encoded['Speed Limit'] ** 2
    df_encoded['vehicle_density'] = df_encoded['Number of Vehicles'] / df_encoded['Speed Limit']
    df_encoded['speed_risk'] = df_encoded['Speed Limit'].apply(lambda x: 1 if x > 60 else (0.7 if x > 40 else 0.3))
    
    # 2. Time Features
    df_encoded['is_rush_hour'] = df_encoded['Time of Day'].isin(['Morning']).astype(int)
    df_encoded['is_night'] = df_encoded['Time of Day'].isin(['Night']).astype(int)
    
    # 3. Weather Features
    weather_severity = {
        'Clear': 0, 'Rainy': 0.6, 'Snowy': 1.0
    }
    df_encoded['weather_severity'] = df_encoded['Weather Condition'].map(weather_severity)
    df_encoded['weather_speed_interaction'] = df_encoded['weather_severity'] * df_encoded['speed_risk']
    
    # 4. Add missing features
    df_encoded['high_risk_combo'] = (
        (df_encoded['Speed Limit'] > 55) & 
        (df_encoded['weather_severity'] > 0.7) & 
        (df_encoded['is_rush_hour'] == 1)
    ).astype(int)
    
    df_encoded['night_weather_risk'] = (
        df_encoded['is_night'] * 
        df_encoded['weather_severity']
    )
    
    # 5. Time Risk Features
    df_encoded['time_risk'] = df_encoded.apply(
        lambda x: 1 if x['is_night'] else (0.8 if x['is_rush_hour'] else 0.4), 
        axis=1
    )
    
    # 6. Road Features
    road_severity = {
        'Residential': 0.3,
        'Urban Road': 0.6,
        'Highway': 1.0
    }
    df_encoded['road_severity'] = df_encoded['Road Type'].map(road_severity)
    
    # 7. Combined Risk Features
    df_encoded['critical_condition'] = (
        (df_encoded['weather_severity'] > 0.6) & 
        (df_encoded['speed_risk'] > 0.7) & 
        (df_encoded['road_severity'] > 0.6)
    ).astype(int)
    
    df_encoded['region_time_risk'] = 0.5  # Default value for prediction
    
    df_encoded['combined_risk_v2'] = (
        df_encoded['speed_risk'] * 0.25 +
        df_encoded['weather_severity'] * 0.20 +
        df_encoded['road_severity'] * 0.15 +
        df_encoded['time_risk'] * 0.15 +
        df_encoded['region_time_risk'] * 0.15 +
        df_encoded['critical_condition'] * 0.10
    )
    
    # 8. Additional Risk Features
    df_encoded['collision_risk'] = (
        df_encoded['Speed Limit'] * 0.4 +
        df_encoded['Number of Vehicles'] * 0.3 +
        df_encoded['road_severity'] * 0.3
    )
    
    df_encoded['severe_conditions'] = (
        (df_encoded['weather_severity'] > 0.7) & 
        (df_encoded['Speed Limit'] > 50) & 
        (df_encoded['Number of Vehicles'] >= 3)
    ).astype(int)
    
    # 9. Encode categorical variables
    for col in ['Region', 'Road Type', 'Time of Day', 'Weather Condition']:
        df_encoded[col] = encoders[col].transform(df_encoded[col])
    
    return df_encoded[[
        'Region', 'Road Type', 'Time of Day', 'Weather Condition',
        'Speed Limit', 'Number of Vehicles', 'speed_vehicle_ratio',
        'speed_squared', 'vehicle_density', 'speed_risk', 'is_rush_hour',
        'is_night', 'time_risk', 'weather_severity', 'weather_speed_interaction',
        'road_severity', 'critical_condition', 'region_time_risk', 'combined_risk_v2',
        'collision_risk', 'severe_conditions', 'high_risk_combo', 'night_weather_risk'
    ]]

def predict_accident_risk(data):
    # Validate input
    errors = validate_input(data)
    if errors:
        raise ValueError("\n".join(errors))
        
    # Load saved model and encoders
    model = joblib.load('accident_risk_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    
    # Format input data
    df = pd.DataFrame([data])
    
    # Prepare features
    df_encoded = prepare_prediction_features(df, encoders)
    
    # Make prediction
    prediction = model.predict(df_encoded)[0]
    probabilities = model.predict_proba(df_encoded)[0]
    
    return {
        'risk_level': prediction,
        'confidence': f"{max(probabilities):.2%}",
        'probabilities': {
            'low_risk': f"{probabilities[0]:.2%}",
            'medium_risk': f"{probabilities[1]:.2%}",
            'high_risk': f"{probabilities[2]:.2%}"
        }
    }

# Get risk factors
risk_analysis = {
    'predicted_risk': result['risk_level'],
    'confidence': result['confidence'],
    'key_factors': {
        'speed_risk': float(result['probabilities']['low_risk'].strip('%')) / 100,
        'weather_severity': float(result['probabilities']['medium_risk'].strip('%')) / 100,
        'road_severity': float(result['probabilities']['high_risk'].strip('%')) / 100,
        'combined_risk': float(result['probabilities']['medium_risk'].strip('%')) / 100
    }
}

print("\nKey Risk Factors:")
for factor, value in risk_analysis['key_factors'].items():
    print(f"{factor}: {value:.2f}") 