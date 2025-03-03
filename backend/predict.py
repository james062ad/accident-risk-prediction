import joblib
import pandas as pd
from validation import validate_input

def predict_accident_risk(data):
    # Validate input
    errors = validate_input(data)
    if errors:
        raise ValueError('\n'.join(errors))
        
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # For now, return mock prediction
    return {
        'risk_level': 'Medium',
        'confidence': '75%',
        'probabilities': {
            'low_risk': '25%',
            'medium_risk': '75%',
            'high_risk': '0%'
        }
    }
