def validate_input(data):
    errors = []
    
    # Check required fields
    required_fields = [
        'Region', 'Road Type', 'Time of Day',
        'Weather Condition', 'Speed Limit', 'Number of Vehicles'
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f'Missing field: {field}')
    
    return errors
