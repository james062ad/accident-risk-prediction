import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_augmentation import create_advanced_features

def load_and_prepare_data():
    print("Loading dataset...")
    df = pd.read_csv('augmented_accident_data.csv', low_memory=False)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Add more sophisticated engineered features
    df['vehicles_speed_ratio'] = df['Number of Vehicles'] / df['Speed Limit']
    df['peak_hour'] = df['Time of Day'].isin(['Morning Rush Hour', 'Evening Rush Hour']).astype(int)
    df['is_highway'] = (df['Road Type'] == 'Highway').astype(int)
    df['is_bad_weather'] = df['Weather Condition'].isin(['Heavy Rain', 'Snow', 'Fog']).astype(int)
    
    # Create risk score
    df['combined_risk_score'] = calculate_combined_risk_score(df)
    
    return df

def calculate_combined_risk_score(df):
    """Calculate a comprehensive risk score"""
    risk_score = pd.Series(0, index=df.index)
    
    # Weather risk
    weather_risk = {
        'Heavy Rain': 3, 'Snow': 4, 'Fog': 4,
        'Light Rain': 2, 'Cloudy': 1, 'Clear': 0
    }
    risk_score += df['Weather Condition'].map(weather_risk).fillna(0)
    
    # Time risk
    time_risk = {
        'Morning Rush Hour': 3, 'Evening Rush Hour': 3,
        'Night': 2, 'Late Night': 2
    }
    risk_score += df['Time of Day'].map(time_risk).fillna(1)
    
    # Road type risk
    road_risk = {
        'Highway': 4, 'Major Road': 3,
        'Urban Road': 2, 'Residential': 1
    }
    risk_score += df['Road Type'].map(road_risk).fillna(2)
    
    # Vehicle and speed interaction
    risk_score += (df['Number of Vehicles'] * df['Speed Limit'] / 50).clip(0, 5)
    
    return risk_score

def prepare_features(df):
    """Prepare features with advanced engineering"""
    label_encoders = {}
    df_encoded = df.copy()
    
    # 1. Enhanced Speed and Vehicle Features
    df_encoded['speed_vehicle_ratio'] = df_encoded['Speed Limit'] / df_encoded['Number of Vehicles']
    df_encoded['speed_squared'] = df_encoded['Speed Limit'] ** 2
    df_encoded['vehicle_density'] = df_encoded['Number of Vehicles'] / df_encoded['Speed Limit']
    df_encoded['speed_risk'] = df_encoded['Speed Limit'].apply(lambda x: 1 if x > 60 else (0.7 if x > 40 else 0.3))
    
    # 2. Enhanced Time Features
    df_encoded['is_rush_hour'] = df_encoded['Time of Day'].isin(['Morning', 'Afternoon']).astype(int)
    df_encoded['is_night'] = df_encoded['Time of Day'].isin(['Night']).astype(int)
    df_encoded['time_risk'] = df_encoded.apply(
        lambda x: 1 if x['is_night'] and x['is_bad_weather'] else 
                 (0.8 if x['is_rush_hour'] else 0.4), 
        axis=1
    )
    
    # 3. Enhanced Weather Features
    weather_severity = {
        'Clear': 0, 'Cloudy': 0.2, 
        'Light Rain': 0.4, 'Rainy': 0.6,
        'Foggy': 0.8, 'Snowy': 1.0
    }
    df_encoded['weather_severity'] = df_encoded['Weather Condition'].map(weather_severity)
    df_encoded['weather_speed_interaction'] = df_encoded['weather_severity'] * df_encoded['speed_risk']
    
    # 4. Road Type Risk
    road_severity = {
        'Residential': 0.3,
        'Urban Road': 0.6,
        'Highway': 1.0
    }
    df_encoded['road_severity'] = df_encoded['Road Type'].map(road_severity)
    
    # 5. Multi-factor Risk Combinations
    df_encoded['critical_condition'] = (
        (df_encoded['weather_severity'] > 0.6) & 
        (df_encoded['speed_risk'] > 0.7) & 
        (df_encoded['road_severity'] > 0.6)
    ).astype(int)
    
    # 6. Regional Risk Patterns with Time
    region_time_risk = df_encoded.groupby(['Region', 'Time of Day'])['accident_risk_level'].apply(
        lambda x: (x == 'High').mean()
    ).reset_index()
    region_time_risk.columns = ['Region', 'Time of Day', 'region_time_risk']
    df_encoded = df_encoded.merge(region_time_risk, on=['Region', 'Time of Day'])
    
    # 7. Advanced Combined Risk Score
    df_encoded['combined_risk_v2'] = (
        df_encoded['speed_risk'] * 0.25 +
        df_encoded['weather_severity'] * 0.20 +
        df_encoded['road_severity'] * 0.15 +
        df_encoded['time_risk'] * 0.15 +
        df_encoded['region_time_risk'] * 0.15 +
        df_encoded['critical_condition'] * 0.10
    )
    
    # Add severity-specific features
    df_encoded['collision_risk'] = (
        df_encoded['Speed Limit'] * 0.4 +
        df_encoded['Number of Vehicles'] * 0.3 +
        df_encoded['road_severity'] * 0.3
    )
    
    # Multi-factor severity indicators
    df_encoded['severe_conditions'] = (
        (df_encoded['weather_severity'] > 0.7) & 
        (df_encoded['Speed Limit'] > 50) & 
        (df_encoded['Number of Vehicles'] >= 3)
    ).astype(int)
    
    # Add these features
    df_encoded['high_risk_combo'] = (
        (df_encoded['Speed Limit'] > 55) & 
        (df_encoded['weather_severity'] > 0.7) & 
        (df_encoded['is_rush_hour'] == 1)
    ).astype(int)
    
    df_encoded['night_weather_risk'] = (
        df_encoded['is_night'] * 
        df_encoded['weather_severity']
    )
    
    # Encode categorical variables
    categorical_cols = ['Region', 'Road Type', 'Time of Day', 'Weather Condition']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col])
    
    # Final feature set
    features = [
        'Region', 'Road Type', 'Time of Day', 'Weather Condition',
        'Speed Limit', 'Number of Vehicles', 'speed_vehicle_ratio',
        'speed_squared', 'vehicle_density', 'speed_risk', 'is_rush_hour',
        'is_night', 'time_risk', 'weather_severity', 'weather_speed_interaction',
        'road_severity', 'critical_condition', 'region_time_risk', 'combined_risk_v2',
        'collision_risk', 'severe_conditions', 'high_risk_combo', 'night_weather_risk'
    ]
    
    X = df_encoded[features]
    y = label_encoders['accident_risk_level'] = LabelEncoder()
    y_encoded = y.fit_transform(df_encoded['accident_risk_level'])
    
    return X, y_encoded, features, label_encoders

def create_and_train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                        random_state=42, stratify=y)
    
    # Optimized pipeline
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=5)),  # Fixed k_neighbors
        ('classifier', RandomForestClassifier(
            n_estimators=1500,  # Fixed optimal value
            max_depth=None,
            class_weight={0: 1.5, 1: 1.0, 2: 0.8},
            criterion='entropy',
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            warm_start=True  # Speed up training
        ))
    ])
    
    # Reduced parameter grid - focus on most important params
    param_grid = {
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__min_samples_split': [2, 3]
    }
    
    # Grid search with fewer CV folds
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=3,  # Reduced from 5
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    print("Training Optimized Model...")
    grid_search.fit(X_train, y_train)
    
    return grid_search, X_train, X_test, y_train, y_test

def evaluate_models(rf_model, X_test, y_test, feature_names):
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    
    # Print results
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, rf_pred))
    
    # Feature importance analysis
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.best_estimator_.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rf_importance, x='importance', y='feature')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
    plt.title('Random Forest Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Add policy-focused insights
    print("\nPolicy Insights and Recommendations:")
    print("------------------------------------")
    print("1. Infrastructure Factors:")
    infra_features = rf_importance[rf_importance['feature'].str.contains('Road|Speed|infrastructure')].head(3)
    for _, row in infra_features.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f} importance weight")
    
    print("\n2. Temporal Patterns:")
    time_importance = rf_importance[rf_importance['feature'].str.contains('Time|period')].iloc[0]
    print(f"   - {time_importance['feature']}: {time_importance['importance']:.3f} importance")
    
    print("\n3. Environmental Impact:")
    env_importance = rf_importance[rf_importance['feature'].str.contains('Weather|environmental')].iloc[0]
    print(f"   - {env_importance['feature']}: {env_importance['importance']:.3f} importance")
    
    # Generate policy recommendations
    generate_policy_recommendations(rf_importance, feature_names)
    
    return rf_importance

def generate_policy_recommendations(importance_df, features):
    """Generate evidence-based policy recommendations"""
    # Create recommendations using pandas concat instead of append
    recommendations = pd.DataFrame([
        {
            'Focus Area': 'Infrastructure',
            'Evidence': 'Analysis of road types and accident severity',
            'Policy Recommendation': 'Prioritize infrastructure improvements in high-risk areas'
        },
        {
            'Focus Area': 'Time-based Patterns',
            'Evidence': 'Temporal analysis of accident frequency',
            'Policy Recommendation': 'Develop targeted interventions for high-risk time periods'
        },
        {
            'Focus Area': 'Environmental Factors',
            'Evidence': 'Impact of weather conditions on accident severity',
            'Policy Recommendation': 'Enhance weather-related safety measures and public awareness'
        },
        {
            'Focus Area': 'Regional Analysis',
            'Evidence': 'Geographic distribution of accidents',
            'Policy Recommendation': 'Allocate resources based on regional risk profiles'
        }
    ])
    
    # Add insights from feature importance
    print("\nKey Findings for Policy Makers:")
    print("-------------------------------")
    for _, row in importance_df.head(5).iterrows():
        print(f"- {row['feature']}: {row['importance']:.3f} importance weight")
    
    # Save recommendations
    recommendations.to_csv('policy_recommendations.csv', index=False)
    return recommendations

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, feature_names, label_encoders = prepare_features(df)  # Get encoders
    
    # Train model
    rf_model, X_train, X_test, y_train, y_test = create_and_train_models(X, y)
    
    # Evaluate and visualize results
    feature_importance = evaluate_models(rf_model, X_test, y_test, feature_names)
    
    # Save results
    results = {
        'rf_best_params': rf_model.best_params_,
        'rf_cv_score': rf_model.best_score_
    }
    
    pd.DataFrame(results).to_csv('model_results.csv')
    feature_importance.to_csv('feature_importance.csv')
    
    # Save the trained model
    print("Saving trained model...")
    joblib.dump(rf_model, 'accident_risk_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')  # Save encoders for preprocessing

if __name__ == "__main__":
    main()
