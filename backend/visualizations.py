import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Read the data
df = pd.read_csv('processed_accident_data.csv')

# Show the columns
print("Available columns in the dataset:")
print(df.columns.tolist())

# Show first few rows
print("\nFirst few rows of the data:")
print(df.head())

# Use a basic matplotlib style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Region Distribution
sns.countplot(data=df, x='Region', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Accident Distribution by Region', pad=15)
axes[0,0].set_xlabel('Region')
axes[0,0].set_ylabel('Number of Accidents')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Risk Level Distribution
sns.countplot(data=df, x='accident_risk_level', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Accident Risk Level Distribution', pad=15)
axes[0,1].set_xlabel('Risk Level')
axes[0,1].set_ylabel('Number of Accidents')
axes[0,1].grid(True, alpha=0.3)

# 3. Weather Conditions Distribution
weather_counts = df['Weather Condition'].value_counts().head(10)
weather_counts.plot(kind='bar', ax=axes[1,0], color='salmon')
axes[1,0].set_title('Top Weather Conditions During Accidents', pad=15)
axes[1,0].set_xlabel('Weather Condition')
axes[1,0].set_ylabel('Number of Accidents')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Time of Day Distribution
time_counts = df['Time of Day'].value_counts()
time_counts.plot(kind='bar', ax=axes[1,1], color='purple')
axes[1,1].set_title('Accidents by Time of Day', pad=15)
axes[1,1].set_xlabel('Time of Day')
axes[1,1].set_ylabel('Number of Accidents')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plots
plt.savefig('accident_analysis_visualizations.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Road Type Distribution
plt.figure(figsize=(10, 6))
road_type_counts = df['Road Type'].value_counts()
plt.barh(range(len(road_type_counts)), road_type_counts.values, color='lightblue')
plt.yticks(range(len(road_type_counts)), road_type_counts.index)
plt.title('Road Types with Most Accidents', pad=15)
plt.xlabel('Number of Accidents')
plt.ylabel('Road Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('road_type_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Correlation Heatmap for Numeric Variables
plt.figure(figsize=(10, 8))
numeric_cols = ['Speed Limit', 'Number of Vehicles']
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation: Speed Limit vs Number of Vehicles', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

def create_advanced_analysis(df):
    """Create advanced visualizations for feature analysis"""
    plt.figure(figsize=(20, 15))
    
    # 1. Risk by Speed and Weather with Road Type
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        data=df,
        x='Speed Limit',
        y='weather_severity',
        hue='accident_risk_level',
        size='road_severity',
        style='Road Type',
        alpha=0.6
    )
    plt.title('Risk Level by Speed, Weather and Road Type')
    
    # 2. Time Risk Patterns
    plt.subplot(2, 2, 2)
    time_risk_matrix = pd.pivot_table(
        df,
        values='combined_risk_v2',
        index='Time of Day',
        columns='Weather Condition',
        aggfunc='mean'
    )
    sns.heatmap(time_risk_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Combined Risk by Time and Weather')
    
    # 3. Regional Risk Analysis
    plt.subplot(2, 2, 3)
    sns.boxplot(
        data=df,
        x='Region',
        y='combined_risk_v2',
        hue='accident_risk_level'
    )
    plt.xticks(rotation=45)
    plt.title('Risk Distribution by Region')
    
    # 4. Critical Conditions Impact
    plt.subplot(2, 2, 4)
    critical_impact = df.groupby('critical_condition')['accident_risk_level'].value_counts(normalize=True)
    critical_impact = critical_impact.unstack()
    critical_impact.plot(kind='bar', stacked=True)
    plt.title('Risk Distribution in Critical vs Normal Conditions')
    
    plt.tight_layout()
    plt.savefig('advanced_risk_analysis.png')
    plt.close()

def create_severity_analysis(df):
    plt.figure(figsize=(20, 15))
    
    # Speed vs Severity
    plt.subplot(2, 2, 1)
    sns.boxplot(
        data=df,
        x='impact_severity',
        y='Speed Limit',
        hue='Road Type'
    )
    plt.title('Speed Distribution by Impact Severity')
    
    # Weather Impact on Severity
    plt.subplot(2, 2, 2)
    severity_weather = pd.crosstab(
        df['Weather Condition'],
        df['impact_severity']
    )
    sns.heatmap(severity_weather, annot=True, cmap='YlOrRd')
    plt.title('Weather Conditions vs Crash Severity')

def plot_test_results(results):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Risk Level Distribution
    risk_counts = pd.Series([r['risk_level'] for r in results]).value_counts()
    sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax1)
    ax1.set_title('Distribution of Predicted Risk Levels')
    ax1.set_xlabel('Risk Level')
    ax1.set_ylabel('Count')
    
    # 2. Confidence by Risk Level
    confidences = [(r['risk_level'], float(r['confidence'].strip('%'))/100) 
                  for r in results]
    conf_df = pd.DataFrame(confidences, columns=['risk_level', 'confidence'])
    sns.boxplot(x='risk_level', y='confidence', data=conf_df, ax=ax2)
    ax2.set_title('Prediction Confidence by Risk Level')
    ax2.set_xlabel('Risk Level')
    ax2.set_ylabel('Confidence')
    
    plt.tight_layout()
    plt.savefig('test_results_analysis.png')
    plt.close()

def analyze_results(results):
    print("\nAnalysis Summary:")
    print("-" * 50)
    
    # Risk level distribution
    risk_counts = pd.Series([r['risk_level'] for r in results]).value_counts()
    print("\nRisk Level Distribution:")
    for level, count in risk_counts.items():
        print(f"Risk Level {level}: {count} predictions")
    
    # Average confidence by risk level
    confidences = [(r['risk_level'], float(r['confidence'].strip('%'))/100) 
                  for r in results]
    conf_df = pd.DataFrame(confidences, columns=['risk_level', 'confidence'])
    print("\nAverage Confidence by Risk Level:")
    print(conf_df.groupby('risk_level')['confidence'].mean())

def create_policy_visualizations(df):
    # 1. Risk Distribution by Condition
    plt.figure(figsize=(20, 15))
    
    # Road Type vs Time Risk
    plt.subplot(2, 2, 1)
    sns.heatmap(
        df.pivot_table(
            values='risk_level',
            index='Road Type',
            columns='Time of Day',
            aggfunc='mean'
        ),
        annot=True,
        cmap='YlOrRd'
    )
    plt.title('Risk Levels by Road Type and Time')
    # Useful for: Targeting high-risk time periods on specific roads
    
    # Weather Impact Analysis
    plt.subplot(2, 2, 2)
    sns.boxplot(
        data=df,
        x='Weather Condition',
        y='risk_level',
        hue='Road Type'
    )
    plt.title('Weather Impact on Different Road Types')
    plt.xticks(rotation=45)
    # Useful for: Weather response planning
    
    # Speed Risk Analysis
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=df,
        x='Speed Limit',
        y='Number of Vehicles',
        hue='risk_level',
        size='accident_severity',
        alpha=0.6
    )
    plt.title('Risk by Speed and Traffic Density')
    # Useful for: Speed limit policy decisions
    
    # Regional Risk Patterns
    plt.subplot(2, 2, 4)
    sns.barplot(
        data=df,
        x='Region',
        y='risk_level',
        hue='Time of Day'
    )
    plt.title('Regional Risk Patterns')
    plt.xticks(rotation=45)
    # Useful for: Regional resource allocation
    
    plt.tight_layout()
    plt.savefig('policy_insights.png')
    plt.close()

def generate_all_visualizations(df):
    print("Generating policy insights visualizations...")
    
    # 1. Basic Analysis Visualizations
    create_basic_analysis(df)  # Creates 'accident_analysis_visualizations.png'
    print("✓ Basic analysis visualizations saved")
    
    # 2. Road Type Analysis
    create_road_analysis(df)   # Creates 'road_type_distribution.png'
    print("✓ Road type analysis saved")
    
    # 3. Advanced Risk Analysis
    create_advanced_analysis(df)  # Creates 'advanced_risk_analysis.png'
    print("✓ Advanced risk analysis saved")
    
    # 4. Policy Insights
    create_policy_visualizations(df)  # Creates 'policy_insights.png'
    print("✓ Policy insights saved")
    
    print("\nAll visualizations generated successfully!")

# Add to test_predictions.py:
if __name__ == "__main__":
    results = run_tests()
    plot_test_results(results)
    analyze_results(results) 