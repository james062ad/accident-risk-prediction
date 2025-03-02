import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_policy_visualizations(df):
    print("Creating policy visualizations...")
    
    # Convert risk levels to numbers
    risk_mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    
    # Create numeric risk level column
    df['risk_level_numeric'] = df['accident_risk_level'].map(risk_mapping)
    
    # 1. Risk Heatmap by Time and Road Type
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        df.pivot_table(
            values='risk_level_numeric',
            index='Road Type',
            columns='Time of Day',
            aggfunc='mean'
        ),
        annot=True,
        cmap='YlOrRd',
        fmt='.2f'
    )
    plt.title('Risk Levels by Road Type and Time of Day')
    plt.savefig('1_risk_heatmap.png')
    plt.close()
    print("✓ Risk heatmap saved")

    # 2. Weather Impact by Road Type
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=df,
        x='Weather Condition',
        y='risk_level_numeric',
        hue='Road Type'
    )
    plt.title('Weather Impact on Different Road Types')
    plt.xticks(rotation=45)
    plt.savefig('2_weather_impact.png')
    plt.close()
    print("✓ Weather impact analysis saved")

    # 3. Speed and Traffic Density Risk
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='Speed Limit',
        y='Number of Vehicles',
        hue='risk_level_numeric',
        alpha=0.6
    )
    plt.title('Risk by Speed and Traffic Density')
    plt.savefig('3_speed_traffic_risk.png')
    plt.close()
    print("✓ Speed and traffic analysis saved")

    # 4. Regional Risk Analysis
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x='Region',
        y='risk_level_numeric',
        hue='Time of Day'
    )
    plt.title('Regional Risk Patterns')
    plt.xticks(rotation=45)
    plt.savefig('4_regional_risk.png')
    plt.close()
    print("✓ Regional analysis saved")

def main():
    print("Loading accident data...")
    df = pd.read_csv('processed_accident_data.csv')
    
    # Add this check
    print("\nVerifying columns:")
    print(df.columns.tolist())
    
    if 'accident_risk_level' not in df.columns:
        print("Error: 'accident_risk_level' column not found!")
        return
        
    if 'accident_severity' not in df.columns:
        print("Warning: 'accident_severity' column not found!")
    
    create_policy_visualizations(df)
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main() 