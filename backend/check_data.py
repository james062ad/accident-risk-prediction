import pandas as pd

def check_dataset():
    print("Checking dataset...")
    df = pd.read_csv('processed_accident_data.csv')
    
    print("\nDataset Info:")
    print(f"Number of records: {len(df)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nSample data:")
    print(df.head())

if __name__ == "__main__":
    check_dataset() 