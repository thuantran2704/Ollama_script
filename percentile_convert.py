import pandas as pd
import numpy as np

def normalize_to_percentiles(input_csv, output_csv):
    # Load the CSV
    df = pd.read_csv(input_csv)

    # For each numeric column, replace values with percentiles
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].rank(pct=True) * 100  # Convert to percentiles (0-100)

    # Save the new CSV
    df.to_csv(output_csv, index=False)
    print(f"Normalized CSV saved to {output_csv}")

# Example usage:
# normalize_to_percentiles("input.csv", "output_percentiles.csv")
