import pandas as pd
import numpy as np

n_samples = 100
lvedp_threshold = 15  # Define the threshold for elevated LVEDP

# Generate synthetic data
data = {
    'ECG': np.random.normal(60, 5, n_samples),
    'PPG': np.random.uniform(0.5, 1.5, n_samples),
    'PTT': np.random.normal(200, 20, n_samples),
    'BP': np.random.uniform(110, 140, n_samples),
    'Flow': np.random.uniform(3.5, 5.5, n_samples),
    'DiameterVariation': np.random.uniform(1.5, 2.5, n_samples),
    'CO': np.random.uniform(4.0, 7.0, n_samples),
    'SVR': np.random.uniform(800, 1600, n_samples),
    'CVP': np.random.uniform(2, 8, n_samples),
    'SV': np.random.uniform(50, 100, n_samples),
    'EF': np.random.uniform(50, 70, n_samples),
    'LVEDP': np.random.uniform(5, 20, n_samples)  # Original LVEDP values
}

df = pd.DataFrame(data)

# Add a binary label for elevated LVEDP
df['ElevatedLVEDP'] = (df['LVEDP'] > lvedp_threshold).astype(int)

# Save to CSV
df.to_csv("synthetic_lvedp_classification_dataset.csv", index=False)
print("Dataset created with classification label and saved as 'synthetic_lvedp_classification_dataset.csv'")
