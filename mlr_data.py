import pandas as pd
import numpy as np

n_samples = 100

# Generates synthetic values within realistic physiological ranges for each parameter
data = {
    'ECG': np.random.normal(60, 5, n_samples),  # Average heart rate (ECG rate) around 60 bpm
    'PPG': np.random.uniform(0.5, 1.5, n_samples),  # Normalized PPG values
    'PTT': np.random.normal(200, 20, n_samples),  # Pulse transit time in ms
    'BP': np.random.uniform(110, 140, n_samples),  # Systolic blood pressure range
    'Flow': np.random.uniform(3.5, 5.5, n_samples),  # Blood flow rate in L/min
    'DiameterVariation': np.random.uniform(1.5, 2.5, n_samples),  # Arterial diameter change in mm
    'CO': np.random.uniform(4.0, 7.0, n_samples),  # Cardiac output in L/min
    'SVR': np.random.uniform(800, 1600, n_samples),  # Systemic vascular resistance in dynes·s/cm^5
    'CVP': np.random.uniform(2, 8, n_samples),  # Central venous pressure in mmHg
    'SV': np.random.uniform(50, 100, n_samples),  # Stroke volume in mL
    'EF': np.random.uniform(50, 70, n_samples),  # Ejection fraction in %
    'LVEDP': np.random.uniform(5, 20, n_samples)  # Target LVEDP values in mmHg
}

df = pd.DataFrame(data)
df.to_csv("synthetic_lvedp_dataset.csv", index=False)

print("Dataset created and saved as 'synthetic_lvedp_regression_dataset.csv'")
