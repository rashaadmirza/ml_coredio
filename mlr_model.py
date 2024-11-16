import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('mlr_data.csv')
X = data[['PPG', 'ECG', 'EF', 'CI', 'CO', 'SBP', 'DBP']]
y = data['LVEDP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Prediction function for LVEDP
def predict_lvedp(PPG, ECG, EF, CI, CO, SBP, DBP):
    input_df = pd.DataFrame([[PPG, ECG, EF, CI, CO, SBP, DBP]],
                            columns=['PPG', 'ECG', 'EF', 'CI', 'CO', 'SBP', 'DBP'])
    estimated_lvedp = model.predict(input_df)
    return estimated_lvedp[0]

# Input parameters function
def input_parameters():
    print("Enter the following cardiovascular parameters:")
    PPG = float(input("PPG: "))
    ECG = float(input("ECG: "))
    EF = float(input("Ejection Fraction (EF): "))
    CI = float(input("Cardiac Index (CI): "))
    CO = float(input("Cardiac Output (CO): "))
    SBP = float(input("Systolic Blood Pressure (SBP): "))
    DBP = float(input("Diastolic Blood Pressure (DBP): "))
    
    estimated_lvedp = predict_lvedp(PPG, ECG, EF, CI, CO, SBP, DBP)
    print(f"Estimated LVEDP: {estimated_lvedp:.2f} mmHg")

# Main function
def main():
    input_parameters()

if __name__ == "__main__":
    main()
