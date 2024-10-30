import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('synthetic_lvedp_dataset.csv')
X = data[['ECG', 'PPG', 'PTT', 'BP', 'Flow', 'DiameterVariation', 'CO', 'SVR', 'CVP', 'SV', 'EF']]
y = data['LVEDP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

def predict_lvedp(ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF):

    input_df = pd.DataFrame([[ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF]],
                            columns=['ECG', 'PPG', 'PTT', 'BP', 'Flow', 'DiameterVariation', 'CO', 'SVR', 'CVP', 'SV', 'EF'])
    predicted_lvedp = model.predict(input_df)
    return predicted_lvedp[0]


def input_parameters():
    print("Enter the following cardiovascular parameters:")
    ECG = float(input("ECG: "))
    PPG = float(input("PPG: "))
    PTT = float(input("PTT: "))
    BP = float(input("BP: "))
    Flow = float(input("Flow: "))
    DiameterVariation = float(input("Diameter Variation: "))
    CO = float(input("Cardiac Output (CO): "))
    SVR = float(input("Systemic Vascular Resistance (SVR): "))
    CVP = float(input("Central Venous Pressure (CVP): "))
    SV = float(input("Stroke Volume (SV): "))
    EF = float(input("Ejection Fraction (EF): "))
    
    predicted_lvedp = predict_lvedp(ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF)
    print(f"Predicted LVEDP: {predicted_lvedp:.2f} mmHg")

def main():
    input_parameters()

if __name__ == "__main__":
    main()
