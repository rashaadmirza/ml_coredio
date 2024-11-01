import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv('synthetic_lvedp_classification_dataset.csv')

# Features and target variable
X = data[['ECG', 'PPG', 'PTT', 'BP', 'Flow', 'DiameterVariation', 'CO', 'SVR', 'CVP', 'SV', 'EF']]
y = data['ElevatedLVEDP']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Function to predict elevated LVEDP status
def predict_elevated_lvedp(ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF):
    input_data = pd.DataFrame([[ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF]],
                              columns=['ECG', 'PPG', 'PTT', 'BP', 'Flow', 'DiameterVariation', 'CO', 'SVR', 'CVP', 'SV', 'EF'])
    prediction = rf_model.predict(input_data)
    return "Elevated" if prediction[0] == 1 else "Normal"

# Function to interactively input parameters
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
    
    result = predict_elevated_lvedp(ECG, PPG, PTT, BP, Flow, DiameterVariation, CO, SVR, CVP, SV, EF)
    print(f"LVEDP Status: {result}")

def main():
    input_parameters()

if __name__ == "__main__":
    main()
