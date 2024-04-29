import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("dataset.csv")

# Data Preprocessing
data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features (X) and target (y)
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Prediction (for new data)
new_data = pd.DataFrame({
    'Age': [30],
    'BusinessTravel': [1],  # Assuming 'Travel_Rarely' corresponds to 1 in the encoded values
    # Add other features accordingly
})
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column] = label_encoders[column].transform(new_data[column])

# Ensure that new_data has the same set of features as X_train
missing_features = set(X_train.columns) - set(new_data.columns)
for feature in missing_features:
    new_data[feature] = 0  # Fill missing features with default values (0 or any appropriate default value)

# Reorder columns to match the order used during model training
new_data = new_data[X_train.columns]

# Make prediction
prediction = model.predict(new_data)
print("Predicted Attrition:", prediction)
