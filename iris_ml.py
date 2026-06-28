import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv('Iris.csv')

# The 'Id' column is just a row number, so we drop it
df = df.drop('Id', axis=1)

# 2. Separate Features (X) and Target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# 3. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize the Machine Learning Model
model = RandomForestClassifier(random_state=42)

# 5. Train the model using the training data
model.fit(X_train, y_train)

# 6. Give the model the test features and let it make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the predictions against the actual test labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))