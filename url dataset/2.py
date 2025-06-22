import pandas as pd
import re

# Load the dataset
df = pd.read_csv(r"C:\Users\omarf\OneDrive\Desktop\url dataset\malicious_phish.csv")

# Confirm column names
print("\n📋 Columns in your DataFrame:")
print(df.columns)

# Binary label conversion
df['type'] = df['type'].apply(lambda x: 'malicious' if x != 'benign' else 'benign')

print("\n✅ Label conversion complete:")
print(df['type'].value_counts())

# --- Feature extraction ---
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['num_slashes'] = url.count('/')
    features['num_digits'] = sum(char.isdigit() for char in url)
    features['num_letters'] = sum(char.isalpha() for char in url)
    return features

# 👇 This will break if 'url' column doesn't exist, so let's check
if 'url' not in df.columns:
    print("\n❌ ERROR: 'url' column not found in your dataset.")
else:
    features_df = df['url'].apply(extract_features).apply(pd.Series)
    final_df = pd.concat([features_df, df['type']], axis=1)

    print("\n✅ Feature extraction complete. Sample output:")
    print(final_df.head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = final_df.drop('type', axis=1)
y = final_df['type'].map({'benign': 0, 'malicious': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Prepare data
X = final_df.drop('type', axis=1)
y = final_df['type'].map({'benign': 0, 'malicious': 1})  # convert labels to 0/1

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Split into features and labels
X = final_df.drop('type', axis=1)
y = final_df['type'].map({'benign': 0, 'malicious': 1})

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
