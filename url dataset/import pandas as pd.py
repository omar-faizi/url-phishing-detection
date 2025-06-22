import pandas as pd

df = pd.read_csv(r"C:\Users\omarf\OneDrive\Desktop\url dataset\malicious_phish.csv")

print(df.head())

# Use 'type' column instead of 'label'
print("\nType counts:")
print(df['type'].value_counts())

# Convert to binary labels
df['type'] = df['type'].apply(lambda x: 'malicious' if x != 'benign' else 'benign')

print("\nBinary type counts:")
print(df['type'].value_counts())
