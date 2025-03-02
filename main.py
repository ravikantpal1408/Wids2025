import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Update the file paths if necessary
socio_demo_path = os.path.expanduser('/Users/ravikantpal/Downloads/widsdatathon2025/train_tsv/socio_demo.csv')
targets_path = os.path.expanduser('/Users/ravikantpal/Downloads/widsdatathon2025/train_tsv/targets.csv')
fmri_path = os.path.expanduser('/Users/ravikantpal/Downloads/widsdatathon2025/train_tsv/fmri_connectome.csv')

# Check if the files exist
if not os.path.exists(socio_demo_path):
    raise FileNotFoundError(f"File not found: {socio_demo_path}")
if not os.path.exists(targets_path):
    raise FileNotFoundError(f"File not found: {targets_path}")
if not os.path.exists(fmri_path):
    raise FileNotFoundError(f"File not found: {fmri_path}")

# Load the socio-demographic, emotions, and parenting data
socio_demo_df = pd.read_csv(socio_demo_path)

# Load the targets
targets_df = pd.read_csv(targets_path)

# Load the functional MRI connectome matrices
fmri_df = pd.read_csv(fmri_path)

# Display the first few rows of each dataframe
print(socio_demo_df.head())
print(targets_df.head())
print(fmri_df.head())

# Identify categorical columns
categorical_columns = socio_demo_df.select_dtypes(include=['object']).columns

# Create dummy variables for categorical columns
socio_demo_df = pd.get_dummies(socio_demo_df, columns=categorical_columns)

# Display the first few rows of the processed dataframe
print(socio_demo_df.head())

# Combine the socio-demographic data with the functional connectome data
combined_df = pd.concat([socio_demo_df, fmri_df], axis=1)

# Add the target variables
combined_df = pd.concat([combined_df, targets_df], axis=1)

# Display the first few rows of the combined dataframe
print(combined_df.head())

# Split the data into features and target variables
X = combined_df.drop(columns=['ADHD_Outcome', 'Sex_F'])
y_adhd = combined_df['ADHD_Outcome']
y_sex = combined_df['Sex_F']

# Split the data into training and testing sets
X_train, X_test, y_adhd_train, y_adhd_test, y_sex_train, y_sex_test = train_test_split(X, y_adhd, y_sex, test_size=0.2, random_state=42)

# Train the model for ADHD prediction
adhd_model = RandomForestClassifier(random_state=42)
adhd_model.fit(X_train, y_adhd_train)

# Train the model for Sex prediction
sex_model = RandomForestClassifier(random_state=42)
sex_model.fit(X_train, y_sex_train)

# Make predictions
y_adhd_pred = adhd_model.predict(X_test)
y_sex_pred = sex_model.predict(X_test)

# Evaluate the models
adhd_accuracy = accuracy_score(y_adhd_test, y_adhd_pred)
sex_accuracy = accuracy_score(y_sex_test, y_sex_pred)

print(f'ADHD Prediction Accuracy: {adhd_accuracy}')
print(f'Sex Prediction Accuracy: {sex_accuracy}')

# Save the predictions and evaluation metrics to a CSV file
evaluation_df = pd.DataFrame({
    'y_adhd_test': y_adhd_test,
    'y_adhd_pred': y_adhd_pred,
    'y_sex_test': y_sex_test,
    'y_sex_pred': y_sex_pred
})

evaluation_df.to_csv('/Users/ravikantpal/Downloads/widsdatathon2025/evaluation_results.csv', index=False)

with open('/Users/ravikantpal/Downloads/widsdatathon2025/evaluation_metrics.txt', 'w') as f:
    f.write(f'ADHD Prediction Accuracy: {adhd_accuracy}\n')
    f.write(f'Sex Prediction Accuracy: {sex_accuracy}\n')