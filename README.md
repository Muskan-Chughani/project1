# project1
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the CSV file
file_path = '/content/drive/MyDrive/internship/loan_approval_dataset.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())
# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)
# Check the data types of each column
print(df.dtypes)
# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
# Fill missing values for numerical columns with mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
# Fill missing values for categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    # Verify that there are no more missing values
missing_values_after = df.isnull().sum()
print("Missing values in each column after handling:")
print(missing_values_after)
# Convert data types if necessary (example: converting 'loan_amount' to float)
# df['loan_amount'] = df['loan_amount'].astype(float)

# Print data types of all columns
print(df.dtypes)
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numerical columns for correlation matrix
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
z_scores = stats.zscore(numerical_df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_no_outliers = df[filtered_entries]
print("Columns in df_no_outliers:", df_no_outliers.columns)

# Assuming 'Loan_Status' is the column to be predicted (adjust this based on your dataset)
if ' loan_status' in df_no_outliers.columns:
    # Separate features (X) and target (y)
    X = df_no_outliers.drop(' loan_status', axis=1)
    y = df_no_outliers[' loan_status']
else:
    print("Column 'Loan_Status' not found in df_no_outliers. Adjust your code accordingly.")
# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Correct data types if necessary
# Example: data['column_name'] = data['column_name'].astype('desired_dtype')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/drive/MyDrive/internship/loan_approval_dataset.csv'
data = pd.read_csv(file_path)

# Print column names to verify
print(data.columns)

# Assuming the dataset has columns including features and a target variable (like 'loan_status')
# Adjust based on the actual column names in your dataset
X = data.drop(' loan_status', axis=1)  # Features
y = data[' loan_status']  # Target variable

# Identify categorical columns
categorical_columns = [' education', ' self_employed']
numerical_columns = X.columns.difference(categorical_columns)

# Preprocessing pipelines for numerical and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')  # Drop first to avoid multicollinearity

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])

# Create and train the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier(n_neighbors=5))])

# Splitting the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=' Approved')
recall = recall_score(y_test, y_pred, pos_label=' Approved')
f1 = f1_score(y_test, y_pred, pos_label=' Approved')
roc_auc = roc_auc_score(pd.get_dummies(y_test, drop_first=True), pd.get_dummies(y_pred, drop_first=True))

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC Score: {roc_auc:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# ROC Curve
fpr, tpr, thresholds = roc_curve(pd.get_dummies(y_test, drop_first=True), pd.get_dummies(y_pred, drop_first=True))
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()









