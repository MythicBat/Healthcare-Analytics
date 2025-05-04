# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Loading the dataset
df = pd.read_csv("GHW_HeartFailure_Readmission.csv")
print("Shape:", df.shape)
print(df.head())

# Checking for missing values
print(df.isnull().sum())

# Basic preprocessing
# Label encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)   # Ensuring that no NaNs for LabelEncoder
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Scale numeric features
scaler = StandardScaler()
X = df.drop('Readmission', axis=1)
y = df['Readmission']
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Preprocessing complete!")