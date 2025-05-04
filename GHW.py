# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Loading the dataset
df = pd.read_csv("data/GHW_HeartFailure_Readmission.csv")
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
X = df.drop('Readmission_30Days', axis=1)
y = df['Readmission_30Days']
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Preprocessing complete!")

# --- Exploratory Data Analysis --- 

# Check class balance
sns.countplot(x='Readmission_30Days', data=df)
plt.title('Class Distribution: Readmitted (1) vs Not Readmitted (0)')
plt.show()

# Correlation heatmap (numerical features)
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Age distribution vs readmission
plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Age', hue='Readmission_30Days', bins=20, kde=True, element='step')
plt.title('Age Distribution by Readmission Status')
plt.show()

# Gender vs Readmission
sns.countplot(x='Gender', hue='Readmission_30Days', data=df)
plt.title('Readmission vs Gender')
plt.show()

# --- Model Building & Evaluation ---

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC: ", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Plot ROC Curve
RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
plt.title('Random Forest ROC Curve')
plt.show()
