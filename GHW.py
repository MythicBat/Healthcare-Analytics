# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", pd.Series(y_train_smote).value_counts())

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title('XGBoost ROC Curve')
plt.show()

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100,200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}

grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                        param_grid,
                        cv=3,
                        scoring='roc_auc',
                        verbose=1)
grid.fit(X_train_smote, y_train_smote)

print("Best Parameters:", grid.best_params_)
print("Best ROC AUC:", grid.best_score_)

best_model = grid.best_estimator_

# Get feature names
feature_names = df.drop('Readmission_30Days', axis=1).columns
importances = xgb_model.feature_importances_

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout
plt.show()

# Saving the Model
joblib.dump(xgb_model, "xgb_readmission_model.pkl")

