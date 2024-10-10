import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# 1. Prepare the data for modeling
# Load the data
df = pd.read_csv('retaildataset/sample_customer_data_for_exam.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Prepare features (X) and target variable (y)
X = df.drop(['id', 'purchase_amount'], axis=1)
y = df['purchase_amount']

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Evaluate the model performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 5. Identify top three features
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
top_features = feature_importance.sort_values('importance', ascending=False).head(3)

print("\nTop 3 features contributing to purchase amount prediction:")
print(top_features)

# Optional: Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
feature_importance.sort_values('importance', ascending=True).plot(x='feature', y='importance', kind='barh')
plt.title('Feature Importance for Purchase Amount Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# i. Prepare the data for classification
# Load the data
df = pd.read_csv('retailsales/sample_customer_data_for_exam.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Prepare features (X) and target variable (y)
X = df.drop(['id', 'promotion_usage'], axis=1)
y = df['promotion_usage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ii. Implement Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# iii. Evaluate the model
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# iv. Create and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# v. Identify top three factors
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_classifier.feature_importances_})
top_features = feature_importance.sort_values('importance', ascending=False).head(3)

print("\nTop 3 factors contributing to promotion usage prediction:")
print(top_features)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.sort_values('importance', ascending=True).plot(x='feature', y='importance', kind='barh')
plt.title('Feature Importance for Promotion Usage Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()