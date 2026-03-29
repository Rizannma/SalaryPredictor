# Load dataset and show class distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import joblib

# Load dataset
df = pd.read_csv("job_salary_prediction_dataset.csv")

# Sample for faster execution  
print("Sampling data for faster execution...")
df = df.sample(n=50000, random_state=42)

print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head())

# Create binary salary_class based on salary median
df['salary_class'] = (df['salary'] >= df['salary'].median()).astype(int)


print("\n Class Distribution BEFORE Balancing ")
print(df['salary_class'].value_counts())

# Visualize before balancing
fig, ax = plt.subplots(figsize=(8, 5))
df['salary_class'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'coral'])
ax.set_title("Class Distribution Before Balancing")
ax.set_xlabel("Salary Class (0=Low, 1=High)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution_before.png")
print("Saved: class_distribution_before.png")
plt.close()

# Prepare features
X = df.drop(['salary', 'salary_class'], axis=1)
y = df['salary_class']

# Convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
print("\n Applying SMOTE ")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(" Class Distribution AFTER Balancing ")
print(pd.Series(y_train_bal).value_counts())

# Visualize after balancing
fig, ax = plt.subplots(figsize=(8, 5))
pd.Series(y_train_bal).value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'coral'])
ax.set_title("Class Distribution After SMOTE Balancing")
ax.set_xlabel("Salary Class (0=Low, 1=High)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution_after.png")
print("Saved: class_distribution_after.png")
plt.close()

# Train models
print("\n Training Models ")
lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)

lr.fit(X_train_bal, y_train_bal)
print("Logistic Regression trained.")
rf.fit(X_train_bal, y_train_bal)
print("Random Forest trained.")

# Evaluate models
def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n {name} Evaluation ")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    for key, val in metrics.items():
        if key != "Model":
            print(f"{key}: {val:.4f}")
    
    return metrics

lr_results = evaluate(lr, "Logistic Regression")
rf_results = evaluate(rf, "Random Forest")

# Compare models
print("\n Model Comparison ")
results_df = pd.DataFrame([lr_results, rf_results]).set_index("Model")
print(results_df)

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
results_df[metrics_to_plot].plot(kind='bar', ax=ax)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100, bbox_inches='tight')
print("\nSaved: model_comparison.png")
plt.close()

# Identify best model
best_model = results_df['F1 Score'].idxmax()
print(f"\n✓ Best-performing model (by F1 Score): {best_model}")
print(f"  F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")

#  FEATURE IMPORTANCE ANALYSIS 
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature names
feature_names = X.columns.tolist()

# Get feature importance from both models
print(f"\nLogistic Regression Coefficients (Feature Weights):")
lr_coef = np.abs(lr.coef_[0])  # Absolute values for importance
lr_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lr_coef
}).sort_values('Importance', ascending=False)
print(lr_importance_df.to_string(index=False))

print(f"\nRandom Forest Feature Importance:")
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(rf_importance_df.to_string(index=False))

# Visualize feature importance for both models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression importance
top_n = 15
lr_top = lr_importance_df.head(top_n)
axes[0].barh(lr_top['Feature'], lr_top['Importance'], color='steelblue')
axes[0].set_xlabel('Absolute Coefficient Value')
axes[0].set_title('Logistic Regression - Feature Importance\n(Top 15 Features)')
axes[0].invert_yaxis()

# Random Forest importance
rf_top = rf_importance_df.head(top_n)
axes[1].barh(rf_top['Feature'], rf_top['Importance'], color='forestgreen')
axes[1].set_xlabel('Importance Score')
axes[1].set_title('Random Forest - Feature Importance\n(Top 15 Features)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100, bbox_inches='tight')
print("\n✓ Saved: feature_importance.png")
plt.close()

# Save importance data
joblib.dump(lr_importance_df, "lr_importance.pkl")
joblib.dump(rf_importance_df, "rf_importance.pkl")
print("✓ Saved: Feature importance data")

# Save the best model
if best_model == "Logistic Regression":
    joblib.dump(lr, "best_model.pkl")
    print("\n✓ Saved: Logistic Regression model as best_model.pkl")
else:
    joblib.dump(rf, "best_model.pkl")
    print("\n✓ Saved: Random Forest model as best_model.pkl")

joblib.dump(X.columns, "features.pkl")
print("✓ Saved: Feature columns as features.pkl")