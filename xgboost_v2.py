import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import matplotlib.pyplot as plt

# Load Excel file
excel_path = "clean_data (1) - Copy.xlsx"
df = pd.read_excel(excel_path)

print("Loading and cleaning data...")

# Clean the data
df = df.dropna(subset=['text ', 'label'])
df['text '] = df['text '].str.strip()
df = df[df['text '].str.len() > 0]

# Prepare data
X = df['text ']
y = df['label']

print("Converting text to features...")

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

print("Balancing data with SMOTE...")

# Balance data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Show class distribution after SMOTE
unique, counts = np.unique(y_resampled, return_counts=True)
print("Class distribution after SMOTE:", dict(zip(unique, counts)))

print("Splitting data into train, validation, and test sets...")

# Split data: 70% train, 20% test, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

print("Tuning hyperparameters with GridSearchCV...")

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

grid = GridSearchCV(estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False, eval_metric='mlogloss'),
                    param_grid=param_grid,
                    cv=3,
                    scoring='f1_macro')
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

print("Cross-validating model...")

# Cross-validation
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1_macro')
print("Cross-validation F1 Macro Score: {:.2f}".format(cv_scores.mean()))

print("Training final model...")

# Train the final model on the training set
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)

print("Calculating metrics...")

# Calculate metrics
precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# Create results table
data = []
for i in range(3):
    data.append([str(i), f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{fscore[i]:.2f}", support[i]])
data.append(["accuracy", "", "", f"{accuracy:.2f}", sum(support)])
data.append(["macro avg", 
             f"{np.mean(precision):.2f}", 
             f"{np.mean(recall):.2f}", 
             f"{np.mean(fscore):.2f}", 
             sum(support)])
data.append(["weighted avg",
             f"{np.average(precision, weights=support/support.sum()):.2f}",
             f"{np.average(recall, weights=support/support.sum()):.2f}",
             f"{np.average(fscore, weights=support/support.sum()):.2f}",
             sum(support)])

print("\nClassification Results:")
print("=" * 60)
print(tabulate(data, headers=["", "precision", "recall", "f1-score", "support"], tablefmt="grid", numalign="right"))
print("=" * 60)

print("Plotting confusion matrix...")
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion Matrix")
plt.show()

# Save detailed predictions to CSV
predictions_df = pd.DataFrame({
    'Text': df.loc[y_test.index, 'text '],
    'True_Label': y_test,
    'Predicted_Label': predictions
})
predictions_df.to_csv('classification_results.csv', index=False)
print("\nDetailed predictions have been saved to 'classification_results.csv'")
