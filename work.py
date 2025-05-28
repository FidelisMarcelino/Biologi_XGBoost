import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Training XGBoost model...")

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    random_state=42
)
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)

print("Calculating metrics...")

# Calculate metrics
precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# Create results table
data = []
# Add class metrics
for i in range(3):
    data.append([str(i), f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{fscore[i]:.2f}", support[i]])
# Add accuracy
data.append(["accuracy", "", "", f"{accuracy:.2f}", sum(support)])
# Add macro avg
data.append(["macro avg", 
             f"{np.mean(precision):.2f}", 
             f"{np.mean(recall):.2f}", 
             f"{np.mean(fscore):.2f}", 
             sum(support)])
# Add weighted avg
data.append(["weighted avg",
             f"{np.average(precision, weights=support/support.sum()):.2f}",
             f"{np.average(recall, weights=support/support.sum()):.2f}",
             f"{np.average(fscore, weights=support/support.sum()):.2f}",
             sum(support)])

# Print table
print("\nClassification Results:")
print("=" * 60)
print(tabulate(data, 
              headers=["", "precision", "recall", "f1-score", "support"],
              tablefmt="grid",
              numalign="right"))
print("=" * 60)

# Save detailed predictions to CSV
predictions_df = pd.DataFrame({
    'Text': df.loc[y_test.index, 'text '],
    'True_Label': y_test,
    'Predicted_Label': predictions
})
predictions_df.to_csv('classification_results.csv', index=False)
print("\nDetailed predictions have been saved to 'classification_results.csv'")