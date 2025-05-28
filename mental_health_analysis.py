import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Detailed text preprocessing"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)

def plot_class_distribution(y, title):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Calculate percentages
    total = len(y)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        ax.annotate(f'{int(p.get_height())}\n({percentage})', 
                   (x, y), 
                   ha='center', 
                   va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix with custom styling to match reference"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap similar to reference
    colors = ['#ffff00', '#2f4f4f', '#4b0082']  # yellow, darkslategray, indigo
    n_bins = 200
    custom_cmap = plt.cm.get_cmap('viridis', n_bins)
    
    # Plot heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='viridis',
                square=True,
                cbar=True,
                cbar_kws={"shrink": .8, "label": ""},
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['0', '1', '2'],
                yticklabels=['0', '1', '2'])
    
    plt.title('Confusion Matrix', pad=20, size=14)
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print confusion matrix metrics
    print(f"\nConfusion Matrix Metrics:")
    print("-" * 50)
    
    # Calculate and print metrics for each class
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        accuracy = (TP + TN) / np.sum(cm)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass {i}:")
        print(f"True Positives (TP): {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")
        print(f"True Negatives (TN): {TN}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
    
    # Overall accuracy
    print(f"\nOverall Accuracy: {np.trace(cm) / np.sum(cm):.3f}")
    
    # Create table with metrics
    metrics_table = pd.DataFrame({
        'Class': ['0', '1', '2'],
        'Precision': [cm[i,i]/np.sum(cm[:,i]) if np.sum(cm[:,i]) > 0 else 0 for i in range(3)],
        'Recall': [cm[i,i]/np.sum(cm[i,:]) if np.sum(cm[i,:]) > 0 else 0 for i in range(3)],
        'F1-Score': [2*(cm[i,i]/np.sum(cm[:,i]))*(cm[i,i]/np.sum(cm[i,:]))/((cm[i,i]/np.sum(cm[:,i]))+(cm[i,i]/np.sum(cm[i,:]))) 
                    if (np.sum(cm[:,i]) > 0 and np.sum(cm[i,:]) > 0) else 0 for i in range(3)]
    })
    
    print("\nMetrics per Class:")
    print(metrics_table.round(3).to_string(index=False))

def plot_roc_curves(y_test, y_score, n_classes):
    """Plot ROC curves for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend(loc="lower right")
    plt.show()

# Load data
print("Loading and preprocessing data...")
excel_path = "clean_data (1) - Copy.xlsx"
df = pd.read_excel(excel_path)

# Clean and preprocess
df = df.dropna(subset=['text ', 'label'])
df['processed_text'] = df['text '].apply(preprocess_text)

# Display preprocessing results
print("\n=== Preprocessing Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Average text length before preprocessing: {df['text '].str.len().mean():.1f} characters")
print(f"Average text length after preprocessing: {df['processed_text'].str.len().mean():.1f} characters")

# Plot original class distribution
print("\n=== Class Distribution Before Balancing ===")
plot_class_distribution(df['label'], 'Class Distribution Before Balancing')

# Prepare features
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])
y = df['label']

# Split data (70:20:10)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp)

print("\n=== Data Split Statistics ===")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\n=== Class Distribution After SMOTE ===")
plot_class_distribution(y_train_balanced, 'Class Distribution After SMOTE')

# Hyperparameter tuning
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1]
}

xgb_clf = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X_train_balanced, y_train_balanced, cv=5)
print("\n=== Cross-validation Scores ===")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train final model
final_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on all sets
sets = {
    'Training': (X_train, y_train),
    'Validation': (X_val, y_val),
    'Test': (X_test, y_test)
}

print("\n=== Model Evaluation ===")
for set_name, (X_set, y_set) in sets.items():
    predictions = final_model.predict(X_set)
    print(f"\n{set_name} Set Results:")
    print("=" * 60)
    print(classification_report(y_set, predictions))
    plot_confusion_matrix(y_set, predictions, f'Confusion Matrix - {set_name} Set')

# ROC curves for test set
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = final_model.predict_proba(X_test)
plot_roc_curves(y_test_bin, y_score, n_classes=3)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': tfidf.get_feature_names_out(),
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance Score')
plt.tight_layout()
plt.show()

# Save results
predictions_df = pd.DataFrame({
    'Text': df.loc[y_test.index, 'text '],
    'True_Label': y_test,
    'Predicted_Label': final_model.predict(X_test)
})
predictions_df.to_csv('detailed_classification_results.csv', index=False)
print("\nDetailed predictions have been saved to 'detailed_classification_results.csv'") 