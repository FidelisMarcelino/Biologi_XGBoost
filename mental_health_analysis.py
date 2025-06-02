import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
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
    """Plot class distribution with percentages"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=y)
    plt.title(title, pad=20, size=14)
    plt.xlabel('Class', size=12)
    plt.ylabel('Count', size=12)
    
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
    """Plot confusion matrix with metrics"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='viridis',
                square=True,
                cbar=True,
                xticklabels=['0', '1', '2'],
                yticklabels=['0', '1', '2'])
    
    plt.title(title, pad=20, size=14)
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    plt.tight_layout()
    plt.show()

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

# Load and preprocess data
print("\n=== Data Loading and Preprocessing ===")
excel_path = "clean_data (1) - Copy.xlsx"
df = pd.read_excel(excel_path)

# Clean and preprocess
df = df.dropna(subset=['text ', 'label'])
df['processed_text'] = df['text '].apply(preprocess_text)

# Display preprocessing statistics
print("\nPreprocessing Statistics:")
print("-" * 50)
print(f"Total samples: {len(df)}")
print(f"Average text length before preprocessing: {df['text '].str.len().mean():.1f} characters")
print(f"Average text length after preprocessing: {df['processed_text'].str.len().mean():.1f} characters")

# Plot original class distribution
print("\nClass Distribution Before Balancing:")
plot_class_distribution(df['label'], 'Class Distribution Before Balancing')

# Prepare features
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])
y = df['label']

# Split data (70:20:10)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp)

print("\nData Split Statistics:")
print("-" * 50)
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Apply SMOTE
print("\n=== Applying SMOTE for Data Balancing ===")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
plot_class_distribution(y_train_balanced, 'Class Distribution After SMOTE')

# Perform detailed cross-validation
print("\n=== Detailed Cross-validation Results ===")
base_model = xgb.XGBClassifier(
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Get detailed metrics for each fold
cv_results = cross_validate(
    base_model, 
    X_train_balanced, 
    y_train_balanced,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=True,
    n_jobs=-1  # Use all CPU cores
)

# Create detailed results table
fold_results = pd.DataFrame({
    'Fold': range(1, 6),
    'Train Accuracy': cv_results['train_accuracy'],
    'Val Accuracy': cv_results['test_accuracy'],
    'Train Precision': cv_results['train_precision_macro'],
    'Val Precision': cv_results['test_precision_macro'],
    'Train Recall': cv_results['train_recall_macro'],
    'Val Recall': cv_results['test_recall_macro'],
    'Train F1': cv_results['train_f1_macro'],
    'Val F1': cv_results['test_f1_macro']
})

# Add mean and std rows
mean_row = fold_results.mean().to_frame('Mean').T
std_row = fold_results.std().to_frame('Std').T
fold_results = pd.concat([fold_results, mean_row, std_row])

# Format the table
print("\nDetailed Cross-validation Results:")
print(tabulate(fold_results.round(3), headers='keys', tablefmt='grid', showindex=False))

# Hyperparameter tuning with optimized parameter grid
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1],  # Reduced learning rate options
    'n_estimators': [100],   # Reduced number of estimators
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [0.8],      # Added subsample parameter
    'colsample_bytree': [0.8]  # Added column sampling parameter
}

# Create and configure the GridSearchCV with verbose output
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores for individual model
        verbosity=0  # Reduce XGBoost verbosity
    ),
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,  # Use all CPU cores for parallel grid search
    verbose=2    # Add verbosity to track progress
)

print("\nStarting Grid Search (this may take a few minutes)...")
grid_search.fit(X_train_balanced, y_train_balanced)

print("\nBest Parameters:")
print("-" * 50)
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")
print(f"\nBest cross-validation score: {grid_search.best_score_:.3f}")

# Train final model with best parameters
print("\n=== Model Evaluation ===")
final_model = xgb.XGBClassifier(
    **grid_search.best_params_,
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbosity=0
)

print("\nTraining final model...")
final_model.fit(
    X_train_balanced,
    y_train_balanced,
    verbose=False
)

# Evaluate on all sets
sets = {
    'Training': (X_train, y_train),
    'Validation': (X_val, y_val),
    'Test': (X_test, y_test)
}

for set_name, (X_set, y_set) in sets.items():
    print(f"\n{set_name} Set Results:")
    print("-" * 50)
    predictions = final_model.predict(X_set)
    
    print(f"\nClassification Report - {set_name} Set:")
    print(classification_report(y_set, predictions))
    
    print(f"\nConfusion Matrix - {set_name} Set:")
    plot_confusion_matrix(y_set, predictions, f'Confusion Matrix - {set_name} Set')

# ROC curves for test set
print("\n=== ROC Curves ===")
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = final_model.predict_proba(X_test)
plot_roc_curves(y_test_bin, y_score, n_classes=3)

# Feature importance visualization
print("\n=== Feature Importance Analysis ===")
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
print("\n=== Saving Results ===")
predictions_df = pd.DataFrame({
    'Text': df.loc[y_test.index, 'text '],
    'True_Label': y_test,
    'Predicted_Label': final_model.predict(X_test),
    'Probability_Class_0': y_score[:, 0],
    'Probability_Class_1': y_score[:, 1],
    'Probability_Class_2': y_score[:, 2]
})
predictions_df.to_csv('detailed_classification_results.csv', index=False)
print("\nDetailed predictions have been saved to 'detailed_classification_results.csv'") 