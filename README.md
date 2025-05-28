# Mental Health Text Classification

This project implements text classification for mental health data using XGBoost algorithm. The model classifies text into three categories (0, 1, 2).

## Features
- Text preprocessing using TF-IDF vectorization
- XGBoost classifier implementation
- Performance metrics calculation (precision, recall, f1-score)
- Results output in both table format and CSV file

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- tabulate

## Usage
1. Place your data file (Excel format) in the project directory
2. Run `python work.py`
3. Check the results in the console and `classification_results.csv` 