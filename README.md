# LAPD Crime Prediction with Perceptron

A machine learning project that predicts crime severity (Part 1 vs Part 2) using Los Angeles Police Department crime data from 2020 to present.

## Project Overview

This project analyzes LAPD crime data to build a predictive model that classifies crimes into two categories:
- **Part 1**: Serious crimes (more severe)
- **Part 2**: Minor crimes (less severe)

The model uses a Perceptron classifier with hyperparameter optimization and probability calibration to achieve robust predictions.

## Key Features

- **Data Analysis**: Comprehensive EDA of 990,293 crime records
- **Feature Engineering**: Temporal, spatial, and demographic feature extraction
- **Model Training**: Perceptron with RandomizedSearchCV for hyperparameter tuning
- **Model Calibration**: Sigmoid calibration for probability estimates
- **Performance Evaluation**: Multiple metrics including accuracy, F1-score, and ROC-AUC
- **Feature Importance**: Permutation importance analysis
- **Visualization**: Comprehensive plots for data understanding and model evaluation

## Model Performance

The trained Perceptron model achieves:
- **Accuracy**: 75.32%
- **F1-Macro Score**: 73.18%
- **ROC-AUC**: 83.49%

### Baseline Comparison
- Dummy (Most Frequent): 59.70%
- Dummy (Random): 51.87%
- **Perceptron**: 75.32%

## Dataset Information

**Source**: LAPD Crime Data from 2020 to Present
- **Records**: 990,293 crime incidents
- **Features**: 28 original columns
- **Time Period**: 2020-2024
- **Geographic Coverage**: Los Angeles area

### Key Features Used
- **Temporal**: Year, Month, Weekday, Hour, Minute, IsWeekend
- **Demographic**: Victim Age, Victim Sex
- **Spatial**: Latitude, Longitude, Area Name
- **Contextual**: Premise Description, Weapon Description

## Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **Contextily**: Geographic visualization with basemaps

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. **Clone or download this repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you encounter issues with `contextily` on Windows, consider using conda instead:
   ```bash
   conda install contextily
   ```

### Data Setup
1. Download the LAPD crime dataset (`Crime_Data_from_2020_to_Present.csv`)
2. Place the CSV file in the project root directory
3. Run the notebook cells sequentially

### Alternative Installation Methods
- **Using conda** (recommended for geographic libraries):
  ```bash
  conda create -n crime-prediction python=3.9
  conda activate crime-prediction
  conda install pandas numpy matplotlib seaborn scikit-learn contextily joblib
  ```
- **Using pip with virtual environment**:
  ```bash
  python -m venv crime-prediction-env
  source crime-prediction-env/bin/activate  # On Windows: crime-prediction-env\Scripts\activate
  pip install -r requirements.txt
  ```

## Usage

### Running the Analysis
1. **Data Exploration**: Execute cells 1-25 for comprehensive EDA
2. **Data Cleaning**: Run cells 26-32 for preprocessing
3. **Model Training**: Execute cells 33-36 for training and evaluation
4. **Visualization**: Run cells 37-42 for results visualization

### Key Functions
- `file_sha256()`: Computes file integrity hash
- `clip_rare()`: Handles rare categorical values
- Model persistence with joblib for deployment

## Data Preprocessing

### Cleaning Steps
1. **Missing Value Handling**: Strategic imputation for different data types
2. **Outlier Detection**: Age validation (>0, <110) and coordinate filtering
3. **Feature Engineering**: 
   - Temporal features from date/time
   - Geographic binning for spatial features
   - Categorical encoding with frequency thresholds
4. **Data Validation**: Type consistency and range checks

### Feature Engineering
- **Temporal Features**: Hour, day of week, weekend indicator
- **Geographic Binning**: Latitude/longitude discretization
- **Categorical Encoding**: One-hot encoding with minimum frequency filtering
- **Missing Value Strategy**: Median for numeric, most frequent for categorical

## Model Architecture

### Pipeline Components
1. **Preprocessing**: ColumnTransformer with separate pipelines for numeric and categorical data
2. **Classifier**: Perceptron with balanced class weights
3. **Calibration**: Sigmoid calibration for probability estimates

### Hyperparameter Optimization
- **Method**: RandomizedSearchCV with 50 iterations
- **Parameters**: penalty, alpha, eta0, fit_intercept
- **Cross-validation**: 5-fold stratified
- **Scoring**: F1-macro score

## Results & Insights

### Top Crime Types
1. Vehicle - Stolen (112,502 incidents)
2. Battery - Simple Assault (74,747 incidents)
3. Burglary from Vehicle (61,944 incidents)

### Temporal Patterns
- **Peak Hours**: 12 PM - 6 PM
- **Peak Days**: Weekdays show higher activity
- **Seasonal**: Consistent patterns across months

### Geographic Distribution
- **High Activity Areas**: Central, 77th Street, Pacific
- **Crime Hotspots**: Street locations and residential areas


## Model Evaluation

### Confusion Matrix
- **True Positives**: 102,579 (Part 1 correctly identified)
- **False Positives**: 15,672 (Part 2 misclassified as Part 1)
- **False Negatives**: 33,209 (Part 1 misclassified as Part 2)
- **True Negatives**: 46,599 (Part 2 correctly identified)

### Classification Report
- **Part 1 Precision**: 75.54%
- **Part 1 Recall**: 86.75%
- **Part 2 Precision**: 74.83%
- **Part 2 Recall**: 58.39%

## Future Enhancements

1. **Model Improvements**: 
   - Try ensemble methods (Random Forest, XGBoost)
   - Deep learning approaches
   - Feature selection optimization

2. **Data Enhancements**:
   - Weather data integration
   - Economic indicators
   - Population density features

3. **Deployment**:
   - Web application for real-time predictions
   - API development
   - Model monitoring and retraining pipeline

## Notes

- The model uses sigmoid calibration to provide probability estimates
- Geographic coordinates are filtered to Los Angeles area bounds
- Rare categorical values are grouped to improve model stability
- All random operations use seed=42 for reproducibility

## Contributing

Feel free to contribute to this project by:
- Improving the model performance
- Adding new features
- Enhancing visualizations
- Optimizing the preprocessing pipeline

## License

This project is for educational and research purposes. Please ensure compliance with data usage policies when working with LAPD crime data.

---

**Author**: [Prangon Sarwar]  
**Date**: 24-10-2025  
**Dataset**: LAPD Crime Data from 2020 to Present
