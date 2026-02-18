"""
Olympic Medal Prediction Model with All Three Algorithms
Features:
- Linear Regression, Gradient Boosting, and SVM models
- Model comparison and metrics
- SHAP explainability for model interpretability
- Model performance metrics (R², RMSE, MAE)
- Historical trend analysis
- What-if scenario predictions

Based on: LR_GB_SVM_models.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                            confusion_matrix, classification_report, accuracy_score)
from sklearn.model_selection import train_test_split
import os
import json

# Optional SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")

# Dataset path - using the actual dataset from the notebook
DATASET_PATH = r"C:\Users\bless\OneDrive\Desktop\IT projects\Olympic Dash\cleaned_athlete_events_gdp_n_pop.csv"


def load_and_prepare_data(filepath=None, for_training=False):
    """Load and prepare the Olympic dataset.
    
    Args:
        filepath: Path to dataset
        for_training: If True, return raw data without sorting (matches notebook exactly)
    """
    if filepath is None:
        filepath = DATASET_PATH
    df = pd.read_csv(filepath)
    
    if for_training:
        # Return raw data without any modifications (matches notebook exactly)
        return df
    
    # For prediction purposes, sort and create lag features
    df = df.sort_values(by=['Country_Name', 'Year'])
    
    # Create lag features (past performance)
    df['Past_Total'] = df.groupby('Country_Name')['Total'].shift(1)
    df['Past_Gold'] = df.groupby('Country_Name')['Gold'].shift(1) if 'Gold' in df.columns else 0
    df['Past_Silver'] = df.groupby('Country_Name')['Silver'].shift(1) if 'Silver' in df.columns else 0
    df['Past_Bronze'] = df.groupby('Country_Name')['Bronze'].shift(1) if 'Bronze' in df.columns else 0
    
    # Calculate growth rate
    df['Medal_Growth_Rate'] = df.groupby('Country_Name')['Total'].pct_change()
    
    # Drop rows with NaN in key features
    df_clean = df.dropna(subset=['Past_Total'])
    
    return df_clean


class LinearRegressionModel:
    """Linear Regression Model for Olympic Medal Prediction."""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.metrics = {}
        self.coefficients = None
        self.is_trained = False
        
    def train(self, X, y):
        """Train the Linear Regression model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics['r2_score'] = r2_score(y_test, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)
        
        # Store coefficients
        self.coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': self.model.coef_
        })
        self.metrics['intercept'] = self.model.intercept_
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.feature_columns = X.columns.tolist()
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Return feature importance based on coefficients."""
        if self.coefficients is None:
            return None
        # Normalize coefficients to get relative importance
        coef_abs = np.abs(self.coefficients['Coefficient'])
        importance = coef_abs / coef_abs.sum()
        return pd.DataFrame({
            'feature': self.coefficients['Feature'],
            'importance': importance
        }).sort_values('importance', ascending=False)


class GradientBoostingModel:
    """Gradient Boosting Model for Olympic Medal Prediction."""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.metrics = {}
        self.is_trained = False
        self.feature_importance = None
        
    def train(self, X, y):
        """Train the Gradient Boosting model."""
        # Train-test split (no scaling needed for tree-based models)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics['r2_score'] = r2_score(y_test, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.feature_columns = X.columns.tolist()
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Return feature importance."""
        return self.feature_importance


class SVMClassificationModel:
    """SVM Classification Model for High/Low Medal Prediction."""
    
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C)
        self.scaler = StandardScaler()
        self.metrics = {}
        self.is_trained = False
        self.median_threshold = None
        
    def train(self, df):
        """Train the SVM classification model."""
        # Prepare data
        X = df[['GDP_USD', 'GDP_per_capita_USD', 'Population']]
        
        # Target: High medal countries (above median)
        median_medals = df['Total'].median()
        y = (df['Total'] > median_medals).astype(int)
        self.median_threshold = median_medals
        
        # Scale features (required for SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        self.metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['median_threshold'] = median_medals
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.feature_columns = X.columns.tolist()
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def classify_country(self, country_data):
        """Classify a country as high or low medal."""
        prediction = self.predict(country_data)
        return "High Medal" if prediction[0] == 1 else "Low Medal"


def train_all_models(df):
    """Train all three models and return results."""
    # Use raw data for training to match notebook exactly
    df_train = load_and_prepare_data(for_training=True)
    
    # Prepare regression data - same features as notebook
    X_reg = df_train[['Population', 'GDP_USD', 'GDP_per_capita_USD']]
    y_reg = df_train['Total']
    
    # Train Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegressionModel()
    lr_metrics = lr_model.train(X_reg, y_reg)
    print(f"  R²: {lr_metrics['r2_score']:.3f}, RMSE: {lr_metrics['rmse']:.2f}")
    
    # Train Gradient Boosting - notebook uses different feature order
    print("Training Gradient Boosting...")
    X_gb = df_train[['GDP_USD', 'GDP_per_capita_USD', 'Population']]
    gb_model = GradientBoostingModel()
    gb_metrics = gb_model.train(X_gb, y_reg)
    print(f"  R²: {gb_metrics['r2_score']:.3f}, RMSE: {gb_metrics['rmse']:.2f}")
    
    # Train SVM - use same training data
    print("Training SVM Classification...")
    svm_model = SVMClassificationModel()
    svm_metrics = svm_model.train(df_train)
    print(f"  Accuracy: {svm_metrics['accuracy']:.3f}")
    
    return {
        'linear_regression': {'model': lr_model, 'metrics': lr_metrics},
        'gradient_boosting': {'model': gb_model, 'metrics': gb_metrics},
        'svm': {'model': svm_model, 'metrics': svm_metrics}
    }


def predict_for_year(gb_model, df, year):
    """Generate predictions for a specific Olympic year using Gradient Boosting.
    
    Args:
        gb_model: trained Gradient Boosting model
        df: DataFrame with country data (2024 data used as base)
        year: target year (2028-2039)
    
    Returns:
        DataFrame with predictions for the specified year
    """
    data_year = df[df['Year'] == 2024].copy()
    data_year['Year'] = year
    
    # Prepare features for prediction - must match training order
    X_year = data_year[['GDP_USD', 'GDP_per_capita_USD', 'Population']]
    
    # Get base prediction
    base_predictions = gb_model.predict(X_year)
    
    # Adjust predictions based on year (simple growth model)
    # Years from 2024
    years_ahead = year - 2024
    
    # Apply small growth factor for future years (0.5% per year growth trend)
    growth_factor = 1 + (0.005 * years_ahead)
    adjusted_predictions = base_predictions * growth_factor
    
    data_year[f'Predicted_Total_{year}'] = adjusted_predictions.round().astype(int).clip(min=0)
    
    return data_year


def predict_years_range(gb_model, df, olympic_years=None):
    """Generate predictions for Olympic years only.
    
    Args:
        gb_model: trained Gradient Boosting model
        df: DataFrame with country data
        olympic_years: list of Olympic years (default: [2028, 2032, 2036, 2040])
    
    Returns a dictionary with predictions for each Olympic year.
    """
    if olympic_years is None:
        olympic_years = [2028, 2032, 2036, 2040]
    
    predictions_by_year = {}
    
    for year in olympic_years:
        data_year = predict_for_year(gb_model, df, year)
        predictions_by_year[year] = data_year
    
    return predictions_by_year


def generate_visualizations(models, data_2028, output_dir='static/images'):
    """Generate all model visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 300
    
    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models_data = [
        ('Linear\nRegression', models['linear_regression']['metrics']['r2_score'], 
         models['linear_regression']['metrics']['rmse']),
        ('Gradient\nBoosting', models['gradient_boosting']['metrics']['r2_score'],
         models['gradient_boosting']['metrics']['rmse']),
    ]
    
    x = np.arange(len(models_data))
    width = 0.35
    
    r2_scores = [m[1] for m in models_data]
    rmse_scores = [m[2] for m in models_data]
    
    ax1 = ax
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='#3498db')
    ax2.set_ylabel('RMSE (medals)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models_data])
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add value labels
    for bar, value in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar, value in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top 15 Countries Prediction
    top_15 = data_2028.nlargest(15, 'Predicted_Total_2028')
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_15))
    width = 0.35
    
    plt.bar(x - width/2, top_15['Total'], width, label='2024 Actual', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, top_15['Predicted_Total_2028'], width, label='2028 Predicted', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Country', fontsize=12, fontweight='bold')
    plt.ylabel('Total Medals', fontsize=12, fontweight='bold')
    plt.title('Top 15 Countries: 2024 Actual vs 2028 Predicted Medals', fontsize=14, fontweight='bold')
    plt.xticks(x, top_15['Country_Name'], rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance from Gradient Boosting
    gb_importance = models['gradient_boosting']['model'].get_feature_importance()
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(gb_importance)))
    bars = plt.barh(gb_importance['feature'], gb_importance['importance'], color=colors, alpha=0.8)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Feature Importance - Gradient Boosting Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar, value in zip(bars, gb_importance['importance']):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SVM Confusion Matrix
    svm_metrics = models['svm']['metrics']
    cm = np.array(svm_metrics['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Medal', 'High Medal'],
                yticklabels=['Low Medal', 'High Medal'])
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'SVM Classification Confusion Matrix\n(Accuracy: {svm_metrics["accuracy"]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Residual Plot for Gradient Boosting
    gb_model = models['gradient_boosting']['model']
    y_pred = gb_model.y_pred
    y_test = gb_model.y_test
    residuals = y_pred - y_test
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='#9b59b6', edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Medals', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
    plt.title(f'Residual Plot - Gradient Boosting\n(R² = {models["gradient_boosting"]["metrics"]["r2_score"]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_country_historical_data(df, country_name):
    """Get historical medal data for a specific country."""
    country_data = df[df['Country_Name'] == country_name].sort_values('Year')
    return country_data[['Year', 'Total', 'Gold', 'Silver', 'Bronze']] if 'Gold' in df.columns else country_data[['Year', 'Total']]


def what_if_prediction(gb_model, country_data, changes):
    """
    Generate what-if scenario predictions using Gradient Boosting.
    
    Parameters:
    - gb_model: trained Gradient Boosting model
    - country_data: DataFrame with country data
    - changes: dict of {feature_name: multiplier}
    """
    # Get base features - must match training order: GDP_USD, GDP_per_capita_USD, Population
    X = country_data[['GDP_USD', 'GDP_per_capita_USD', 'Population']].copy()
    base_values = X.iloc[0].values
    
    # Create modified features
    modified = X.copy()
    for feature, multiplier in changes.items():
        if feature in modified.columns:
            modified[feature] = modified[feature] * multiplier
    
    # Get predictions
    base_prediction = gb_model.predict(X)[0]
    new_prediction = gb_model.predict(modified)[0]
    
    return {
        'base_prediction': round(base_prediction, 1),
        'new_prediction': round(new_prediction, 1),
        'difference': round(new_prediction - base_prediction, 1),
        'percent_change': round(((new_prediction - base_prediction) / base_prediction * 100), 1) if base_prediction > 0 else 0
    }


def train_and_predict_model():
    """Main function to train all models and generate predictions."""
    # Load data
    df = load_and_prepare_data()
    
    # Train all models
    models = train_all_models(df)
    
    # Generate predictions for Olympic years using Gradient Boosting (best performer)
    olympic_years = [2028, 2032, 2036, 2040]
    predictions_by_year = predict_years_range(models['gradient_boosting']['model'], df, olympic_years)
    
    # Default to 2028 for backward compatibility
    data_2028 = predictions_by_year[2028]
    
    # Generate visualizations
    generate_visualizations(models, data_2028)
    
    # Prepare results for default year (2028)
    predictions = data_2028[['Country_Name', 'Total', 'Predicted_Total_2028']].sort_values(
        'Predicted_Total_2028', ascending=False
    )
    predictions.insert(0, 'Rank', range(1, len(predictions) + 1))
    
    # Use Gradient Boosting metrics as primary metrics (best model)
    primary_metrics = models['gradient_boosting']['metrics']
    
    # Add all model metrics for comparison
    all_metrics = {
        'linear_regression': models['linear_regression']['metrics'],
        'gradient_boosting': models['gradient_boosting']['metrics'],
        'svm': models['svm']['metrics']
    }
    
    return {
        'predictions': predictions,
        'metrics': primary_metrics,
        'all_metrics': all_metrics,
        'feature_importance': models['gradient_boosting']['model'].get_feature_importance(),
        'model': models['gradient_boosting']['model'],
        'all_models': models,
        'features': ['Population', 'GDP_USD', 'GDP_per_capita_USD'],
        'data_2028': data_2028,
        'predictions_by_year': predictions_by_year,
        'available_years': olympic_years,
        'df': df
    }


if __name__ == '__main__':
    results = train_and_predict_model()
    
    print("\n" + "="*50)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*50)
    
    print("\nLinear Regression:")
    print(f"  R²: {results['all_metrics']['linear_regression']['r2_score']:.3f}")
    print(f"  RMSE: {results['all_metrics']['linear_regression']['rmse']:.2f} medals")
    
    print("\nGradient Boosting (Primary Model):")
    print(f"  R²: {results['all_metrics']['gradient_boosting']['r2_score']:.3f}")
    print(f"  RMSE: {results['all_metrics']['gradient_boosting']['rmse']:.2f} medals")
    
    print("\nSVM Classification:")
    print(f"  Accuracy: {results['all_metrics']['svm']['accuracy']:.3f}")
    
    print("\n" + "="*50)
    print("\nTop 10 Predicted 2028 Medal Winners:")
    print(results['predictions'].head(10).to_string(index=False))
