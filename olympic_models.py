"""
Olympic Medal Prediction Models
Implements Linear Regression, Gradient Boosting, and SVM models
Based on the analysis from LR_GB_SVM_models.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, 
                            confusion_matrix, classification_report,
                            mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

# Dataset path - using the actual dataset
DATASET_PATH = r"C:\Users\bless\OneDrive\Desktop\IT projects\Olympic Dash\cleaned_athlete_events_gdp_n_pop.csv"


def load_data():
    """Load and return the Olympic dataset."""
    df = pd.read_csv(DATASET_PATH)
    return df


def prepare_regression_data(df):
    """Prepare features and target for regression models."""
    # Features (economic indicators) and target (Olympic performance)
    X = df[['Population', 'GDP_USD', 'GDP_per_capita_USD']]
    y = df['Total']   # Total medals
    return X, y


def prepare_classification_data(df):
    """Prepare features and target for SVM classification."""
    X = df[['GDP_USD', 'GDP_per_capita_USD', 'Population']]
    
    # Target: High medal countries (above median)
    median_medals = df['Total'].median()
    y = (df['Total'] > median_medals).astype(int)
    
    return X, y, median_medals


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
            'Feature': self.coefficients['Feature'],
            'Importance': importance
        }).sort_values('Importance', ascending=False)


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
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
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
        
    def train(self, X, y):
        """Train the SVM classification model."""
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
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        self.metrics['accuracy'] = self.metrics['classification_report']['accuracy']
        
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get decision function values (distance from hyperplane)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)


class OlympicPredictor:
    """Main class that combines all three models for Olympic predictions."""
    
    def __init__(self):
        self.df = None
        self.lr_model = LinearRegressionModel()
        self.gb_model = GradientBoostingModel()
        self.svm_model = SVMClassificationModel()
        self.is_trained = False
        
    def load_and_train(self):
        """Load data and train all models."""
        print("Loading dataset...")
        self.df = load_data()
        
        # Prepare data for regression models
        X_reg, y_reg = prepare_regression_data(self.df)
        
        # Prepare data for classification model
        X_clf, y_clf, median_threshold = prepare_classification_data(self.df)
        self.svm_model.median_threshold = median_threshold
        
        # Train Linear Regression
        print("Training Linear Regression model...")
        lr_metrics = self.lr_model.train(X_reg, y_reg)
        print(f"  R²: {lr_metrics['r2_score']:.3f}, RMSE: {lr_metrics['rmse']:.2f}")
        
        # Train Gradient Boosting
        print("Training Gradient Boosting model...")
        gb_metrics = self.gb_model.train(X_reg, y_reg)
        print(f"  R²: {gb_metrics['r2_score']:.3f}, RMSE: {gb_metrics['rmse']:.2f}")
        
        # Train SVM
        print("Training SVM Classification model...")
        svm_metrics = self.svm_model.train(X_clf, y_clf)
        print(f"  Accuracy: {svm_metrics['accuracy']:.3f}")
        
        self.is_trained = True
        
        return {
            'linear_regression': lr_metrics,
            'gradient_boosting': gb_metrics,
            'svm': svm_metrics
        }
    
    def predict_2028(self, country_data):
        """Predict 2028 medals for given country data."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Use Gradient Boosting for final prediction (best performer)
        X = country_data[['Population', 'GDP_USD', 'GDP_per_capita_USD']]
        prediction = self.gb_model.predict(X)
        
        return prediction
    
    def classify_country(self, country_data):
        """Classify if a country is high or low medal using SVM."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        X = country_data[['GDP_USD', 'GDP_per_capita_USD', 'Population']]
        prediction = self.svm_model.predict(X)
        
        return prediction
    
    def compare_models(self):
        """Return comparison of all three models."""
        if not self.is_trained:
            return None
        
        return {
            'linear_regression': {
                'name': 'Linear Regression',
                'r2_score': self.lr_model.metrics['r2_score'],
                'rmse': self.lr_model.metrics['rmse'],
                'mae': self.lr_model.metrics['mae']
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'r2_score': self.gb_model.metrics['r2_score'],
                'rmse': self.gb_model.metrics['rmse'],
                'mae': self.gb_model.metrics['mae']
            },
            'svm': {
                'name': 'SVM Classification',
                'accuracy': self.svm_model.metrics['accuracy'],
                'type': 'classification'
            }
        }
    
    def get_all_feature_importance(self):
        """Get feature importance from all applicable models."""
        if not self.is_trained:
            return None
        
        return {
            'linear_regression': self.lr_model.get_feature_importance(),
            'gradient_boosting': self.gb_model.get_feature_importance()
        }


def train_all_models():
    """Main function to train all models and return the predictor."""
    predictor = OlympicPredictor()
    all_metrics = predictor.load_and_train()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"\nLinear Regression R²: {all_metrics['linear_regression']['r2_score']:.3f}")
    print(f"Linear Regression RMSE: {all_metrics['linear_regression']['rmse']:.2f} medals")
    print(f"\nGradient Boosting R²: {all_metrics['gradient_boosting']['r2_score']:.3f}")
    print(f"Gradient Boosting RMSE: {all_metrics['gradient_boosting']['rmse']:.2f} medals")
    print(f"\nSVM Classification Accuracy: {all_metrics['svm']['accuracy']:.3f}")
    print("="*50)
    
    return predictor


if __name__ == "__main__":
    # Run training
    predictor = train_all_models()
    
    # Show model comparison
    comparison = predictor.compare_models()
    print("\nModel Comparison:")
    for model_name, metrics in comparison.items():
        print(f"\n{metrics['name']}:")
        for metric, value in metrics.items():
            if metric != 'name':
                print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
