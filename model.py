import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import os

def train_and_predict_model():
    # Load the dataset
    df = pd.read_csv('C:/Users/bless/OneDrive/Desktop/IT projects/Olympic Dash/cleaned_athlete_events_gdp_n_pop.csv')
    
    # Sort data to ensure chronological order by country and year
    df = df.sort_values(by=['Country_Name', 'Year'])
    
    # Shift the "Total" column by one Olympic cycle (4 years)
    df['Past_Total'] = df.groupby('Country_Name')['Total'].shift(1)
    
    # Drop rows where Past_Total is NaN
    df = df.dropna(subset=['Past_Total'])
    
    # Select features and target
    features = ['GDP_USD', 'GDP_per_capita_USD', 'Population', 'Past_Total']
    target = 'Total'
    
    # Split data into training set
    train_data = df[df['Year'] <= 2024]
    X_train = train_data[features]
    y_train = train_data[target]
    
    # Train the model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    # Prepare 2028 data
    data_2028 = df[df['Year'] == 2024].copy()
    data_2028['Year'] = 2028
    data_2028['Past_Total'] = data_2028['Total']
    
    # Predict 2028 medals
    X_2028 = data_2028[features]
    data_2028['Predicted_Total_2028'] = model.predict(X_2028)
    
    # Generate Graphs
    generate_graphs(model, X_train, y_train, data_2028)
    
    return data_2028[['Country_Name', 'Total', 'Predicted_Total_2028']]

def generate_graphs(model, X_train, y_train, data_2028):
    # Ensure the images directory exists
    os.makedirs('static/images', exist_ok=True)
    # Increase figure size and resolution
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 300  # Higher resolution
    
    # 1. Comparison Bar Graph
    plt.figure(figsize=(12, 8))
    results_comparison = data_2028[['Country_Name', 'Total', 'Predicted_Total_2028']].set_index('Country_Name')
    width = 0.35
    indices = np.arange(len(results_comparison))
    plt.bar(indices - width/2, results_comparison['Total'], width=width, label='2024 Actual')
    plt.bar(indices + width/2, results_comparison['Predicted_Total_2028'], width=width, label='2028 Predicted')
    plt.xlabel('Country')
    plt.ylabel('Total Medals')
    plt.title('Comparison of Actual 2024 vs Predicted 2028 Olympic Medals')
    plt.xticks(indices, results_comparison.index, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/comparison_graph.png')
    plt.close()
    
    # 2. Feature Importance Graph
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    feature_importance = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png')
    plt.close()
    
    # 3. Residual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, model.predict(X_train) - y_train)
    plt.xlabel('Actual Total Medals')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('static/images/residual_plot.png')
    plt.close()