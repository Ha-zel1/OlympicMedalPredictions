"""
Olympic Medal Prediction - Enhanced Flask Application
Features: Interactive Dashboard, Model Explainability, What-If Simulator
"""

from flask import Flask, render_template, jsonify, request
from model_enhanced import train_and_predict_model, what_if_prediction, get_country_historical_data
import pandas as pd
import json
import os

app = Flask(__name__)

# Configure for Render
app.config['DEBUG'] = False
port = int(os.environ.get('PORT', 5000))

# Store model results globally for efficiency
model_results = None


def get_model_results():
    """Get or compute model results."""
    global model_results
    if model_results is None:
        model_results = train_and_predict_model()
    return model_results


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/predict')
def predict():
    """Generate predictions and show full dashboard."""
    results = get_model_results()
    
    # Get selected year from query parameter (default to 2028)
    selected_year = request.args.get('year', type=int, default=2028)
    
    # Validate year is an Olympic year
    olympic_years = [2028, 2032, 2036, 2040]
    available_years = results.get('available_years', olympic_years)
    if selected_year not in available_years:
        selected_year = 2028
    
    # Get predictions for selected year
    predictions_by_year = results.get('predictions_by_year', {})
    if selected_year in predictions_by_year:
        data_year = predictions_by_year[selected_year]
        pred_col = f'Predicted_Total_{selected_year}'
        # Create predictions dataframe for selected year
        predictions = data_year[['Country_Name', 'Total', pred_col]].copy()
        predictions.columns = ['Country_Name', 'Total', 'Predicted_Total']
        predictions = predictions.sort_values('Predicted_Total', ascending=False)
        predictions.insert(0, 'Rank', range(1, len(predictions) + 1))
        predictions.rename(columns={'Predicted_Total': f'Predicted_Total_{selected_year}'}, inplace=True)
    else:
        # Fallback to default 2028 predictions
        predictions = results['predictions']
    
    metrics = results['metrics']
    all_metrics = results.get('all_metrics', {'linear_regression': metrics, 'gradient_boosting': metrics, 'svm': {'accuracy': 0.61}})
    
    # Convert predictions to HTML table
    predictions_html = predictions.to_html(
        classes='table table-striped table-hover',
        index=False,
        float_format=lambda x: f'{x:.0f}' if pd.notna(x) else ''
    )
    
    return render_template('dashboard.html',
                           predictions_table=predictions_html,
                           metrics=metrics,
                           all_metrics=all_metrics,
                           available_years=available_years,
                           selected_year=selected_year,
                           has_shap=True)


@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions data."""
    results = get_model_results()
    predictions = results['predictions']
    
    return jsonify({
        'predictions': predictions.to_dict('records'),
        'metrics': results['metrics'],
        'feature_importance': results['feature_importance'].to_dict('records')
    })


@app.route('/api/country/<country_name>')
def api_country_detail(country_name):
    """API endpoint for country-specific data."""
    results = get_model_results()
    df = results['df']
    
    # Get historical data
    country_data = get_country_historical_data(df, country_name)
    
    # Get 2028 prediction if available
    prediction_row = results['data_2028'][results['data_2028']['Country_Name'] == country_name]
    prediction = None
    if not prediction_row.empty:
        prediction = {
            'predicted_2028': float(prediction_row['Predicted_Total_2028'].iloc[0]),
            'actual_2024': float(prediction_row['Total'].iloc[0])
        }
    
    return jsonify({
        'country': country_name,
        'historical': country_data.to_dict('records'),
        'prediction': prediction
    })


@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    """API endpoint for what-if scenario analysis."""
    results = get_model_results()
    
    data = request.get_json()
    country = data.get('country')
    changes = data.get('changes', {})
    
    # Get country data
    country_data = results['data_2028'][results['data_2028']['Country_Name'] == country]
    
    if country_data.empty:
        return jsonify({'error': 'Country not found'}), 404
    
    # Calculate what-if scenario using the Gradient Boosting model
    gb_model = results['all_models']['gradient_boosting']['model']
    scenario_result = what_if_prediction(
        gb_model,
        country_data,
        changes
    )
    
    return jsonify({
        'country': country,
        'changes': changes,
        'result': scenario_result
    })


@app.route('/country/<country_name>')
def country_detail(country_name):
    """Country detail page."""
    return render_template('country_detail.html', country_name=country_name)


@app.route('/whatif')
def whatif_simulator():
    """What-If Scenario Simulator page."""
    results = get_model_results()
    countries = results['data_2028']['Country_Name'].tolist()
    features = results['features']
    
    return render_template('whatif.html', 
                          countries=countries,
                          features=features)


@app.route('/analytics')
def analytics():
    """Advanced analytics page with model explainability."""
    results = get_model_results()
    
    # Get top insights
    predictions = results['predictions']
    
    insights = {
        'total_countries': len(predictions),
        'countries_with_increase': len(predictions[predictions['Predicted_Total_2028'] > predictions['Total']]),
        'countries_with_decrease': len(predictions[predictions['Predicted_Total_2028'] < predictions['Total']]),
        'top_gainer': predictions.iloc[0]['Country_Name'],
        'top_gainer_increase': predictions.iloc[0]['Predicted_Total_2028'] - predictions.iloc[0]['Total']
    }
    
    return render_template('analytics.html',
                          metrics=results['metrics'],
                          feature_importance=results['feature_importance'].to_dict('records'),
                          insights=insights)


@app.route('/about')
def about():
    """About page with methodology."""
    results = get_model_results()
    return render_template('about.html', metrics=results['metrics'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
