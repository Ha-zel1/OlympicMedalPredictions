# OlympicAI 2028 🏅

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ** Olympic Medal Predictions Powered with AI capabilities for the 2028 Los Angeles Games and further on**

![OlympicAI Dashboard]([static/images/dashboard-preview.png](https://olympicmedalpredictions.onrender.com))

## 🎯 Project Overview

OlympicAI is a comprehensive machine learning solution that predicts Olympic medal counts for the 2028 Los Angeles Olympics and further on. Built with modern data science techniques and explainable AI principles, this project demonstrates end-to-end capabilities from data preprocessing to model deployment and visualization.

### Key Features

- 🤖 **ML-Powered Predictions** - Gradient Boosting Regressor trained on historical Olympic data
- 💡 **Explainable AI** - SHAP values and feature importance analysis
- 🧪 **What-If Simulator** - Test scenarios and hypothesis analysis
- 📊 **Rich Visualizations** - Interactive charts and model performance metrics
- 🔌 **RESTful API** - Programmatic access to predictions
- 📱 **Responsive Design** - Modern UI built with TailwindCSS

---

## 🚀 Live Demo

```bash
# Clone the repository
git clone https://github.com/yourusername/olympicai-2028.git
cd olympicai-2028

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit `(https://olympicmedalpredictions.onrender.com)` to explore the dashboard.

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | ~0.85 | Model explains 85% of variance |
| **RMSE** | ~8.5 medals | Average prediction error |
| **MAE** | ~6.2 medals | Mean absolute error |

### Algorithm
- **Model**: Gradient Boosting Regressor
- **Estimators**: 200
- **Learning Rate**: 0.1
- **Max Depth**: 4

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Layer    │────▶│  Feature Eng.    │────▶│  ML Model (GBR) │
│                 │     │                  │     │                 │
│ • Historical    │     │ • Lag Features   │     │ • 200 Trees     │
│ • Economic      │     │ • Growth Rates   │     │ • Shap Values   │
│ • Population    │     │ • Normalization  │     │ • Predictions   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Dashboard  │  │  Analytics  │  │  What-If    │             │
│  │  (/predict) │  │  (/analytics)│ │  (/whatif)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │    API      │  │   About     │                               │
│  │(/api/...)   │  │  (/about)   │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
olympic_prediction/
├── app.py                    # Flask application with API endpoints
├── model_enhanced.py         # ML model with SHAP explainability
├── requirements.txt          # Python dependencies
├── data/
│   └── olympic_data.csv      # Dataset (or link to your dataset)
├── static/
│   └── images/               # Generated visualizations
├── templates/
│   ├── index.html            # Landing page
│   ├── dashboard.html        # Main dashboard
│   ├── analytics.html        # Model analytics & SHAP
│   ├── whatif.html           # What-If simulator
│   ├── country_detail.html   # Country deep-dive
│   └── about.html            # Project documentation
└── README.md
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd olympic_prediction
```

2. **Create virtual environment (recommended)**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure dataset path**
Edit `model_enhanced.py` and update the dataset path:
```python
df = pd.read_csv('path/to/your/cleaned_athlete_events_gdp_n_pop.csv')
```

5. **Run the application**
```bash
python app.py
```

6. **Access the dashboard**
Open your browser and navigate to `http://localhost:5000`

---

## 📖 Usage Guide

### Dashboard (`/predict`)
The main dashboard displays:
- Model performance metrics (R², RMSE, MAE)
- Top 15 countries 2028 predictions
- Feature importance visualization
- Comparison charts (2024 vs 2028)
- Medal distribution analysis

### Analytics (`/analytics`)
Deep dive into model explainability:
- SHAP value visualizations
- Feature importance analysis
- Model architecture diagram
- Methodology documentation

### What-If Simulator (`/whatif`)
Test scenarios interactively:
- Select any country
- Adjust GDP, Population, or Past Performance
- See real-time prediction changes
- Compare baseline vs scenario results

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions` | GET | All predictions with metrics |
| `/api/country/<name>` | GET | Country-specific data |
| `/api/whatif` | POST | Run scenario simulation |

Example API call:
```bash
curl http://localhost:5000/api/predictions
```

---

## 🧠 Model Methodology

### Features Used
1. **Past Total Medals** - Previous Olympic performance (4-year lag)
2. **GDP (USD)** - Gross Domestic Product in US dollars
3. **Population** - Total country population
4. **GDP per Capita** - Economic development indicator

### Why These Features?
- **Past Performance**: Strongest predictor of future success
- **GDP**: Reflects investment in sports infrastructure
- **Population**: Larger talent pool for athlete selection
- **GDP per Capita**: Economic development affects sports participation

### Model Validation
- Trained on historical data through 2024
- Validated using cross-validation
- Residual analysis confirms no systematic bias

---

## 🎓 Interview Talking Points

### Technical Depth
> "I implemented a Gradient Boosting Regressor with hyperparameter optimization, achieving an R² of ~0.85. The model uses SHAP values for interpretability, allowing us to understand exactly why each prediction is made."

### Business Impact
> "The What-If simulator enables sports federations to understand how economic investments could impact Olympic performance, providing actionable insights for strategic planning."

### Full-Stack Skills
> "This is a complete end-to-end solution: Python data pipeline, scikit-learn ML model, Flask REST API, and a modern responsive web interface with TailwindCSS."

### XAI (Explainable AI)
> "Model interpretability was crucial - I used SHAP values and permutation importance to ensure stakeholders can trust and understand the predictions."

---

## 🛣️ Roadmap

- [x] Core ML model with Gradient Boosting
- [x] Interactive dashboard with TailwindCSS
- [x] SHAP explainability integration
- [x] What-If scenario simulator
- [x] RESTful API
- [ ] Time-series forecasting with ARIMA/LSTM
- [ ] Sport-specific medal predictions
- [ ] Historical trends analysis
- [ ] Mobile app companion

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Olympic Games data sources
- scikit-learn documentation and community
- Flask framework team
- TailwindCSS for the beautiful UI

---

## 📧 Contact

For questions or feedback, please reach out:
- Email: Blessatwork@gmail.com & mahasenihazel@gmail.com

---

<p align="center">
  <strong>Built with ❤️ and Machine Learning</strong><br>
  <em>Predicting the future of Olympic sports</em>
</p>
