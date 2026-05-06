# Customer Churn Prediction Model

> Predict which customers will cancel, before they do.

## What It Does

This project builds an end-to-end machine learning system to predict customer churn — identifying customers likely to cancel their subscription or service. It includes:

- **Jupyter exploratory analysis** — understand churn patterns in your data
- **Multiple ML models** — logistic regression, random forest, gradient boosting
- **Model evaluation** — accuracy, precision, recall, AUC-ROC
- **Streamlit UI** — upload customer data, get churn predictions instantly
- **Batch prediction** — predict churn for entire customer databases

Typical use case: You have 10,000 customers. This model identifies the 500 most likely to churn. You then reach out proactively before they leave.

## Architecture

```
┌──────────────┐
│   CSV Data   │
│  (customers) │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│  Jupyter Exploration         │
│  - EDA                       │
│  - Feature engineering       │
│  - Train/test split          │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Model Training              │
│  - Logistic Regression       │
│  - Random Forest             │
│  - XGBoost / Gradient Boost  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Model Serialization         │
│  - Save best model (pickle)  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Streamlit UI                │
│  - Upload customer CSV       │
│  - Get churn predictions     │
│  - View probability scores   │
│  - Export results            │
└──────────────────────────────┘
```

## Tech Stack

- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Jupyter:** Exploratory Data Analysis

## Dataset

Expected CSV format for predictions:
```
customer_id, monthly_charges, tenure_months, total_charges, contract_type, internet_service
1001,65.50,12,786,Month-to-month,DSL
1002,89.00,24,2136,One year,Fiber
...
```

The model expects features like:
- `monthly_charges` — customer's monthly bill
- `tenure_months` — how long they've been a customer
- `total_charges` — lifetime spend
- `contract_type` — month-to-month, 1-year, 2-year
- `internet_service` — DSL, Fiber, No internet

## Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip

### Installation

```bash
git clone https://github.com/Aamna-s/customer-churn-prediction-model.git
cd customer-churn-prediction-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Jupyter Notebook

```bash
jupyter notebook
# Open notebook_exploration.ipynb
# Run cells to train and evaluate models
```

### Running the Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

## Usage

### 1. Explore Data in Jupyter
```bash
jupyter notebook notebook_exploration.ipynb
```

The notebook includes:
- Data loading and cleaning
- Exploratory Data Analysis (EDA) — churn rate, feature distributions
- Feature engineering — scaling, encoding categorical variables
- Train/test split
- Model training and comparison
- Evaluation metrics

### 2. Make Predictions via Streamlit UI
```bash
streamlit run app.py
```

In the UI:
1. Upload a CSV file with customer data
2. Click "Predict"
3. View results:
   - Churn probability for each customer
   - Top 10 highest-risk customers
   - Overall churn rate estimate
4. Download results as CSV

**Example:**
```
Input: customers.csv (1000 customers)
Output: predictions.csv
  ├── customer_id
  ├── churn_probability (0.0–1.0)
  ├── churn_risk (Low / Medium / High)
  └── recommended_action (Contact, Monitor, etc.)
```

### 3. Batch Prediction (Command Line)
```python
import pickle
import pandas as pd

# Load trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load customer data
customers = pd.read_csv('customers.csv')

# Predict
predictions = model.predict_proba(customers)[:, 1]  # Churn probability
customers['churn_probability'] = predictions

# Save results
customers.to_csv('churn_predictions.csv', index=False)
```

## Model Performance

| Model | Accuracy | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 0.81 | 0.72 | 0.68 | 0.85 |
| Random Forest | 0.85 | 0.78 | 0.74 | 0.89 |
| **XGBoost** | **0.87** | **0.81** | **0.76** | **0.91** |

**Best model:** XGBoost — balances accuracy and recall for identifying at-risk customers.

## Key Results

- **87% accuracy** — correctly identifies churn vs. no-churn customers
- **81% precision** — when model says "will churn," it's right 81% of the time
- **76% recall** — catches 76% of actual churners (vs. missing 24%)
- **Production ready** — serialized model, Streamlit deployment

## What I'd Do Differently (If Rebuilding)

If I rebuilt this today, I'd:

1. **Handle imbalanced data explicitly** — Churn is often rare (5-10%). Would use SMOTE oversampling or class weighting from the start
2. **Add SHAP explainability** — Show which features drive each prediction, not just overall importance
3. **Implement real-time monitoring** — Track model performance drift; retrain when accuracy drops below threshold
4. **Add retention action recommendations** — Not just "customer will churn" but "try discount" or "escalate to support"
5. **Build A/B testing framework** — Measure if targeting high-risk customers actually reduces churn

**Why I didn't add these:** The core goal was a working churn prediction pipeline. Explainability and monitoring are important in production, but not critical for the MVP.

## Project Learnings

- **Class imbalance matters** — churn is often rare (5–10%), so need careful metric selection
- **Business context is critical** — precision vs. recall trade-off depends on cost of false positives/negatives
- **Feature engineering is 80% of the work** — raw data rarely tells the full story
- **Model decay is real** — model trained 6 months ago won't perform as well on new data

## File Structure

```
customer-churn-prediction-model/
├── README.md
├── requirements.txt
├── notebook_exploration.ipynb         # EDA and model training
├── app.py                             # Streamlit app
├── models/
│   ├── best_model.pkl                 # Serialized trained model
│   └── preprocessing.pkl              # Scaler/encoder for features
├── data/
│   ├── raw/
│   │   └── customers.csv              # Sample data
│   └── processed/
│       └── churn_train.csv            # Cleaned training data
└── utils/
    ├── data_loader.py                 # CSV loading and validation
    ├── preprocessing.py               # Feature scaling, encoding
    └── model.py                       # Training and prediction logic
```

## Contributing

Found a bug? Have a suggestion? Open an issue or PR.

## License

MIT

---

**Questions?** Reach out: [linkedin.com/in/aamna-siddique](https://linkedin.com/in/aamna-siddique)
