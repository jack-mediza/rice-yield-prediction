# 🌾 Multi-Factor Prediction of Rice Yield Using Agronomical Factors

A complete AI/ML web application that predicts rice crop yield from
agronomical data using multiple machine learning regression models.

---

## 📌 Features

- Upload CSV or XLSX agricultural datasets
- Automatic dataset inspection and validation
- Interactive Exploratory Data Analysis with charts
- Trains 4 regression models: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- 5-fold cross-validation (no data leakage)
- Hyperparameter tuning (RandomizedSearchCV) for RF and XGBoost
- Automatic best-model selection by R² score
- Saves trained model, preprocessor, and metadata
- Manual yield prediction form (dynamically generated from model features)
- Feature importance visualization
- Dashboard with KPI cards and charts

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3 + Flask |
| ML | scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Frontend | HTML5, Bootstrap 5, Chart.js |
| Model saving | joblib |
| Excel support | openpyxl |

---

## 📂 Project Structure

```
rice-yield-prediction/
│
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── dataset/
│   ├── create_synthetic.py ← Script to generate test data
│   └── synthetic_rice_dataset.csv
│
├── uploads/                ← Your uploaded datasets go here
│
├── model/                  ← Saved model files (created after training)
│   ├── best_model.pkl
│   ├── preprocessing.pkl
│   └── metadata.json
│
├── training/
│   └── train_model.py      ← Full ML pipeline
│
├── templates/              ← HTML pages
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── analysis.html
│   ├── training.html
│   ├── training_status.html
│   ├── prediction.html
│   ├── results.html
│   ├── dashboard.html
│   ├── about.html
│   └── error.html
│
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## 🚀 Installation (Windows)

### Step 1 — Install Python 3

Download from https://www.python.org/downloads/ and install.
Make sure to tick **"Add Python to PATH"** during setup.

### Step 2 — Open Command Prompt

Press `Win + R`, type `cmd`, press Enter.

### Step 3 — Navigate to project folder

```cmd
cd C:\path\to\rice-yield-prediction
```

Replace `C:\path\to\` with the actual folder path.

### Step 4 — Create a virtual environment

```cmd
python -m venv venv
```

### Step 5 — Activate the virtual environment

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the start of the prompt.

### Step 6 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 7 — Generate the synthetic dataset (first run only)

```cmd
python dataset\create_synthetic.py
```

---

## ▶️ Run the Application

```cmd
python app.py
```

---

## 🌐 Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 Test Procedure

1. Open http://127.0.0.1:5000 — home page should load
2. Click **"Try with Demo Data"** or **Upload Dataset** → upload your CSV/XLSX
3. Click **"Data Analysis"** in the navbar — charts should appear
4. Click **"Train Models"** → select target column → click Start Training
5. Wait for training to complete (progress shown live)
6. Click **"View Results"** — model comparison table and charts appear
7. Click **"Predict Yield"** — fill in values, click Predict
8. Click **"Dashboard"** — KPI cards should show actual values

---

## 📊 Using Your Own Dataset

Replace the synthetic dataset at any time:

1. Go to **Upload Dataset** in the navigation bar
2. Upload your CSV or XLSX file
3. The system will automatically inspect columns and detect the yield column
4. Proceed to **Train Models** and select your actual yield column

Your dataset should have:
- One row per field/crop record
- Agricultural feature columns (numeric values)
- A target yield column (e.g., `Rice_Yield_t_ha`, `yield`, `production`)

---

## ⚠️ Synthetic Data Notice

The built-in synthetic dataset is for **testing and demonstration only**.
It does NOT represent real agricultural measurements.
Model results from synthetic data should NOT be used for real farming decisions.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 5000 in use | Change `port=5000` to `port=5001` in app.py |
| File upload fails | Check file is .csv or .xlsx and under 50 MB |
| Training very slow | Dataset may be large; reduce rows or disable tuning |
| Prediction form empty | Train models first before predicting |

---

## 📝 Important Notes

- This is a **100% software project** — no hardware or IoT involved
- All ML is done with Python (scikit-learn, XGBoost)
- The test set is never used for training or tuning (no data leakage)
- Models are saved as `.pkl` files and loaded for prediction (no retraining)
