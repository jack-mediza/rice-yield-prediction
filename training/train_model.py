"""
Rice Yield Prediction — ML Training Pipeline
Handles preprocessing, feature selection, model training,
cross-validation, hyperparameter tuning, and model saving.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def detect_target_column(df):
    """Detect the most likely yield/target column."""
    # Prioritise longer/more-specific matches first
    yield_keywords = [
        'rice_yield', 'crop_yield', 'yield_t_ha', 'yield_kg_ha', 'total_yield',
        'yield', 'production', 'productivity', 'output', 'harvest',
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for kw in yield_keywords:
        for col in df.columns:
            if kw in col.lower().replace(' ', '_'):
                return col
    # fallback: last numeric column
    return numeric_cols[-1] if numeric_cols else None


def inspect_dataset(df):
    """Return a structured inspection report."""
    report = {
        'n_rows': int(len(df)),
        'n_cols': int(len(df.columns)),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing': {col: int(df[col].isna().sum()) for col in df.columns},
        'missing_pct': {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns},
        'duplicates': int(df.duplicated().sum()),
        'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': list(df.select_dtypes(exclude=[np.number]).columns),
        'suggested_target': detect_target_column(df),
        'stats': {}
    }
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        desc = numeric_df.describe().round(4)
        report['stats'] = desc.to_dict()
    return report


def preprocess_and_train(df, target_col, progress_callback=None):
    """
    Full ML pipeline:
    1. Validate and clean data
    2. Split into train/test
    3. Build preprocessing pipeline
    4. Feature selection
    5. Train 4 models
    6. Cross-validation
    7. Hyperparameter tuning
    8. Select best model
    9. Save artifacts
    Returns results dict.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    results = {'steps': [], 'errors': []}

    # --- 1. Validate ---
    log("Step 1: Validating data...")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df.drop_duplicates()
    log(f"  Removed duplicates. Remaining rows: {len(df)}")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    log(f"  Dropped rows with missing target. Remaining: {len(df)}")

    if len(df) < 20:
        raise ValueError("Dataset too small after cleaning (< 20 rows).")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log(f"  Numeric features: {numeric_features}")
    log(f"  Categorical features: {categorical_features}")
    log(f"  Target: {target_col}, range [{y.min():.3f}, {y.max():.3f}]")

    # --- 2. Train/Test Split ---
    log("Step 2: Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # --- 3. Preprocessing Pipeline ---
    log("Step 3: Building preprocessing pipeline...")
    transformers = []
    if numeric_features:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_pipeline, numeric_features))

    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # Fit on training data only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except Exception:
        feature_names_out = [f'feature_{i}' for i in range(X_train_proc.shape[1])]

    log(f"  Preprocessed shape: {X_train_proc.shape}")

    # --- 4. Feature Selection (correlation-based info) ---
    log("Step 4: Computing feature importance via correlation...")
    correlations = {}
    if numeric_features:
        for feat in numeric_features:
            valid_mask = X_train[feat].notna()
            if valid_mask.sum() > 5:
                corr = float(np.corrcoef(X_train.loc[valid_mask, feat], y_train[valid_mask])[0, 1])
                if not np.isnan(corr):
                    correlations[feat] = round(corr, 4)

    # --- 5. Train Models ---
    log("Step 5: Training models...")

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    model_results = {}
    trained_models = {}

    for name, model in models.items():
        log(f"  Training {name}...")
        try:
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            model_results[name] = {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}
            trained_models[name] = model
            log(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        except Exception as e:
            log(f"    ERROR training {name}: {e}")
            results['errors'].append(f"{name}: {str(e)}")

    # --- 6. Cross-Validation ---
    log("Step 6: Cross-validation (5-fold, on training data)...")
    cv_results = {}
    for name, model in trained_models.items():
        try:
            scores = cross_val_score(
                model, X_train_proc, y_train, cv=5, scoring='r2', n_jobs=-1
            )
            cv_results[name] = {
                'cv_r2_mean': round(float(scores.mean()), 4),
                'cv_r2_std': round(float(scores.std()), 4),
                'cv_r2_scores': [round(float(s), 4) for s in scores],
            }
            log(f"  {name} CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            log(f"  CV error for {name}: {e}")

    # --- 7. Hyperparameter Tuning (RF and XGBoost) ---
    log("Step 7: Hyperparameter tuning (RF and XGBoost)...")
    tuned_models = {}

    # Random Forest tuning
    try:
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid, n_iter=6, cv=3, scoring='r2',
            random_state=42, n_jobs=-1
        )
        rf_search.fit(X_train_proc, y_train)
        best_rf = rf_search.best_estimator_
        y_pred_rf = best_rf.predict(X_test_proc)
        tuned_mae = float(mean_absolute_error(y_test, y_pred_rf))
        tuned_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        tuned_r2 = float(r2_score(y_test, y_pred_rf))
        model_results['Random Forest (Tuned)'] = {
            'MAE': round(tuned_mae, 4), 'RMSE': round(tuned_rmse, 4), 'R2': round(tuned_r2, 4)
        }
        trained_models['Random Forest (Tuned)'] = best_rf
        tuned_models['Random Forest'] = rf_search.best_params_
        log(f"  Tuned RF → R²={tuned_r2:.4f}, params={rf_search.best_params_}")
    except Exception as e:
        log(f"  RF tuning error: {e}")

    # XGBoost tuning
    try:
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, verbosity=0),
            xgb_param_grid, n_iter=6, cv=3, scoring='r2',
            random_state=42, n_jobs=-1
        )
        xgb_search.fit(X_train_proc, y_train)
        best_xgb = xgb_search.best_estimator_
        y_pred_xgb = best_xgb.predict(X_test_proc)
        tuned_mae = float(mean_absolute_error(y_test, y_pred_xgb))
        tuned_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
        tuned_r2 = float(r2_score(y_test, y_pred_xgb))
        model_results['XGBoost (Tuned)'] = {
            'MAE': round(tuned_mae, 4), 'RMSE': round(tuned_rmse, 4), 'R2': round(tuned_r2, 4)
        }
        trained_models['XGBoost (Tuned)'] = best_xgb
        tuned_models['XGBoost'] = xgb_search.best_params_
        log(f"  Tuned XGB → R²={tuned_r2:.4f}, params={xgb_search.best_params_}")
    except Exception as e:
        log(f"  XGB tuning error: {e}")

    # --- 8. Select Best Model ---
    log("Step 8: Selecting best model...")
    best_name = max(model_results, key=lambda k: model_results[k]['R2'])
    best_metrics = model_results[best_name]
    best_model = trained_models[best_name]
    log(f"  Best model: {best_name} | R²={best_metrics['R2']}, MAE={best_metrics['MAE']}, RMSE={best_metrics['RMSE']}")

    # --- 9. Feature Importance ---
    log("Step 9: Computing feature importance...")
    feature_importance = {}
    try:
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feat_imp = dict(zip(feature_names_out, importances.tolist()))
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
            feature_importance = {k: round(float(v), 6) for k, v in feat_imp.items()}
        elif hasattr(best_model, 'coef_'):
            coefs = np.abs(best_model.coef_)
            feat_imp = dict(zip(feature_names_out, coefs.tolist()))
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
            feature_importance = {k: round(float(v), 6) for k, v in feat_imp.items()}
    except Exception as e:
        log(f"  Feature importance error: {e}")

    # --- 10. Save Artifacts ---
    log("Step 10: Saving model artifacts...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessing.pkl'))

    metadata = {
        'target_col': target_col,
        'feature_cols': list(X.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_names_out': list(feature_names_out),
        'best_model_name': best_name,
        'best_metrics': best_metrics,
        'all_model_results': model_results,
        'cv_results': cv_results,
        'correlations': correlations,
        'feature_importance': feature_importance,
        'tuned_params': tuned_models,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'target_min': float(y.min()),
        'target_max': float(y.max()),
        'target_mean': float(y.mean()),
    }

    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    log("TRAINING COMPLETE!")
    results['model_results'] = model_results
    results['cv_results'] = cv_results
    results['best_model_name'] = best_name
    results['best_metrics'] = best_metrics
    results['feature_importance'] = feature_importance
    results['correlations'] = correlations
    results['metadata'] = metadata
    return results


def predict(input_data: dict):
    """
    Load saved model and predict.
    input_data: {feature_name: value, ...}
    Returns predicted yield value.
    """
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessing.pkl'))
    with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_cols']
    input_df = pd.DataFrame([input_data], columns=feature_cols)

    # Convert to numeric where possible
    for col in metadata['numeric_features']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    input_proc = preprocessor.transform(input_df)
    prediction = model.predict(input_proc)
    return float(prediction[0])


if __name__ == '__main__':
    # Quick test with synthetic data
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'dataset', 'synthetic_rice_dataset.csv'))
    target = detect_target_column(df)
    print(f"Detected target: {target}")
    results = preprocess_and_train(df, target)
    print("\nModel Results:")
    for name, metrics in results['model_results'].items():
        print(f"  {name}: {metrics}")
    print(f"\nBest: {results['best_model_name']}")
