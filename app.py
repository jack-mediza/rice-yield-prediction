"""
Rice Yield Prediction — Flask Application
Multi-Factor Prediction of Rice Yield Using Agronomical Factors
"""
import os
import json
import traceback
import threading

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, session)
from werkzeug.utils import secure_filename

# ── App config ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = 'rice_yield_pred_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MODEL_DIR'] = os.path.join(BASE_DIR, 'model')
app.config['DATASET_DIR'] = os.path.join(BASE_DIR, 'dataset')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

for d in [app.config['UPLOAD_FOLDER'], app.config['MODEL_DIR']]:
    os.makedirs(d, exist_ok=True)

# ── Global training state ─────────────────────────────────────────────────────
training_state = {
    'running': False,
    'progress': [],
    'done': False,
    'error': None,
    'results': None,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset():
    """Load the current uploaded dataset from session path."""
    path = session.get('dataset_path')
    if not path or not os.path.exists(path):
        # Try synthetic
        syn = os.path.join(app.config['DATASET_DIR'], 'synthetic_rice_dataset.csv')
        if os.path.exists(syn):
            return pd.read_csv(syn), syn, True
        return None, None, False
    try:
        if path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path), path, False
        return pd.read_csv(path), path, False
    except Exception:
        return None, None, False


def load_metadata():
    path = os.path.join(app.config['MODEL_DIR'], 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def model_trained():
    return os.path.exists(os.path.join(app.config['MODEL_DIR'], 'best_model.pkl'))


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    meta = load_metadata()
    return render_template('index.html', meta=meta, trained=model_trained())


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)

        file = request.files['dataset']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV or XLSX file.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Validate
        try:
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(save_path)
            else:
                df = pd.read_csv(save_path)
        except Exception as e:
            flash(f'Could not read file: {e}', 'danger')
            os.remove(save_path)
            return redirect(request.url)

        if df.empty:
            flash('Uploaded file is empty.', 'danger')
            os.remove(save_path)
            return redirect(request.url)

        if len(df) < 10:
            flash('Dataset is too small (minimum 10 rows required).', 'danger')
            os.remove(save_path)
            return redirect(request.url)

        session['dataset_path'] = save_path
        session['dataset_name'] = filename
        flash(f'Dataset "{filename}" uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)', 'success')
        return redirect(url_for('analysis'))

    # GET — show form + dataset info if already loaded
    df, path, is_synthetic = load_dataset()
    info = None
    if df is not None:
        from training.train_model import inspect_dataset
        info = inspect_dataset(df)
        info['is_synthetic'] = is_synthetic
        info['filename'] = session.get('dataset_name', 'synthetic_rice_dataset.csv' if is_synthetic else 'unknown')
    return render_template('upload.html', info=info)


@app.route('/use_synthetic')
def use_synthetic():
    """Use the built-in synthetic dataset."""
    syn = os.path.join(app.config['DATASET_DIR'], 'synthetic_rice_dataset.csv')
    if not os.path.exists(syn):
        flash('Synthetic dataset not found. Please restart the app.', 'danger')
        return redirect(url_for('upload'))
    session['dataset_path'] = syn
    session['dataset_name'] = 'synthetic_rice_dataset.csv'
    flash('Using synthetic development dataset. Results are for testing only.', 'warning')
    return redirect(url_for('analysis'))


@app.route('/analysis')
def analysis():
    df, path, is_synthetic = load_dataset()
    if df is None:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('upload'))

    from training.train_model import inspect_dataset
    info = inspect_dataset(df)
    info['is_synthetic'] = is_synthetic
    info['filename'] = session.get('dataset_name', 'Dataset')

    # Prepare chart data
    numeric_cols = info['numeric_cols']
    chart_data = {}
    for col in numeric_cols:
        series = df[col].dropna()
        chart_data[col] = {
            'values': [round(float(v), 4) for v in series.tolist()[:500]],
            'mean': round(float(series.mean()), 4),
            'std': round(float(series.std()), 4),
        }

    # Correlation matrix
    corr_matrix = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        corr_matrix = {
            'labels': list(corr.columns),
            'values': corr.values.tolist(),
        }

    # Yield distribution (detect target)
    from training.train_model import detect_target_column
    target_col = detect_target_column(df)

    return render_template('analysis.html',
                           info=info,
                           chart_data=json.dumps(chart_data),
                           corr_matrix=json.dumps(corr_matrix) if corr_matrix else 'null',
                           target_col=target_col,
                           numeric_cols=numeric_cols,
                           trained=model_trained())


@app.route('/training', methods=['GET', 'POST'])
def training():
    global training_state
    df, path, is_synthetic = load_dataset()
    if df is None:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('upload'))

    from training.train_model import detect_target_column, inspect_dataset
    info = inspect_dataset(df)
    suggested_target = info['suggested_target']
    numeric_cols = info['numeric_cols']

    if request.method == 'POST':
        target_col = request.form.get('target_col', suggested_target)
        if target_col not in df.columns:
            flash(f'Target column "{target_col}" not found.', 'danger')
            return redirect(request.url)

        # Reset state
        training_state = {
            'running': True, 'progress': [], 'done': False,
            'error': None, 'results': None,
        }

        # Capture path now while we're in request context
        dataset_path = session.get('dataset_path')
        is_syn = False
        if not dataset_path or not os.path.exists(dataset_path):
            syn = os.path.join(app.config['DATASET_DIR'], 'synthetic_rice_dataset.csv')
            if os.path.exists(syn):
                dataset_path = syn
                is_syn = True

        def run_training(dpath, tgt):
            global training_state
            try:
                from training.train_model import preprocess_and_train
                if dpath.endswith(('.xlsx', '.xls')):
                    import pandas as pd
                    df_fresh = pd.read_excel(dpath)
                else:
                    import pandas as pd
                    df_fresh = pd.read_csv(dpath)
                results = preprocess_and_train(
                    df_fresh, tgt,
                    progress_callback=lambda msg: training_state['progress'].append(msg)
                )
                training_state['results'] = results
                training_state['done'] = True
                training_state['running'] = False
            except Exception as e:
                import traceback
                training_state['error'] = str(e)
                training_state['done'] = True
                training_state['running'] = False

        t = threading.Thread(target=run_training, args=(dataset_path, target_col), daemon=True)
        t.start()
        return redirect(url_for('training_status'))

    return render_template('training.html',
                           suggested_target=suggested_target,
                           numeric_cols=numeric_cols,
                           trained=model_trained(),
                           meta=load_metadata())


@app.route('/training/status')
def training_status():
    return render_template('training_status.html',
                           trained=model_trained(),
                           meta=load_metadata())


@app.route('/api/training_progress')
def training_progress():
    global training_state
    return jsonify({
        'running': training_state['running'],
        'progress': training_state['progress'],
        'done': training_state['done'],
        'error': training_state['error'],
        'has_results': training_state['results'] is not None,
    })


@app.route('/results')
def results():
    if not model_trained():
        flash('Please train the models first.', 'warning')
        return redirect(url_for('training'))

    meta = load_metadata()
    df, _, is_synthetic = load_dataset()

    # Prepare chart data for results
    model_results = meta.get('all_model_results', {})
    chart_labels = list(model_results.keys())
    mae_vals = [model_results[m]['MAE'] for m in chart_labels]
    rmse_vals = [model_results[m]['RMSE'] for m in chart_labels]
    r2_vals = [model_results[m]['R2'] for m in chart_labels]

    fi = meta.get('feature_importance', {})
    # Limit top 15
    fi = dict(list(fi.items())[:15])

    cv_results = meta.get('cv_results', {})

    return render_template('results.html',
                           meta=meta,
                           model_results=model_results,
                           chart_labels=json.dumps(chart_labels),
                           mae_vals=json.dumps(mae_vals),
                           rmse_vals=json.dumps(rmse_vals),
                           r2_vals=json.dumps(r2_vals),
                           feature_importance=json.dumps(fi),
                           cv_results=cv_results,
                           is_synthetic=is_synthetic,
                           trained=True)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if not model_trained():
        flash('Please train the models first.', 'warning')
        return redirect(url_for('training'))

    meta = load_metadata()
    feature_cols = meta.get('feature_cols', [])
    target_col = meta.get('target_col', 'Rice Yield')

    # Build form field info
    df, _, _ = load_dataset()
    field_info = {}
    if df is not None:
        for col in feature_cols:
            if col in df.columns:
                series = df[col].dropna()
                if series.dtype in [np.float64, np.int64, float, int]:
                    field_info[col] = {
                        'type': 'number',
                        'min': round(float(series.min()), 4),
                        'max': round(float(series.max()), 4),
                        'mean': round(float(series.mean()), 4),
                        'step': '0.01',
                    }
                else:
                    unique_vals = series.unique().tolist()[:50]
                    field_info[col] = {'type': 'select', 'options': unique_vals}

    prediction_result = None
    error_msg = None

    if request.method == 'POST':
        input_data = {}
        try:
            for col in feature_cols:
                val = request.form.get(col, '')
                if field_info.get(col, {}).get('type') == 'number':
                    fval = safe_float(val)
                    if fval is None:
                        raise ValueError(f"Invalid value for '{col}': '{val}'")
                    input_data[col] = fval
                else:
                    input_data[col] = val

            from training.train_model import predict
            pred = predict(input_data)
            prediction_result = round(pred, 4)
        except Exception as e:
            error_msg = str(e)

    return render_template('prediction.html',
                           meta=meta,
                           feature_cols=feature_cols,
                           field_info=field_info,
                           target_col=target_col,
                           prediction_result=prediction_result,
                           error_msg=error_msg,
                           trained=True)


@app.route('/dashboard')
def dashboard():
    meta = load_metadata()
    df, _, is_synthetic = load_dataset()
    info = None
    if df is not None:
        from training.train_model import inspect_dataset
        info = inspect_dataset(df)

    return render_template('dashboard.html',
                           meta=meta,
                           info=info,
                           is_synthetic=is_synthetic,
                           trained=model_trained())


@app.route('/about')
def about():
    return render_template('about.html', trained=model_trained())


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 50 MB.', 'danger')
    return redirect(url_for('upload'))


@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, msg='Page not found.'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, msg='Internal server error.'), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Rice Yield Prediction App")
    print("URL: http://127.0.0.1:5000")
    print("=" * 60)
    # Auto-load synthetic dataset into session on first run is handled
    # by load_dataset() falling back to synthetic
    app.run(debug=True, host='0.0.0.0', port=5000)
