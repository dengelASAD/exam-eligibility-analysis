import os
import sys
import traceback
import logging
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Load Environment Variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = ['Student_ID', 'Attendance_Rate', 'Final_Exam_Score']
DATA_CONSTRAINTS = {
    'Attendance_Rate': {'min': 0, 'max': 100},
    'Final_Exam_Score': {'min': 0, 'max': 100}
}
MIN_SAMPLES = 10

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')

# Configure CORS
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', '')
if allowed_origins_str:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]
    CORS(app, resources={r"/upload": {"origins": allowed_origins}})
else:
    CORS(app) # Fallback if not configured

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

@app.route('/')
def index():
    return app.send_static_file('index.html')

def validate_dataframe(df):
    """
    Validates the uploaded dataframe against required columns and constraints.
    Returns (True, "") if valid, (False, error_message) if invalid.
    """
    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # 2. Check sample size
    if len(df) < MIN_SAMPLES:
        return False, f"Dataset must contain at least {MIN_SAMPLES} samples."
    
    # 3. Check data ranges
    for col, constraints in DATA_CONSTRAINTS.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column {col} must be numeric."
        
        if df[col].min() < constraints['min'] or df[col].max() > constraints['max']:
            return False, f"Column {col} values must be between {constraints['min']} and {constraints['max']}."
            
    return True, ""

def train_classification_model(df):
    """
    Trains multiple classification models using an 80/20 train/test split
    and returns evaluation metrics based strictly on the test set.
    """
    logger.info(f"Training classification models with {len(df)} samples")
    
    # Preprocessing
    df["Allowed_Exam"] = np.where(df["Attendance_Rate"] >= 75, "Allowed", "Not Allowed")
    X = df.drop(columns=["Student_ID", "Pass_Fail", "Allowed_Exam", "Attendance_Rate", "Final_Exam_Score"], errors='ignore')
    y = df["Allowed_Exam"]
    
    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Test Split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    logger.info(f"Train/Test split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    models = {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "ANN": MLPClassifier(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    all_results = {}
    try:
        pos_label = int(target_encoder.transform(["Allowed"])[0])
    except:
        pos_label = 1

    for name, clf in models.items():
        # Train on X_train, y_train
        clf.fit(X_train, y_train)
        
        # Predict on X_test ONLY
        y_pred = clf.predict(X_test)
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4)
        }
        cm = confusion_matrix(y_test, y_pred)
        
        all_results[name] = {
            "metrics": metrics,
            "cm": cm.tolist()
        }
        logger.info(f"Model {name} evaluated. Accuracy: {metrics['accuracy']}")
    
    return X.columns.tolist(), all_results

def train_regression_models(df):
    """
    Trains multiple regression models using an 80/20 train/test split
    and returns evaluation metrics based strictly on the test set.
    """
    logger.info(f"Training regression models with {len(df)} samples")
    
    X = df.drop(columns=["Student_ID", "Pass_Fail", "Final_Exam_Score", "Allowed_Exam"], errors='ignore')
    y = df["Final_Exam_Score"]
    
    # Encode
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train/Test split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, reg_model in models.items():
        # Train on X_train, y_train
        reg_model.fit(X_train, y_train)
        
        # Predict on X_test ONLY
        y_pred = reg_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate SSE safely handling both pandas Series and numpy arrays
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        sse = np.sum((y_test_array - y_pred)**2)
        
        results[name] = {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "SSE": round(float(sse), 4)
        }
        logger.info(f"Model {name} evaluated. RMSE: {results[name]['RMSE']}")
        
    return results

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Upload request received")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename uploaded")
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    mode = request.form.get('mode', 'classification')
    
    try:
        # Read CSV
        try:
            df = pd.read_csv(file)
        except pd.errors.ParserError as pe:
            logger.error(f"Failed to parse CSV: {pe}", exc_info=True)
            return jsonify({"error": "Invalid CSV file format. Please ensure it is properly delimited."}), 400
            
        # Validate DataFrame
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            logger.warning(f"Data validation failed: {error_msg}")
            return jsonify({"error": error_msg}), 400
            
        # Process based on mode
        if mode == 'classification':
            features, all_classification_results = train_classification_model(df)
            return jsonify({
                "mode": "classification",
                "total": len(df),
                "results": all_classification_results,
                "sample": []
            })
        elif mode == 'regression':
            reg_results = train_regression_models(df)
            return jsonify({
                "mode": "regression",
                "results": reg_results
            })
        else:
            return jsonify({"error": f"Unknown mode: {mode}"}), 400
            
    except ValueError as ve:
        logger.error(f"Value Error during processing: {ve}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready"}), 200

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting Flask app. Host: {host}, Port: {port}, Debug: {debug_mode}")
    app.run(host=host, port=port, debug=debug_mode)
