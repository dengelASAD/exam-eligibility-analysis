# Exam Eligibility Analysis Machine Learning System

A comprehensive machine learning system designed for educational institutions to analyze student performance data, predict exam eligibility, and estimate final exam scores using a variety of predictive models.

## Features

- **Classification Mode**: Predicts if a student is "Allowed" or "Not Allowed" to take the exam. Evaluates 7 models:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Artificial Neural Network (ANN)
  - Support Vector Machine (SVM)
- **Regression Mode**: Predicts the "Final Exam Score" of students. Evaluates 5 models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Artificial Neural Network (MLP Regressor)
- **Robust Evaluation**: Uses an 80/20 train/test split to ensure honest model evaluation and prevent data leakage.
- **Dynamic Visualizations**: Utilizes Chart.js for real-time model leaderboard comparisons and confusion matrices.
- **Data Validation**: Enforces strict data quality checks before processing any CSV file.

## Project Structure

```text
├── app.py                # Main Flask backend application (ML models & APIs)
├── requirements.txt      # Python dependencies with specific versions
├── .env / .env.example   # Environment configuration variables
├── index.html            # Main frontend user interface
├── script.js             # Frontend logic, API communication, Chart.js rendering
├── style.css             # Responsive styling for the UI
└── chart.js              # Local Chart.js library file
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd exam-eligibility-analysis
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Copy the example environment file and configure it if needed.
   ```bash
   cp .env.example .env
   ```

## Usage

1. **Run the backend server**:
   ```bash
   python app.py
   ```
   The server will start on `http://127.0.0.1:5000` (or as configured in `.env`).

2. **Open the Application**:
   Simply navigate to `http://127.0.0.1:5000` in your web browser. 

3. **Analyze Data**:
   - Select either **Classification** or **Regression** mode.
   - Upload a valid CSV file.
   - View the generated charts, model leaderboards, and accuracy metrics.

## Dataset Format Requirements

The uploaded CSV file must contain the following required columns:

| Column Name | Data Type | Valid Range | Meaning |
|---|---|---|---|
| `Student_ID` | String/Int | Any | Unique identifier for the student |
| `Attendance_Rate` | Numeric | 0 to 100 | Percentage of classes attended |
| `Final_Exam_Score` | Numeric | 0 to 100 | The student's final score |

*Note: The dataset must contain at least 10 valid rows for the machine learning models to train successfully.*

## API Documentation

### `GET /health`
Returns the operational status of the server.
**Response (200 OK)**:
```json
{
  "status": "ready"
}
```

### `POST /upload`
Uploads a CSV file and runs the specified machine learning mode.
**Form Data**:
- `file`: The CSV file
- `mode`: `"classification"` or `"regression"`

**Success Response (200 OK)**:
Returns JSON containing model metrics, leaderboards, and confusion matrices.

**Error Responses**:
- `400 Bad Request`: Returned if the CSV is missing, improperly formatted, or fails validation (e.g., missing columns).
- `500 Internal Server Error`: Unexpected server-side failures.

## Troubleshooting

- **"Connection Error"**: Ensure the Flask server is running and your `.env` file's `ALLOWED_ORIGINS` includes your frontend URL.
- **"Invalid CSV format"**: Check that your CSV is delimited properly and does not contain corrupted rows.
- **"Validation Error"**: Ensure your CSV contains the exact column names specified in the dataset format requirements.
