from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import time

# Prometheus imports
from prometheus_client import Counter, Histogram, generate_latest

# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

# -----------------------------------
# Sample Dataset
# Experience vs Salary
# -----------------------------------

X = np.array([[1], [2], [3], [4], [5], [6]])

y = np.array([
    30000,
    40000,
    50000,
    60000,
    70000,
    80000
])

# -----------------------------------
# Train Linear Regression Model
# -----------------------------------

model = LinearRegression()

model.fit(X, y)

# -----------------------------------
# Prometheus Metrics
# -----------------------------------

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction'
)

# -----------------------------------
# Home Route
# -----------------------------------

@app.route('/')
def home():
    return "ML Monitoring API Running"

# -----------------------------------
# Prediction Route
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    REQUEST_COUNT.inc()

    start_time = time.time()

    try:

        data = request.get_json()

        years = data['years_experience']

        prediction = model.predict([[years]])

        latency = time.time() - start_time

        PREDICTION_LATENCY.observe(latency)

        return jsonify({
            'years_experience': years,
            'predicted_salary': round(float(prediction[0]), 2)
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        })

# -----------------------------------
# Metrics Route
# -----------------------------------

@app.route('/metrics')
def metrics():
    return generate_latest()

# -----------------------------------
# Run Application
# -----------------------------------

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )