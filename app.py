from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load("best_model.pkl")
features = joblib.load("features.pkl")

# Salary ranges for each class (based on training data statistics)
# Class 0: Below median salary (~$125k)
# Class 1: Above median salary (~$155k)
SALARY_ESTIMATES = {
    0: {"min": 75000, "max": 125000, "avg": 100000},   # Low earner
    1: {"min": 125000, "max": 180000, "avg": 155000}   # High earner
}

@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")

@app.route("/predict-page", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['job_title', 'experience_years', 'education_level', 'skills_count', 
                          'industry', 'company_size', 'location', 'remote_work', 'certifications']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Validate input ranges
        try:
            experience_years = int(data['experience_years'])
            skills_count = int(data['skills_count'])
            certifications = int(data['certifications'])
            remote_work = int(data['remote_work'])
            
            if not (0 <= experience_years <= 70):
                return jsonify({"error": "Experience years must be between 0 and 70"}), 400
            if not (0 <= skills_count <= 100):
                return jsonify({"error": "Skills count must be between 0 and 100"}), 400
            if not (0 <= certifications <= 50):
                return jsonify({"error": "Certifications must be between 0 and 50"}), 400
            if remote_work not in [0, 1]:
                return jsonify({"error": "Remote work must be 0 or 1"}), 400
                
        except ValueError as e:
            return jsonify({"error": "Numeric fields must be valid numbers"}), 400
        
        # Create DataFrame with the same structure as training
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=features, fill_value=0)
        
        # Make prediction (0 = low earner, 1 = high earner)
        prediction_class = model.predict(df)[0]
        
        # Get probability for better estimate
        prediction_proba = model.predict_proba(df)[0]
        confidence = max(prediction_proba) * 100
        
        # Estimate salary based on class and features
        salary_range = SALARY_ESTIMATES.get(int(prediction_class), SALARY_ESTIMATES[0])
        base_salary = salary_range["avg"]
        
        # Adjust salary based on experience and skills
        experience_years = data.get('experience_years', 5)
        skills_count = data.get('skills_count', 5)
        
        # Experience bonus: +$2000 per year
        experience_bonus = experience_years * 2000
        # Skills bonus: +$1000 per skill
        skills_bonus = skills_count * 1000
        
        estimated_salary = max(
            salary_range["min"],
            min(
                salary_range["max"],
                base_salary + experience_bonus + skills_bonus
            )
        )
        
        return jsonify({
            "prediction": int(estimated_salary),
            "class": int(prediction_class),
            "confidence": round(confidence, 2),
            "description": "High Earner" if prediction_class == 1 else "Average Earner",
            "range": {
                "min": salary_range["min"],
                "max": salary_range["max"],
                "estimated": int(estimated_salary)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)