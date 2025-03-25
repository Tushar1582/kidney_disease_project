from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

# Load the saved machine learning model
with open('tunned_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Expected features (must match your form's "name" attributes)
FEATURE_NAMES = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells", "Pus Cells", "Pus Cell Clumps", "Bacteria", "Blood Glucose Random",
    "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell count", "Red Blood Cell count", "Hypertension", "Diabetes Mellitus",
    "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia"
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather and convert inputs
        features = []
        for field in FEATURE_NAMES:
            value = request.form.get(field)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing or empty value for {field}")
            features.append(float(value))

        input_features = np.array(features).reshape(1, -1)
        prediction = model.predict(input_features)

        # Check prediction and set message, advice, and color accordingly
        if prediction[0] == 1:
            message = "You have a kidney disease"
            advice = "Please consult a doctor immediately"
            color = "red"
        else:
            message = "Good, you are healthy"
            advice = "Please be cautious"
            color = "green"
    except Exception as e:
        message = f"Error in prediction: {e}"
        advice = ""
        color = "black"

    # Render the result page with the messages and color
    return render_template("result.html", message=message, advice=advice, color=color)


if __name__ == "__main__":
    app.run(debug=True)
