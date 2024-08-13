from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'gaussnb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    nitrogen_content = request.form.get("n")
    phosphorus_content = request.form.get("p")
    potassium_content = request.form.get("k")
    temp = request.form.get("temp")
    humidity = request.form.get("humid")
    ph_content = request.form.get("ph")
    rainfall = request.form.get("rain")
    test_row = [nitrogen_content, phosphorus_content, potassium_content, temp, humidity, ph_content, rainfall]
    test_row = [float(x) for x in test_row]
    final_features = np.array(test_row).reshape(1,-1)

    print("########", final_features)
    
    # Make prediction
    prediction = model.predict(final_features)[0]

    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)