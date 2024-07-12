from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('D:/Flask/models/model0.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the form
        data = [
            float(request.form['clonesize']),
            float(request.form['honeybee']),
            float(request.form['andrena']),
            float(request.form['osmia']),
            float(request.form['MinOfUpperTRange']),
            float(request.form['AverageOfUpperTRange']),
            float(request.form['AverageOfLowerTRange']),
            float(request.form['RainingDays']),
            float(request.form['seeds'])
        ]
        
        # Convert to numpy array and reshape for the model
        data = np.array(data).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(data)
        
        return render_template('predict.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
