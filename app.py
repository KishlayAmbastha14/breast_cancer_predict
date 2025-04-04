from flask import Flask, request, render_template
import pickle
import sklearn

import numpy as np

app = Flask(__name__)

# Load the saved Logistic Regression model
# model = pickle.load(open(r'c:\Users\kishl\Downloads\lr_model.pkl', 'rb'))
# print("done")
models = pickle.load(open('lr_model.pkl','rb'))
print("done")

@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        # features = [float(x) for x in request.form.values()]
        features = [
           float(request.form['mean_radius']),
           float(request.form['mean_perimeter']),
           float(request.form['mean_concave_points']),
           float(request.form['worst_perimeter']),
           float(request.form['worst_area'])
        ]

        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = models.predict(features)[0]

        # Display result
        # result = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-Cancerous)"
        if prediction == 0:
           result = 'Malignant (Cancerous)'
           color = 'red'
        else:
           result = 'Benign (Non-Cancerous)'
           color = 'green'

        return render_template('index.html', prediction_text=f'Result: {result}',color=color)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
  app.run(debug=True)
