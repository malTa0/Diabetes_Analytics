from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and preprocess the diabetes dataset
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['blood_pressure']),
            'SkinThickness': float(request.form['skin_thickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree_function']),
            'Age': float(request.form['age']),
        }

        # Scale the user input
        user_input_scaled = scaler.transform(pd.DataFrame(user_input, index=[0]))

        # Make a prediction
        prediction = model.predict(user_input_scaled)[0]

        # Define a result description based on the prediction
        result_description = "The model predicts that the patient has diabetes." if prediction == 1 else "The model predicts that the patient does not have diabetes."

        # Render the result page
        return render_template('result.html', prediction=prediction, result_description=result_description)

    # Render the input page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
