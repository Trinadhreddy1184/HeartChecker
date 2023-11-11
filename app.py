# File: app.py
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pandas as pd
import base64
from io import BytesIO

from ml_code.ml_manager import MLManager
from ml_code.ml_visualizer import MLVisualizer

app = Flask(__name__)
ml_instance = MLManager()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process user input and interact with the MLManager class
        user_input = {
            'age': [float(request.form['age'])],
            'sex': [int(request.form['sex'])],
            # ... Include other features
        }
        predictions = ml_instance.predict(user_input)
        probability = ml_instance.predict_proba(user_input)
        deviations = ml_instance.calculate_attribute_deviations(
            pd.DataFrame(user_input, columns=ml_instance.model.feature_names))
        return render_template('result.html', predictions=predictions, probability=probability,
                               deviations=deviations)
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    # Process data for dashboard and interact with the MLManager class
    features_test = ml_instance.model.df.drop(ml_instance.model.target, axis=1)
    target_test = ml_instance.model.df[ml_instance.model.target]
    X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2,
                                                        random_state=10)
    predictions = ml_instance.predict(X_test)
    probability = ml_instance.predict_proba(X_test)

    ml_instance.model.visualize_results(X_test, y_test, predictions, probability)
    img = BytesIO()
    MLVisualizer.save_visualization_results(file_path=img)
    img_str = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()

    return render_template('dashboard.html', img_str=img_str)


if __name__ == '__main__':
    # Load and preprocess the dataset
    data_path = "/Users/trinadhreddy/Downloads/heart.csv"  # Update with the correct path
   # ml_instance.model.preprocess_data(data_path)

    # Train the model
    #ml_instance.train_model()

    # Set normal values
    normal_values = {'cp': 2, 'trestbps': 120, 'chol': 200, 'fbs': 0,
                     'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 0, 'slope': 1,
                     'num_major_vessels': 0, 'thalassemia': 1}
    #ml_instance.model.set_normal_values(normal_values)

    app.run(debug=True)
