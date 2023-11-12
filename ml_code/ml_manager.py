# File: ml_manager.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import shap


class MLManager:
    model = None
    normal_values = {'cp': 2, 'trestbps': 120, 'chol': 200, 'fbs': 0,
                    'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 0, 'slope': 1,
                    'num_major_vessels': 0, 'thalassemia': 1}

    def __init__(self, data_path):
        self.df, self.feature_names = self.preprocess_data(data_path)
        self.train_model()

    def preprocess_data(self, data_path):
        dt = pd.read_csv(data_path)
        dt.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                      'num_major_vessels', 'thalassemia', 'target']
        dt.loc[dt['sex'] == 0, 'sex'] = 'female'
        dt.loc[dt['sex'] == 1, 'sex'] = 'male'
        df = pd.get_dummies(dt, drop_first=True)

        # Drop rows with NaN values
        df = df.dropna()

        feature_names = df.columns.tolist()
        return df, feature_names

    def train_model(self):
        model = RandomForestClassifier(max_depth=5)
        features = self.df.drop('target', axis=1)
        target = self.df['target']
        model.fit(features, target)
        MLManager.model = model

    def predict(self, user_input):
        if MLManager.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        # Exclude the target variable if it exists in the input
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return MLManager.model.predict(features_for_prediction)

    def predict_proba(self, user_input):
        if MLManager.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        # Exclude the target variable if it exists in the input
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return MLManager.model.predict_proba(features_for_prediction)[:, 1]

    def calculate_attribute_deviations(self, user_input):
        deviations = {attribute: user_input.get(attribute, 0) - MLManager.normal_values.get(attribute, 0)
                      for attribute in set(user_input.keys()) & set(MLManager.normal_values.keys())}
        return deviations

    def generate_shap_plots(self, data_for_prediction):
        explainer = shap.TreeExplainer(MLManager.model)
        shap_values = explainer.shap_values(data_for_prediction)
        shap.summary_plot(shap_values[1], data_for_prediction)


# # Example usage:
# if __name__ == "__main__":
#     data_path = "/Users/trinadhreddy/Downloads/heart.csv"
#     ml_instance = MLManager(data_path)
#
#     # Sample user input for testing
#     user_input = {'age': [52], 'sex': [1], 'cp': [0], 'trestbps': [125], 'chol': [212], 'fbs': [0],
#                   'restecg': [1], 'thalach': [168], 'exang': [0], 'oldpeak': [1], 'slope': [2],
#                   'ca': [2], 'thal': [3]}
#
#     # Evaluate the model on the test set
#     features_test = ml_instance.df.drop('target', axis=1)
#     target_test = ml_instance.df['target']
#     X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2, random_state=10)
#     predictions = ml_instance.predict(X_test)
#     probability = ml_instance.predict_proba(X_test)
#
#     # Visualize the results
#     #ml_instance.visualize_results(X_test, y_test, predictions, probability)
#
#     # Calculate attribute deviations
#     deviations = ml_instance.calculate_attribute_deviations(pd.DataFrame(user_input, columns=ml_instance.feature_names))
#     print('Attribute Deviations:')
#     for attribute, deviation in deviations.items():
#         print(f'{attribute}: {deviation}')
#
#     # Example SHAP plot generation
#     data_for_prediction = pd.DataFrame(user_input, columns=ml_instance.feature_names).astype(float)
#     ml_instance.generate_shap_plots(data_for_prediction)
