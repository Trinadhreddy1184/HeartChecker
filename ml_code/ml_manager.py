# File: ml_manager.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import shap


class MLManager:

    def __init__(self, data_path):
        self.model = None
        self.df, self.feature_names = self.preprocess_data(data_path)
        # self.normal_values = {'cp': 2, 'trestbps': 120, 'chol': 200, 'fbs': 0,
        #                       'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 0, 'slope': 1,
        #                       'num_major_vessels': 0, 'thal': 1}
        self.normal_values = {'cp': 0, 'trestbps': 131, 'chol': 245, 'fbs': 0, 'restecg': 0, 'thalach': 149, 'exang': 0, 'oldpeak': 1, 'slope': 1, 'num_major_vessels': 0, 'thal': 2}
        self.train_model()


    def preprocess_data(self, data_path):
        dt = pd.read_csv(data_path)
        print(dt.columns)
        # dt.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope','num_major_vessels', 'thalassemia', 'target']
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
        self.model = model

    def predict(self, user_input):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        # Exclude the target variable if it exists in the input
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return self.model.predict(features_for_prediction)

    def predict_proba(self, user_input):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        # Exclude the target variable if it exists in the input
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return self.model.predict_proba(features_for_prediction)[:, 1]

