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
        self.normal_values = {'cp': 0, 'trestbps': 131, 'chol': 245, 'fbs': 0, 'restecg': 0, 'thalach': 149, 'exang': 0,
                              'oldpeak': 1, 'slope': 1, 'num_major_vessels': 0, 'thal': 2}
        self.male_model = self.category_model(self.df, 'sex', 1)
        self.female_model = self.category_model(self.df, 'sex', 0)
        self.age1_model = self.category_model(self.df, 'age', (1, 25))
        self.age2_model = self.category_model(self.df, 'age', (25, 50))
        self.age3_model = self.category_model(self.df, 'age', (50, 75))
        self.age4_model = self.category_model(self.df, 'age', (75, 121))
        self.train_model()

    def preprocess_data(self, data_path):
        dt = pd.read_csv(data_path)
        df = pd.get_dummies(dt, drop_first=True)
        df = df.dropna()
        feature_names = df.columns.tolist()
        return df, feature_names

    def train_model(self):
        model = RandomForestClassifier(max_depth=5)
        features = self.df.drop('target', axis=1)
        target = self.df['target']
        model.fit(features, target)
        self.model = model

    def predict(self, user_input, category_model=None):
        if category_model is None:
            category_model = self.model
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return category_model.predict(features_for_prediction)

    def predict_proba(self, user_input, category_model=None):
        if category_model is None:
            category_model = self.model
        user_input_df = pd.DataFrame(user_input, columns=self.feature_names)
        features_for_prediction = user_input_df.drop('target', axis=1, errors='ignore')
        return category_model.predict_proba(features_for_prediction)[:, 1]

    def category_model(self, dt, category_column, category_value):
        if isinstance(category_value, tuple):
            # Check if the values in category_column fall within the specified range
            filtered_df = dt[
                (dt[category_column] >= category_value[0]) & (dt[category_column] <= category_value[1])].copy()
            print(filtered_df.head()['age'])
        else:
            # Assume category_value is a direct match
            filtered_df = dt[dt[category_column] == category_value].copy()

        # Check if there are samples in the filtered dataset
        if filtered_df.shape[0] == 0:
            # If no samples, return None or raise an exception based on your needs
            return None

        model = RandomForestClassifier(max_depth=5)
        features = filtered_df.drop('target', axis=1)
        target = filtered_df['target']
        model.fit(features, target)
        return model

