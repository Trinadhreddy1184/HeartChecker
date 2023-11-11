import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap


class MLManager:
    model = None
    normal_values = None

    @staticmethod
    def train_model(df, target_variable='target'):
        model = RandomForestClassifier(max_depth=5)
        features = df.drop(target_variable, axis=1)
        target = df[target_variable]
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, random_state=10)
        model.fit(X_train, y_train)
        MLManager.model = model

    @staticmethod
    def predict(user_input):
        if MLManager.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=MLManager.model.feature_names)
        features_for_prediction = user_input_df.drop(MLManager.model.target, axis=1)
        return MLManager.model.predict(features_for_prediction)

    @staticmethod
    def predict_proba(user_input):
        if MLManager.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        user_input_df = pd.DataFrame(user_input, columns=MLManager.model.feature_names)
        features_for_prediction = user_input_df.drop(MLManager.model.target, axis=1)
        return MLManager.model.predict_proba(features_for_prediction)[:, 1]

    @staticmethod
    def calculate_attribute_deviations(user_input):
        if MLManager.normal_values is None:
            raise ValueError("Normal values not set. Call set_normal_values first.")
        deviations = {attribute: user_input.get(attribute, 0) - MLManager.normal_values.get(attribute, 0)
                      for attribute in set(user_input.keys()) & set(MLManager.normal_values.keys())}
        return deviations

    @staticmethod
    def set_normal_values(normal_values):
        MLManager.normal_values = normal_values

    @staticmethod
    def generate_shap_plots(data_for_prediction):
        explainer = shap.TreeExplainer(MLManager.model)
        shap_values = explainer.shap_values(data_for_prediction)
        shap.summary_plot(shap_values[1], data_for_prediction)
