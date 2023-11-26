import concurrent.futures
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from io import BytesIO
from ml_code.ml_manager import MLManager

class SubMLManager:
    def __init__(self, ml_instance):
        self.ml_instance = ml_instance
        self.create_visualizations()

    def get_dashboard_data(self):
        features_test = self.ml_instance.df.drop('target', axis=1)
        target_test = self.ml_instance.df['target']
        X_train, X_test, y_train, y_test = train_test_split(features_test, target_test, test_size=0.2, random_state=100)
        predictions = self.ml_instance.predict(X_test)
        probability = self.ml_instance.predict_proba(X_test)

        accuracy = float(format(accuracy_score(y_test, predictions) * 100, ".2f"))
        cm = confusion_matrix(y_test, predictions)

        y_pred_quant = self.ml_instance.model.predict_proba(X_test)[:, 1]
        y_pred_bin = np.where(y_pred_quant > 0.5, 1, 0)

        conf_matrix = confusion_matrix(y_test, y_pred_bin)

        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
        specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
        roc_auc = auc(fpr, tpr)

        fpr = fpr.tolist()
        tpr = tpr.tolist()
        thresholds = thresholds.tolist()

        # Return the dashboard data
        return {
            'accuracy': accuracy,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_auc': roc_auc
        }

    def create_visualization(self, fig, filename):
        fig.write_html(f'templates/{filename}.html')

    def create_visualizations(self):
        # Gender Distribution
        gender_distribution_fig = px.pie(self.ml_instance.df, names='sex', title='Gender Distribution')
        gender_distribution_fig.update_traces(textinfo='percent+label')

        # Age Distribution
        age_distribution_fig = px.histogram(self.ml_instance.df, x='age', nbins=20, title='Age Distribution')
        age_distribution_fig.update_layout(xaxis_title='Age', yaxis_title='Count')

        # Chest Pain Type vs. Heart Disease
        chest_pain_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='cp', color='target',
                                                 labels={'cp': 'Chest Pain Type', 'target': 'Heart Disease'},
                                                 title='Chest Pain Type vs. Heart Disease')
        chest_pain_vs_heart_disease_fig.update_layout(xaxis_title='Chest Pain Type', yaxis_title='Count')

        # Resting Blood Pressure vs. Heart Disease
        resting_bp_vs_heart_disease_fig = px.box(self.ml_instance.df, x='target', y='trestbps', points='all',
                                                 labels={'target': 'Heart Disease', 'trestbps': 'Resting Blood Pressure'},
                                                 title='Resting Blood Pressure vs. Heart Disease')
        resting_bp_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Resting Blood Pressure')

        # Serum Cholesterol vs. Heart Disease
        chol_vs_heart_disease_fig = px.box(self.ml_instance.df, x='target', y='chol', points='all',
                                           labels={'target': 'Heart Disease', 'chol': 'Serum Cholesterol'},
                                           title='Serum Cholesterol vs. Heart Disease')
        chol_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Serum Cholesterol')

        # Fasting Blood Sugar vs. Heart Disease
        fbs_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='fbs', color='target',
                                           labels={'fbs': 'Fasting Blood Sugar', 'target': 'Heart Disease'},
                                           title='Fasting Blood Sugar vs. Heart Disease')
        fbs_vs_heart_disease_fig.update_layout(xaxis_title='Fasting Blood Sugar', yaxis_title='Count')

        # Resting Electrocardiographic Results vs. Heart Disease
        restecg_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='restecg', color='target',
                                               labels={'restecg': 'Resting Electrocardiographic Results', 'target': 'Heart Disease'},
                                               title='Resting Electrocardiographic Results vs. Heart Disease')
        restecg_vs_heart_disease_fig.update_layout(xaxis_title='Resting Electrocardiographic Results', yaxis_title='Count')

        # Maximum Heart Rate Achieved vs. Heart Disease
        thalach_vs_heart_disease_fig = px.box(self.ml_instance.df, x='target', y='thalach', points='all',
                                               labels={'target': 'Heart Disease', 'thalach': 'Maximum Heart Rate Achieved'},
                                               title='Maximum Heart Rate Achieved vs. Heart Disease')
        thalach_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='Maximum Heart Rate Achieved')

        # Exercise Induced Angina vs. Heart Disease
        exang_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='exang', color='target',
                                             labels={'exang': 'Exercise Induced Angina', 'target': 'Heart Disease'},
                                             title='Exercise Induced Angina vs. Heart Disease')
        exang_vs_heart_disease_fig.update_layout(xaxis_title='Exercise Induced Angina', yaxis_title='Count')

        # ST Depression Induced by Exercise Relative to Rest vs. Heart Disease
        oldpeak_vs_heart_disease_fig = px.box(self.ml_instance.df, x='target', y='oldpeak', points='all',
                                               labels={'target': 'Heart Disease', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest'},
                                               title='ST Depression Induced by Exercise Relative to Rest vs. Heart Disease')
        oldpeak_vs_heart_disease_fig.update_layout(xaxis_title='Heart Disease', yaxis_title='ST Depression Induced by Exercise Relative to Rest')

        # Slope of the Peak Exercise ST Segment vs. Heart Disease
        slope_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='slope', color='target',
                                             labels={'slope': 'Slope of the Peak Exercise ST Segment', 'target': 'Heart Disease'},
                                             title='Slope of the Peak Exercise ST Segment vs. Heart Disease')
        slope_vs_heart_disease_fig.update_layout(xaxis_title='Slope of the Peak Exercise ST Segment', yaxis_title='Count')

        # Number of Major Vessels Colored by Fluoroscopy vs. Heart Disease
        ca_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='ca', color='target',
                                         labels={'ca': 'Number of Major Vessels Colored by Fluoroscopy', 'target': 'Heart Disease'},
                                         title='Number of Major Vessels Colored by Fluoroscopy vs. Heart Disease')
        ca_vs_heart_disease_fig.update_layout(xaxis_title='Number of Major Vessels Colored by Fluoroscopy', yaxis_title='Count')

        # Thalassemia vs. Heart Disease
        thal_vs_heart_disease_fig = px.bar(self.ml_instance.df, x='thal', color='target',
                                           labels={'thal': 'Thalassemia', 'target': 'Heart Disease'},
                                           title='Thalassemia vs. Heart Disease')
        thal_vs_heart_disease_fig.update_layout(xaxis_title='Thalassemia', yaxis_title='Count')

        # Use multi-threading to create visualizations concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each visualization for parallel execution
            future_to_fig = {
                executor.submit(self.create_visualization, fig, filename): filename
                for fig, filename in [
                    (gender_distribution_fig, 'gender_distribution'),
                    (age_distribution_fig, 'age_distribution'),
                    (chest_pain_vs_heart_disease_fig, 'chest_pain_vs_heart_disease'),
                    (resting_bp_vs_heart_disease_fig, 'resting_bp_vs_heart_disease'),
                    (chol_vs_heart_disease_fig, 'chol_vs_heart_disease'),
                    (fbs_vs_heart_disease_fig, 'fbs_vs_heart_disease'),
                    (restecg_vs_heart_disease_fig, 'restecg_vs_heart_disease'),
                    (thalach_vs_heart_disease_fig, 'thalach_vs_heart_disease'),
                    (exang_vs_heart_disease_fig, 'exang_vs_heart_disease'),
                    (oldpeak_vs_heart_disease_fig, 'oldpeak_vs_heart_disease'),
                    (slope_vs_heart_disease_fig, 'slope_vs_heart_disease'),
                    (ca_vs_heart_disease_fig, 'ca_vs_heart_disease'),
                    (thal_vs_heart_disease_fig, 'thal_vs_heart_disease'),
                ]
            }

            # Wait for all visualizations to complete
            for future in concurrent.futures.as_completed(future_to_fig):
                filename = future_to_fig[future]
                try:
                    future.result()
                except Exception as e:
                    print(f'Error creating visualization {filename}: {e}')
