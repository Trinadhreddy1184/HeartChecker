import concurrent.futures
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import pandas as pd

from sklearn.model_selection import train_test_split

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
        # Hide the modebar
        fig.update_layout(showlegend=False)

        # Make it responsive
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=0))

        # Save visualization
        fig.write_html(f'templates/{filename}.html')

    def create_visualizations(self):
        # Gender Distribution
        visual_df = self.ml_instance.df.copy()

        # Replace 'sex' values
        visual_df['sex'] = visual_df['sex'].replace({1: 'Male', 0: 'Female'})

        # Replace 'target' values
        visual_df['target'] = visual_df['target'].replace({0: 'Not Effected', 1: 'Effected'})

        gender_distribution_fig = px.pie(visual_df, names='sex', title='Gender Distribution', labels={'sex': 'Gender'})
        gender_distribution_fig.update_traces(textinfo='percent+label')
        self.create_visualization(gender_distribution_fig, 'gender_distribution')

        # Age Distribution
        age_distribution_fig = px.histogram(
            visual_df, x='age', nbins=30, title='Age Distribution',
            color='target', labels={'age': 'Age', 'target': 'Heart Disease'},
            facet_col='target', facet_col_wrap=2
        )
        age_distribution_fig.update_layout(xaxis_title='Age', yaxis_title='Count')
        self.create_visualization(age_distribution_fig, 'age_distribution')

        # Chest Pain Type Distribution
        cp_distribution_fig = px.histogram(visual_df, x='cp', color='target', nbins=30,
                                          labels={'cp': 'Chest Pain Type', 'target': 'Heart Disease'},
                                          title='Chest Pain Type Distribution', facet_col='target', facet_col_wrap=2)
        cp_distribution_fig.update_layout(xaxis_title='Chest Pain Type', yaxis_title='Count')
        self.create_visualization(cp_distribution_fig, 'cp_distribution')

        # Resting Blood Pressure Distribution
        trestbps_distribution_fig = px.histogram(visual_df, x='trestbps', color='target', nbins=30,
                                                 labels={'trestbps': 'Resting Blood Pressure', 'target': 'Heart Disease'},
                                                 title='Resting Blood Pressure Distribution', facet_col='target', facet_col_wrap=2)
        trestbps_distribution_fig.update_layout(xaxis_title='Resting Blood Pressure', yaxis_title='Count')
        self.create_visualization(trestbps_distribution_fig, 'trestbps_distribution')

        # Serum Cholesterol Distribution
        chol_distribution_fig = px.histogram(visual_df, x='chol', color='target', nbins=30,
                                             labels={'chol': 'Serum Cholesterol', 'target': 'Heart Disease'},
                                             title='Serum Cholesterol Distribution', facet_col='target', facet_col_wrap=2)
        chol_distribution_fig.update_layout(xaxis_title='Serum Cholesterol', yaxis_title='Count')
        self.create_visualization(chol_distribution_fig, 'chol_distribution')

        # Fasting Blood Sugar Distribution
        fbs_distribution_fig = px.histogram(visual_df, x='fbs', color='target', nbins=30,
                                            labels={'fbs': 'Fasting Blood Sugar', 'target': 'Heart Disease'},
                                            title='Fasting Blood Sugar Distribution', facet_col='target', facet_col_wrap=2)
        fbs_distribution_fig.update_layout(xaxis_title='Fasting Blood Sugar', yaxis_title='Count')
        self.create_visualization(fbs_distribution_fig, 'fbs_distribution')

        # Resting Electrocardiographic Results Distribution
        restecg_distribution_fig = px.histogram(visual_df, x='restecg', color='target', nbins=30,
                                                labels={'restecg': 'Resting Electrocardiographic Results', 'target': 'Heart Disease'},
                                                title='Resting Electrocardiographic Results Distribution', facet_col='target', facet_col_wrap=2)
        restecg_distribution_fig.update_layout(xaxis_title='Resting Electrocardiographic Results', yaxis_title='Count')
        self.create_visualization(restecg_distribution_fig, 'restecg_distribution')

        # Maximum Heart Rate Achieved Distribution
        thalach_distribution_fig = px.histogram(visual_df, x='thalach', color='target', nbins=30,
                                                labels={'thalach': 'Maximum Heart Rate Achieved', 'target': 'Heart Disease'},
                                                title='Maximum Heart Rate Achieved Distribution', facet_col='target', facet_col_wrap=2)
        thalach_distribution_fig.update_layout(xaxis_title='Maximum Heart Rate Achieved', yaxis_title='Count')
        self.create_visualization(thalach_distribution_fig, 'thalach_distribution')

        # Exercise Induced Angina Distribution
        exang_distribution_fig = px.histogram(visual_df, x='exang', color='target', nbins=30,
                                              labels={'exang': 'Exercise Induced Angina', 'target': 'Heart Disease'},
                                              title='Exercise Induced Angina Distribution', facet_col='target', facet_col_wrap=2)
        exang_distribution_fig.update_layout(xaxis_title='Exercise Induced Angina', yaxis_title='Count')
        self.create_visualization(exang_distribution_fig, 'exang_distribution')

        # ST Depression Induced by Exercise Relative to Rest Distribution
        oldpeak_distribution_fig = px.histogram(visual_df, x='oldpeak', color='target', nbins=30,
                                               labels={'oldpeak': 'ST Depression Induced by Exercise Relative to Rest', 'target': 'Heart Disease'},
                                               title='ST Depression Induced by Exercise Relative to Rest Distribution', facet_col='target', facet_col_wrap=2)
        oldpeak_distribution_fig.update_layout(xaxis_title='ST Depression Induced by Exercise Relative to Rest', yaxis_title='Count')
        self.create_visualization(oldpeak_distribution_fig, 'oldpeak_distribution')

        # Slope of the Peak Exercise ST Segment Distribution
        slope_distribution_fig = px.histogram(visual_df, x='slope', color='target', nbins=30,
                                              labels={'slope': 'Slope of the Peak Exercise ST Segment', 'target': 'Heart Disease'},
                                              title='Slope of the Peak Exercise ST Segment Distribution', facet_col='target', facet_col_wrap=2)
        slope_distribution_fig.update_layout(xaxis_title='Slope of the Peak Exercise ST Segment', yaxis_title='Count')
        self.create_visualization(slope_distribution_fig, 'slope_distribution')

        # Number of Major Vessels Colored by Fluoroscopy Distribution
        ca_distribution_fig = px.histogram(visual_df, x='ca', color='target', nbins=30,
                                           labels={'ca': 'Number of Major Vessels Colored by Fluoroscopy', 'target': 'Heart Disease'},
                                           title='Number of Major Vessels Colored by Fluoroscopy Distribution', facet_col='target', facet_col_wrap=2)
        ca_distribution_fig.update_layout(xaxis_title='Number of Major Vessels Colored by Fluoroscopy', yaxis_title='Count')
        self.create_visualization(ca_distribution_fig, 'ca_distribution')

        # Thalassemia Distribution
        thal_distribution_fig = px.histogram(visual_df, x='thal', color='target', nbins=30,
                                             labels={'thal': 'Thalassemia', 'target': 'Heart Disease'},
                                             title='Thalassemia Distribution', facet_col='target', facet_col_wrap=2)
        thal_distribution_fig.update_layout(xaxis_title='Thalassemia', yaxis_title='Count')
        self.create_visualization(thal_distribution_fig, 'thal_distribution')

        # Wait for all visualizations to complete
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each visualization for parallel execution
            future_to_fig = {
                executor.submit(self.create_visualization, fig, filename): (fig, filename)
                for fig, filename in [
                    (gender_distribution_fig, 'gender_distribution'),
                    (age_distribution_fig, 'age_distribution'),
                    (cp_distribution_fig, 'cp_distribution'),
                    (trestbps_distribution_fig, 'trestbps_distribution'),
                    (chol_distribution_fig, 'chol_distribution'),
                    (fbs_distribution_fig, 'fbs_distribution'),
                    (restecg_distribution_fig, 'restecg_distribution'),
                    (thalach_distribution_fig, 'thalach_distribution'),
                    (exang_distribution_fig, 'exang_distribution'),
                    (oldpeak_distribution_fig, 'oldpeak_distribution'),
                    (slope_distribution_fig,'slope_distribution'),
                    (ca_distribution_fig, 'ca_distribution'),
                    (thal_distribution_fig, 'thal_distribution'),
                ]
            }

            # Wait for all visualizations to complete
            for future in concurrent.futures.as_completed(future_to_fig):
                filename = future_to_fig[future]
                try:
                    future.result()
                except Exception as e:
                    print(f'Error creating visualization {filename}: {e}')
