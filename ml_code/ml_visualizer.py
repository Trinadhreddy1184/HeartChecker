import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


class MLVisualizer:
    @staticmethod
    def visualize_results(y_test, predictions, probability):
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy Score: {accuracy:.2%}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        user_index = 0  # Change this index as needed
        user_prediction = probability[user_index] * 100
        print(f"User Prediction Score: {user_prediction:.2f}%")
        threshold = 0.5  # You can adjust this threshold as needed

        # Classify the samples based on the threshold
        binary_predictions = (probability >= threshold).astype(int)

        # Print the classification results
        print("Binary Predictions:")
        print(binary_predictions)

        # Print the percentage of users predicted to have the disease
        percentage_disease = (binary_predictions.sum() / len(binary_predictions)) * 100
        print(f"Percentage of Users Predicted to Have the Disease: {percentage_disease:.2f}%")

    @staticmethod
    def visualize_distributions(df):
        # Visualize attribute distributions in the dataset
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.columns):
            plt.subplot(4, 4, i + 1)
            sns.histplot(df[column], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_attribute_distributions(df, attribute1, attribute2):
        # Compare the distribution of two attributes
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[attribute1], label=attribute1, fill=True)
        sns.kdeplot(df[attribute2], label=attribute2, fill=True)
        plt.title(f'Comparison of {attribute1} and {attribute2} Distributions')
        plt.legend()
        plt.show()

    @staticmethod
    def dataset_info(df):
        # Display detailed information about the dataset
        print(df.info())

