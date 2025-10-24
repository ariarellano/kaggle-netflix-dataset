import pandas as pd
import numpy as np
import torch
import joblib
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_FILE, MODELS_DIR, OUTPUTS_DIR,
    RANDOM_STATE, LOG_FORMAT, LOG_LEVEL
)
from train import NetflixClassifier

# Setup logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelTester:
    """Class to handle model testing and evaluation"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize tester
        
        Args:
            model_type: 'random_forest' or 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading {self.model_type} model...")
        
        # Load label encoder
        le_path = MODELS_DIR / 'label_encoder.pkl'
        if le_path.exists():
            self.label_encoder = joblib.load(le_path)
            logger.info(f"Loaded label encoder with classes: {self.label_encoder.classes_}")
        
        if self.model_type == 'random_forest':
            model_path = MODELS_DIR / 'random_forest_model.pkl'
            self.model = joblib.load(model_path)
            logger.info(f"Loaded Random Forest model from {model_path}")
            
        elif self.model_type == 'neural_network':
            model_path = MODELS_DIR / 'neural_network_model.pth'
            # Need to reconstruct model architecture
            # This is a simplified version - in production, save architecture too
            self.model = NetflixClassifier(input_size=10, hidden_size=128, num_classes=2)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded Neural Network model from {model_path}")
    
    def prepare_test_data(self, df: pd.DataFrame, target_col='type'):
        """
        Prepare test data
        
        Args:
            df: DataFrame with test data
            target_col: Name of target column
            
        Returns:
            X_test, y_test
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        X_test = df[numeric_cols].fillna(0)
        y_test = df[target_col]
        
        # Encode target if string
        if y_test.dtype == 'object' and self.label_encoder:
            y_test = self.label_encoder.transform(y_test)
        
        return X_test, y_test
    
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        if self.model_type == 'random_forest':
            return self.model.predict(X_test)
            
        elif self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test.values).to(self.device)
                outputs = self.model(X_tensor)
                _, predictions = torch.max(outputs, 1)
                return predictions.cpu().numpy()
    
    def evaluate(self, y_test, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
        """
        logger.info("="*50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*50)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = None
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = OUTPUTS_DIR / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {plot_path}")
        plt.close()
        
        return accuracy
    
    def predict_single(self, features: dict):
        """
        Make prediction for a single sample
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction and probability
        """
        # Convert to DataFrame
        df_single = pd.DataFrame([features])
        
        if self.model_type == 'random_forest':
            prediction = self.model.predict(df_single)[0]
            probabilities = self.model.predict_proba(df_single)[0]
            
            if self.label_encoder:
                prediction = self.label_encoder.inverse_transform([prediction])[0]
            
            return prediction, probabilities
        
        return None, None


def main():
    """Main testing function"""
    logger.info("="*50)
    logger.info("STARTING MODEL TESTING")
    logger.info("="*50)
    
    # Load processed data
    logger.info(f"Loading data from {PROCESSED_DATA_FILE}")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Initialize tester
    tester = ModelTester(model_type='random_forest')
    
    # Load model
    tester.load_model()
    
    # Prepare test data
    X, y = tester.prepare_test_data(df, target_col='type')
    
    # Split to get test set (same random state as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = tester.predict(X_test)
    
    # Evaluate
    accuracy = tester.evaluate(y_test, y_pred)
    
    # Test single prediction
    logger.info("\n" + "="*50)
    logger.info("TESTING SINGLE PREDICTION")
    logger.info("="*50)
    
    sample_features = X_test.iloc[0].to_dict()
    prediction, probabilities = tester.predict_single(sample_features)
    
    if prediction:
        logger.info(f"Predicted class: {prediction}")
        logger.info(f"Probabilities: {probabilities}")
    
    logger.info("="*50)
    logger.info("TESTING COMPLETE")
    logger.info("="*50)


if __name__ == "__main__":
    main()