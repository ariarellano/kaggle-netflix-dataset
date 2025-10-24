import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

from config import (
    PROCESSED_DATA_FILE, MODELS_DIR, RANDOM_STATE, 
    BATCH_SIZE, LEARNING_RATE, EPOCHS, LOG_FORMAT, LOG_LEVEL
)
from preprocessing import NetflixPreprocessor

# Setup logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class NetflixDataset(Dataset):
    """PyTorch Dataset for Netflix data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NetflixClassifier(nn.Module):
    """Simple neural network for classification"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NetflixClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ModelTrainer:
    """Class to handle model training"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize trainer
        
        Args:
            model_type: 'random_forest' or 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame, target_col='type'):
        """
        Prepare data for training
        
        Args:
            df: Preprocessed DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of prepared data
        """
        logger.info(f"Preparing data for {self.model_type} model")
        
        # Select numeric features for modeling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if it's numeric
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        X = df[numeric_cols].fillna(0)
        y = df[target_col]
        
        # Encode target if it's not already numeric
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Save label encoder
            joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
            logger.info(f"Target classes: {le.classes_}")
        
        return X, y
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Save model
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return self.model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        logger.info("Training Neural Network model...")
        
        # Create datasets and dataloaders
        train_dataset = NetflixDataset(X_train, y_train)
        val_dataset = NetflixDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        hidden_size = 128
        
        self.model = NetflixClassifier(input_size, hidden_size, num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save model
        model_path = MODELS_DIR / 'neural_network_model.pth'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        return self.model


def main():
    """Main training function"""
    logger.info("="*50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*50)
    
    # Load processed data
    logger.info(f"Loading data from {PROCESSED_DATA_FILE}")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Initialize trainer (change to 'neural_network' if you want to use PyTorch)
    trainer = ModelTrainer(model_type='random_forest')
    
    # Prepare data
    X, y = trainer.prepare_data(df, target_col='type')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train model
    if trainer.model_type == 'random_forest':
        model = trainer.train_random_forest(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info("="*50)
        logger.info(f"TEST SET ACCURACY: {accuracy:.4f}")
        logger.info("="*50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    elif trainer.model_type == 'neural_network':
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
        )
        model = trainer.train_neural_network(X_train, y_train, X_val, y_val)
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)


if __name__ == "__main__":
    main()