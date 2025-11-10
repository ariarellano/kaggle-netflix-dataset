import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

from config import (
    PROCESSED_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE,
    RANDOM_STATE, TEST_SIZE, LOG_FORMAT, LOG_LEVEL
)
from data_loader import NetflixDataLoader

#logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class NetflixPreprocessor:
    """Class to handle preprocessing of Netflix data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df)
        
        df_clean = df.copy()
        

        logger.info("Handling missing values...")
        
        # fill na with 'Unknown'
        text_cols = ['director', 'cast', 'country']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        if 'date_added' in df_clean.columns:
            df_clean['date_added'] = df_clean['date_added'].fillna('Unknown')
        

        if 'rating' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['rating'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        logger.info(f"Cleaned data: {initial_rows} -> {len(df_clean)} rows")
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Performing feature engineering...")
        df_fe = df.copy()
        
        if 'date_added' in df_fe.columns:
            df_fe['year_added'] = pd.to_datetime(
                df_fe['date_added'], 
                errors='coerce'
            ).dt.year
        
        if 'release_year' in df_fe.columns:
            current_year = 2025
            df_fe['content_age'] = current_year - df_fe['release_year']
        
        if 'duration' in df_fe.columns:
            df_fe['duration_numeric'] = df_fe['duration'].str.extract('(\d+)').astype(float)
        
        if 'country' in df_fe.columns:
            df_fe['num_countries'] = df_fe['country'].str.split(',').str.len()
        
        if 'cast' in df_fe.columns:
            df_fe['num_cast'] = df_fe['cast'].apply(
                lambda x: 0 if x == 'Unknown' else len(str(x).split(','))
            )
        
        logger.info(f"Created new features. Total columns: {len(df_fe.columns)}")
        return df_fe
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode
            
        Returns:
            DataFrame with encoded columns
        """
        logger.info(f"Encoding categorical columns: {columns}")
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
        
        return df_encoded
    
    def split_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, ...]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("="*50)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*50)
        
        df_clean = self.clean_data(df)
        
        df_features = self.feature_engineering(df_clean)
        
        categorical_cols = ['type', 'rating', 'country']
        df_processed = self.encode_categorical(df_features, categorical_cols)
        
        logger.info(f"Saving processed data to {PROCESSED_DATA_FILE}")
        df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
        
        logger.info("="*50)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*50)
        
        return df_processed


if __name__ == "__main__":
    loader = NetflixDataLoader()
    raw_data = loader.load_data()
    preprocessor = NetflixPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(raw_data)
    
    print("\n" + "="*50)
    print("PROCESSED DATA SAMPLE")
    print("="*50)
    print(processed_data.head())
    
    print("\n" + "="*50)
    print("NEW FEATURES CREATED")
    print("="*50)
    new_cols = [col for col in processed_data.columns if col not in raw_data.columns]
    print(new_cols)