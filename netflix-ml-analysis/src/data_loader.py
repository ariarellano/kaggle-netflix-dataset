"""
Data loading utilities for Netflix dataset
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from config import RAW_DATA_FILE, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class NetflixDataLoader:
    """Class to handle loading and basic validation of Netflix data"""
    
    def __init__(self, file_path: Optional[Path] = None):
        """
        Initialize the data loader
        
        Args:
            file_path: Path to the CSV file. If None, uses default from config
        """
        self.file_path = file_path or RAW_DATA_FILE
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Netflix dataset from CSV
        
        Returns:
            DataFrame containing the Netflix data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded {len(self.data)} records")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
        
        info = {
            "num_records": len(self.data),
            "num_columns": len(self.data.columns),
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Dataset contains {info['num_records']} records and {info['num_columns']} columns")
        return info
    
    def display_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Display a sample of the data
        
        Args:
            n: Number of rows to display
            
        Returns:
            DataFrame with sample rows
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        return self.data.head(n)


if __name__ == "__main__":
    loader = NetflixDataLoader()
    data = loader.load_data()
    
    print("\n" + "="*50)
    print("NETFLIX DATASET INFO")
    print("="*50)
    
    info = loader.get_basic_info()
    print(f"\nTotal Records: {info['num_records']}")
    print(f"Total Columns: {info['num_columns']}")
    print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
    
    print("\n" + "="*50)
    print("SAMPLE DATA")
    print("="*50)
    print(loader.display_sample())
    
    print("\n" + "="*50)
    print("MISSING VALUES")
    print("="*50)
    for col, missing in info['missing_values'].items():
        if missing > 0:
            print(f"{col}: {missing} ({missing/info['num_records']*100:.1f}%)")