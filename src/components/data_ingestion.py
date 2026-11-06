import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from data_transformation import Data_transformation

@dataclass
class DataIngestionConfig:
    artifacts_dir: Path = Path(__file__).resolve().parents[2] / "artifacts"
    train_data_path: str = str(artifacts_dir / "train.csv")
    test_data_path: str = str(artifacts_dir / "test.csv")
    raw_data_path: str = str(artifacts_dir / "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data_source_path = (
                Path(__file__).resolve().parents[2] / "notebook" / "data" / "stud.csv"
            )
            df = pd.read_csv(data_source_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation=Data_transformation()
    data_transformation.initate_transformation(train_data,test_data)