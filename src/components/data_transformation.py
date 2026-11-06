import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class data_transformation_config:
    preprocessor_obj_path: str = str(
        Path(__file__).resolve().parents[2] / "artifacts" / "preprocesser.pkl"
    )


class Data_transformation:

    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('simpleimputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('simpleimputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('num_transformer', num_pipeline, numerical_columns),
                ('cat_transformer', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initate_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('Read the training and test data')

            train_data_inp = train_data.drop(columns=['math_score'], axis=1)
            train_data_op = train_data['math_score']

            test_data_inp = test_data.drop(columns=['math_score'], axis=1)
            test_data_op = test_data['math_score']

            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_transformer_object()

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            train_data_inp = preprocessor.fit_transform(train_data_inp)  # train: fit + transform
            test_data_inp = preprocessor.transform(test_data_inp)        # test: sirf transform

            train_arr = np.c_[train_data_inp, np.array(train_data_op)]
            test_arr = np.c_[test_data_inp, np.array(test_data_op)]

            logging.info("Saving preprocessing object")
            save_obj(
                self.data_transformation_config.preprocessor_obj_path,
                preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)