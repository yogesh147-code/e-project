import sys
from dataclasses import dataclass
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import find_model,save_obj
from src.exception import CustomException
from src.logger import logging


@dataclass
class Model_trainer_config:
    model_trainer_path:str=str(
        Path(__file__).resolve().parents[2]/"artifacts"/"model.pkl")
    
class Model_trainer:
    
    def __init__(self):
        self.model_path=Model_trainer_config()
        
    def initate_model_training(self,train_data,test_data):
        
        try:
            X_train,y_train,X_test,y_test=(
                train_data[:,:-1],train_data[:,-1],
                test_data[:,:-1],test_data[:,-1]
            )
            
            models = {"Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Linear Regression": LinearRegression(),
                        "Ridge": Ridge(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor(),
                    }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss': ['linear', 'square', 'exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }
            report=find_model(X_train,y_train,X_test,y_test,models,params)
            best_score=max(list(report.values()))
            best_model_name=list(report.keys())[list(report.values()).index(best_score)]
            best_model=models[best_model_name]
            if best_score<0.6:
                raise CustomException('no best model found')
            logging.info(f"Best found model on both training and testing dataset")
            save_obj(self.model_path.model_trainer_path,best_model)
            return best_score
        except Exception as e:
            raise CustomException(e,sys)
        