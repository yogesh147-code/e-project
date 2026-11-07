import os,sys
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def save_obj(preprocessor_path,preprocessor_obj):
    
    try:
        os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
        
        with open(preprocessor_path,'wb') as file_obj:
            pickle.dump(preprocessor_obj,file_obj)
            
    except Exception as e:
        raise(CustomException(e,sys))
    
def find_model(X_train,y_train,X_test,y_test,models,params):
    try:
        r2_scores={}
        for model_name, model in models.items():
            param=params[model_name]
            grid=GridSearchCV(model,param_grid=param,cv=3)
            grid.fit(X_train,y_train)
            best_model=grid.best_estimator_
            test_score=r2_score(y_test,best_model.predict(X_test))
            models[model_name]=best_model
            r2_scores[model_name]=test_score
        return r2_scores
    except Exception as e:
        raise CustomException(e,sys)
        
def load_obj(path):
    try:
        with open(path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)