import os,sys
import pickle
from src.exception import CustomException

def save_obj(preprocessor_path,preprocessor_obj):
    
    try:
        os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
        
        with open(preprocessor_path,'wb') as file_obj:
            pickle.dump(preprocessor_obj,file_obj)
            
    except Exception as e:
        raise(CustomException(e,sys))                                                                                            
    