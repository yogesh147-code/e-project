import logging
import os
from datetime import datetime

log_file=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(log_dir,exist_ok=True)

log_path=os.path.join(log_dir,log_file)

logging.basicConfig(
    filename=log_path,
    filemode='w',
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

if __name__=="__main__":
    logging.info("THis is a info message")