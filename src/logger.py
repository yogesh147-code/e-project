import logging
from datetime import datetime
from pathlib import Path

log_file = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_dir = Path(__file__).resolve().parents[1] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / log_file

logging.basicConfig(
    filename=str(log_path),
    filemode='w',
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

if __name__=="__main__":
    logging.info("THis is a info message")