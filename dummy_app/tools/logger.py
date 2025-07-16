import logging 
import datetime 
import os 
from dummy_app.tools.common import jupyter_logger

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

parent_dir = os.getcwd()
assets_dir = parent_dir + "/assets"
if not os.path.exists(assets_dir):
    os.mkdir(assets_dir)

log_dir = parent_dir + "/assets/logs"

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(filename=log_file, filemode='w', format='[%(asctime)a][%(levelname)s]:%(message)s', encoding='utf-8', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger("milp")

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Default console level

# Create formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Add formatter to handler
ch.setFormatter(formatter)

# Add handler to logger if not already added (avoids duplicate logs)
if not logger.hasHandlers():
    logger.addHandler(ch)

jupyter_handler = jupyter_logger(level=logging.INFO)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(jupyter_handler) 

