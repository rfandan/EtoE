import os
import sys
import logging

# Define the format of the logs: [Timestamp: Level: Module: Message]
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Specify the directory and file path for logs
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    datefmt="%d %B %Y %H:%M:%S", # Custom date format: Day Month Year Time
    handlers=[
        logging.FileHandler(log_filepath), # Save logs to a file
        logging.StreamHandler(sys.stdout)  # Also print logs to the terminal
    ]
)

# Create a logger object that we can import and use in other files
logger = logging.getLogger("mlProjectLogger")
