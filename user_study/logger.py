# logger_config.py
import logging

# Create a logger
logger = logging.getLogger('my_global_logger')
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)