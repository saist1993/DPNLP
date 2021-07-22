# File where everything begins.

# Library Imports
import logging

# Custom Imports
from utils.misc import *
from tokenizer_wrapper import init_tokenizer
from create_data import generate_data_iterators

# Setting up logger
logger = logging.getLogger(__name__)

print(tester("Hello World"))