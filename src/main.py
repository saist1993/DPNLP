# File where everything begins.

# Library Imports
import logging

# Custom Imports
from tokenizer_wrapper import init_tokenizer
from utils.misc import *

# Setting up logger
logger = logging.getLogger(__name__)

print(tester("Hello World"))