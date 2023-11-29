import numpy as np
import pandas as pd
import argparse

class Parser(utils.Parser):
    subdirectory_retnet: str
    subdirectory_gpt: str

args = Parser().parse_args('plot')

