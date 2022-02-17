import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
from copy import deepcopy


class Debugger:  # TODO: good way of selecting debug section from training/env creation.
    def __init__(self, env, sections=None):
        raise NotImplementedError("This feature is still in private beta...")
