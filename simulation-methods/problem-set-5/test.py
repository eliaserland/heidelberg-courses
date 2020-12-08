import numpy
from matplotlib.pyplot import *
import math
import time


def rand_gauss():
        return numpy.random.normal()


for i in range(20):
        print(f'{rand_gauss():.3}')

