import os
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ast import literal_eval

def timesPredicted(data):
    return Counter(re.findall("'[a-z_][a-z_]*'", data.readline()))

data = open("predictions.txt","r+")

def meanPredValue(data):
    str = data.readline()
    print(str)
    return np.array(literal_eval(data))


print(meanPredValue(data))

data.close()
