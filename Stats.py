import os
import re
from collections import Counter
import matplotlib.pyplot as plt

def times_predicted(data):
    return Counter(re.findall("'[a-z_][a-z_]*'", data.readline()))

data = open("predictions.txt","r+")
stats = open("stats.txt","w+")

print(times_predicted(data))

data.close()
data.close()
