# ============================================== 
# Run this file to test whether your final project file works. 
# Feel free to replace the data in training_data.txt 
# ==============================================  

import os 
import subprocess
from math import sqrt  
from ast import literal_eval

root = os.path.dirname(os.path.realpath(__file__))

# runs a process to evoke the finalproject.py file with training_data as input 
subprocess.call(["python", root + "/finalproject.py", root + "/training_data.txt"])

predict = open(root + '/prediction.txt', 'r') 
predict_data = [literal_eval(l.strip()) for l in predict.readlines()]
predict.close()
print "prediction:", predict_data

test = open(root + '/test_data.txt', 'r') 
test_data = [literal_eval(l.strip()) for l in test.readlines()]
test.close() 
print "testing:", test_data 

def distance(pa, pb):
    return sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)

error = sqrt(sum([distance(predict_data[i], test_data[i])**2 for i in range(len(test_data))]))
print "error:", error 
