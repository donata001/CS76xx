# ============================================== 
# REPLACE this file with your final project file 
# ==============================================  

import sys 
import os 
from ast import literal_eval

root = os.path.dirname(os.path.realpath(__file__))

input_file = open(sys.argv[1], 'r')
output_file = open(root + '/prediction.txt', 'w+') 

prediction = [literal_eval(l.strip()) for l in input_file.readlines()][:60]
for p in prediction:
	output_file.write(str(p[0]) + ',' + str(p[1]) + '\n')

input_file.close()
output_file.close()

