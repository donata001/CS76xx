# You want to make sure your version produces better error rates than this :)

import sys

filename = sys.argv[1]
x, y = open(filename, 'r').readlines()[-1].split(',')

with open('prediction.txt', 'w') as f:
    for _ in range(60):
        print >> f, '%s,%s' % (x.strip(), y.strip())
