# ============================================ 
# USAGE: 
# cd path/to/this/directory
# python exampleViz.py 
# ============================================ 

import hexbug_visualize.visualizer as viz 

truth = [(1403, 265), (1413, 245), (1426, 224), (1426, 213), (1434, 197), (1448, 177), 
(1458, 157), (1462, 166), (1462, 184), (1463, 191), (1479, 188), (1492, 186), (1521, 178), 
(1550, 179), (1568, 185), (1584, 184), (1612, 186), (1617, 189), (1634, 192), (1626, 192), 
(1624, 202), (1609, 210), (1607, 226), (1607, 232), (1598, 247), (1591, 277), (1585, 295), 
(1572, 314), (1564, 331), (1547, 351)]

predict = [(1435, 313), (1449, 215), (1377, 238), (1384, 242), (1405, 238), (1470, 201), 
(1497, 147), (1451, 135), (1421, 164), (1415, 177), (1442, 231), (1453, 190), (1482, 198), 
(1566, 204), (1528, 145), (1556, 168), (1622, 213), (1604, 167), (1628, 177), (1666, 154), 
(1632, 160), (1636, 199), (1568, 178), (1620, 261), (1636, 268), (1569, 252), (1575, 248), 
(1609, 327), (1554, 301), (1583, 326)]

blueColor = (255,150,50)
redColor = (150,150,255)
viz.setupColors(blueColor, redColor)

# NOTE: this generates out two image files in the current directory 
viz.drawPositions(truth[:50], 1920, 1080, "examplePath.jpg")
viz.drawComparePositions(truth[:50], predict, 50., 1920, 1080, "exampleCompare.jpg")