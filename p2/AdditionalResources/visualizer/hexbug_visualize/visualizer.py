import cv2 
import numpy as np 
import math 

# ===============================================================
# globals 

VIZ_COLORS = {
    "first": (255,150,50),
    "second": (150,150,255)
}

def setupColors(firstColor, secondColor):
    global VIZ_COLORS
    VIZ_COLORS["first"] = firstColor
    VIZ_COLORS["second"] = secondColor

# ===============================================================
# helpers 

def distance(pa, pb):
    return math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)

def colorBetween(startColor, endColor, percentage):
    percentage = min(1., max(0., percentage)) # limit percentage to 0-1 

    ib = int(endColor[0] * percentage + startColor[0] * (1 - percentage))
    ig = int(endColor[1] * percentage + startColor[1] * (1 - percentage))
    ir = int(endColor[2] * percentage + startColor[2] * (1 - percentage))
    return (ib,ig,ir)

# ===============================================================
# drawing  

def drawComparePositions(truthSet, predictSet, maxError, maxW, maxH, outputFile):
    truth_color = VIZ_COLORS["first"] # feel free to overwrite 
    predict_color = VIZ_COLORS["second"] # feel free to overwrite 
    bg_image = np.ones(shape=(maxH,maxW,3), dtype=np.uint8) * 255

    for i in range(len(truthSet)):
        if i > 0:
            # connect the dots for sets 
            cv2.line(bg_image,(truthSet[i-1][0],truthSet[i-1][1]),(truthSet[i][0],truthSet[i][1]),truth_color,2)
            cv2.line(bg_image,(predictSet[i-1][0],predictSet[i-1][1]),(predictSet[i][0],predictSet[i][1]),predict_color,2)
    # write out 
    cv2.imwrite(outputFile, bg_image)

def drawPositions(positions, maxW, maxH, outputFile):
    start_color = VIZ_COLORS["first"] # feel free to overwrite 
    end_color = VIZ_COLORS["second"] # feel free to overwrite 
    bg_image = np.ones(shape=(maxH,maxW,3), dtype=np.uint8) * 255

    for i in range(len(positions)):
        if i > 0:
            color = colorBetween(start_color,end_color,(i+1) * 1.0 / (len(positions)))
            cv2.line(bg_image,(positions[i-1][0],positions[i-1][1]),(positions[i][0],positions[i][1]),color,2)
    # write out 
    cv2.imwrite(outputFile, bg_image)
