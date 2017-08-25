import os
import sys 
import glob
import shutil
import argparse
import time
import subprocess

import numpy as np 
from scipy import signal as sg 
import scipy.stats as stats 
import cv2 
import math 

import md5
from hexbug_lib import * 

# =================================================================================
# ================================================================================= 
# Uses background subtraction to deterministically track a hexbug in video files. 
# Written for OMSCS 8803 final project, Fall 2015 & Spring 2016.
# 
# All permissions granted freely without restriction for use, modification, etc. 
# No warranties, disclaim all liability for damages resulting from use, etc. 
# 
# Example usage: python hexbug_tracker.py -file test01 -debug 1
# ================================================================================= 

# constants for directory names, input types, etc - feel free to modify accordingly 
INTENSITY_THRESHOLD = 0.75

DIST_THRESHOLD = 20 
TINY_NUMBER = 0.000001

# just a timer 
start_time = time.time()

# =================================================================================  
# =================================================================================  
# general functions 

# ----------------------------------------------- 
# creates and attempts to set up the video capture 

def read_all_frames(base_name):
    os.chdir(INPUT_DIR + '/' + base_name)
    frames = glob.glob(base_name + '*' + '.jpg')
    os.chdir('../../')
    return frames

# ----------------------------------------------- 
# guesses a position from the intensity image 
# given by a background subtraction 

def guess_position(foreground_image):
    # threshold the intensity image 
    image = np.array(foreground_image)
    image[image < (np.max(image) * INTENSITY_THRESHOLD)] = 0

    # generate the xy from remaining distribution 
    # first normalize and then multiply by values on each row/col 
    y_image = np.sum(image, axis=1) / max(TINY_NUMBER,np.sum(image))
    x_image = np.sum(image, axis=0) / max(TINY_NUMBER,np.sum(image))
    y_est = np.sum(y_image * np.array(range(y_image.shape[0])))
    x_est = np.sum(x_image * np.array(range(x_image.shape[0])))

    # 10x10 square to help visualization 
    miny, maxy = int(y_est - 10), int(y_est + 10)
    minx, maxx = int(x_est - 10), int(x_est + 10)
    return minx, miny, maxx, maxy

def guess_postion_wrapper(foreground_image, prev_pos, confidence=1.0):
    if not prev_pos is None:
        radius = DIST_THRESHOLD
        mask = np.ones(shape=foreground_image.shape) * (1.0/confidence)
        mask[  max(0,prev_pos[0]-radius):min(foreground_image.shape[0],prev_pos[0]+radius), 
                max(0,prev_pos[1]-radius):min(foreground_image.shape[1],prev_pos[1]+radius)] = 1
        foreground_image *= mask 

    minx, miny, maxx, maxy = guess_position(foreground_image)
    return np.array([(miny+maxy)/2, (minx+maxx)/2])

def guess_position_denoise_bounds(foreground_image, prev_pos, confidence=1.0):

    guess_pos = np.array([0.0, 0.0])

    # threshold the intensity image 
    image = np.array(cv2.blur(foreground_image, (3,3)), dtype='uint8')
    image = cv2.medianBlur(image, 3)
    image = np.array(image, dtype = np.float)
    image[image < (np.max(image) * INTENSITY_THRESHOLD)] = 0
    
    if not prev_pos is None:
        radius = DIST_THRESHOLD
        image[  max(0,prev_pos[0]-radius):min(image.shape[0],prev_pos[0]+radius), 
                max(0,prev_pos[1]-radius):min(image.shape[1],prev_pos[1]+radius)] *= confidence

    # generate the xy from remaining distribution 
    # first normalize and then multiply by values on each row/col 
    y_image = np.sum(image, axis=1) / np.sum(image)
    x_image = np.sum(image, axis=0) / np.sum(image)
    guess_pos[0] = np.sum(y_image * np.array(range(y_image.shape[0])))
    guess_pos[1] = np.sum(x_image * np.array(range(x_image.shape[0])))

    return guess_pos

# ================================================================================= 
# ================================================================================= 
# video based tracker functions for BACKGROUND SUBTRACTION 

# ----------------------------------------------- 
# generate N number of backgrounds to be used 
# for background subtraction  

def generate_background_estimations(base_name, frames, estimation_size, is_debug):
    # we need a num of estimations for backgrounds 
    bg_estimations = []

    # get a sampling of frames to generate an estimation 
    sampling_frames = []

    for current_frame in range(len(frames)):
        frame = cv2.imread(INPUT_DIR + '/' + base_name + '/' + frames[current_frame])

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampling_frames.append(grey_frame)

        if current_frame % estimation_size == (estimation_size-1):
            # generate a background image from the sampling 
            background_est = np.mean(np.array(sampling_frames), axis=0)
            bg_estimations.append(background_est)

            # reset sampling pool 
            sampling_frames = []

            if is_debug:
                cv2.imwrite(DEBUG_DIR + '/' + base_name + '_sample' + str(int(current_frame/estimation_size)) + '.jpg', background_est)

        if current_frame % (len(frames)/10) == 0:
            print "prep A - " + str(int(current_frame * 100.0 / len(frames))) + '%', time.time()-start_time

    if len(sampling_frames) > 0:
        background_est = np.mean(np.array(sampling_frames), axis=0)
        bg_estimations.append(background_est)

    # return the backgrounds 
    return bg_estimations

# ================================================================================= 
# ================================================================================= 
# video based tracker functions for CONTOURS 

def generate_contour_background(base_name, frames, threshold, is_debug = False): 
    # init the video capture 
    background_edges = None 

    for current_frame in range(len(frames)):
        frame = cv2.imread(INPUT_DIR + '/' + base_name + '/' + frames[current_frame])
        
        if background_edges is None: 
            background_edges = np.zeros(shape=(len(frame), len(frame[0])), dtype=np.int)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray_frame,(5,5))
        edges = cv2.Canny(blurred,25,60)

        edges_counter = edges.copy()
        edges_counter[edges_counter > 1] = 1
        background_edges += edges_counter

        if current_frame % (len(frames)/10) == 0:
            print "prep B - " + str(int(current_frame * 100.0 / len(frames))) + '%', time.time()-start_time

    background_edges = background_edges * 255 / np.max(background_edges)
    background_edges[background_edges > threshold] = 255
    background_edges[background_edges <= threshold] = 0

    if is_debug:
        video_name = video_full_name.rsplit('/',1)[-1].replace(INPUT_FILE_EXT,"")
        cv2.imwrite(video_name + '_edges.jpg', background_edges)

    return background_edges

# ================================================================================= 
# =================================================================================  
# confidence is the percentage of pix in the area of position 

def getPositionConfidence(image, position):
    # test if very low confidence
    image_checklow = np.array(image) - 50
    image_checklow[image_checklow < 0] = 0 
    if np.sum(image_checklow) == 0:
        return TINY_NUMBER 

    img = np.array(image, dtype=np.float)
    imgPos = img[max(position[0]-20,0):min(position[0]+20,img.shape[0]-1),
                max(position[1]-20,0):min(position[1]+20,img.shape[1]-1)]
    return np.sum(imgPos) / np.sum(img)

def run_bgsub_contours(base_name, sample_num = 10, is_debug = False):

    # init the video capture 
    frames = read_all_frames(base_name)

    # default management vars 
    estimation_size = len(frames) / sample_num
    locations = []

    # sample for the background estimations and edges 
    bg_estimations = generate_background_estimations(base_name, frames, estimation_size, False)
    background_edges = generate_contour_background(base_name, frames, 5, False) 

    # guesses at x and y position 
    guess_pos = None 
    bad_guesses = 0 

    for current_frame in range(len(frames)):
        frame = cv2.imread(INPUT_DIR + '/' + base_name + '/' + frames[current_frame])

        # save the previous 
        prev_pos = guess_pos

        # apply background subtraction on a grayscale image 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----
        # guess the position of the hexbug for background subtraction
        # and normalize to fit 0-255  

        result = abs(gray_frame - bg_estimations[int(current_frame/estimation_size)])
        result *= (255. / max(1,np.max(result)))
        bgsub_guess = guess_postion_wrapper(result, prev_pos, (1.0 + 1.0 * current_frame)) 

        # ----
        # guess the position of the hexbug for background edges 
        blurred = cv2.blur(gray_frame,(5,5))
        edges = cv2.Canny(blurred,25,60)
        edges = np.array(edges, dtype='uint8') - np.array(background_edges, dtype="uint8") 
        edges_guess = guess_position_denoise_bounds(edges, prev_pos, (1.0 + 1.0 * current_frame)) 

        bgsub_conf = getPositionConfidence(result, bgsub_guess)
        edges_conf = getPositionConfidence(edges, edges_guess)
        bconf = bgsub_conf / (bgsub_conf+edges_conf)
        econf = edges_conf / (bgsub_conf+edges_conf)

        # merged guesses for positions based on confidence 
        merge_guess = [(bgsub_guess[0]*bconf + edges_guess[0]*econf), (bgsub_guess[1]*bconf + edges_guess[1]*econf)] 

        dist = 0.0 
        if not prev_pos is None and not guess_pos is None and bgsub_conf > TINY_NUMBER and edges_conf > TINY_NUMBER:
            bgsub_dist = np.linalg.norm(np.array(prev_pos)-np.array(bgsub_guess))
            edges_dist = np.linalg.norm(np.array(prev_pos)-np.array(edges_guess))

            if bgsub_dist > DIST_THRESHOLD and edges_dist > DIST_THRESHOLD:
                guess_pos = prev_pos
                bad_guesses += 1 
                print "BAD GUESS", bad_guesses, "@", current_frame

                result_img = result.copy() * np.max(result) / 255.0
                miny, maxy = int(bgsub_guess[0] - 10), int(bgsub_guess[0] + 10)
                minx, maxx = int(bgsub_guess[1] - 10), int(bgsub_guess[1] + 10)
                cv2.rectangle(result_img, (minx,miny), (maxx,maxy), (200,255,100), 2)

                edges_img = edges.copy()
                miny, maxy = int(edges_guess[0] - 10), int(edges_guess[0] + 10)
                minx, maxx = int(edges_guess[1] - 10), int(edges_guess[1] + 10)
                cv2.rectangle(edges_img, (minx,miny), (maxx,maxy), (200,255,100), 2)

                
                miny, maxy = int(prev_pos[0] - 10), int(prev_pos[0] + 10)
                minx, maxx = int(prev_pos[1] - 10), int(prev_pos[1] + 10)
                cv2.rectangle(edges_img, (minx,miny), (maxx,maxy), (255,255,255), 1)
                cv2.rectangle(result_img, (minx,miny), (maxx,maxy), (255,255,255), 1)

                cv2.imwrite(DEBUG_DIR + '/' + base_name + '_res_' + index_str + '_BAD_BGSUB.jpg', result_img)
                cv2.imwrite(DEBUG_DIR + '/' + base_name + '_res_' + index_str + '_BAD_EDGES.jpg', edges_img)

            elif bgsub_dist <= DIST_THRESHOLD and edges_dist <= DIST_THRESHOLD:
                guess_pos = merge_guess
            else:
                guess_pos = edges_guess if edges_dist < bgsub_dist else bgsub_guess
        else:
            guess_pos = merge_guess

        if is_debug:
            index_str = ('000' + str(current_frame))[-4:]
            cv2.imwrite(DEBUG_DIR + '/' + base_name + '_res_' + index_str + '.jpg', result)

            index_str = ('000' + str(current_frame))[-4:]
            cv2.imwrite(DEBUG_DIR + '/' + base_name + '_edges_' + index_str + '.jpg', result)

            colored = frame.copy()

            miny, maxy = int(guess_pos[0] - 10), int(guess_pos[0] + 10)
            minx, maxx = int(guess_pos[1] - 10), int(guess_pos[1] + 10)
            cv2.rectangle(colored, (minx,miny), (maxx,maxy), (200,255,100), 2)

            cv2.imwrite(DEBUG_DIR + '/' + base_name + '_res_color_' + index_str + '.jpg', colored)

        # add location 
        locations.append((guess_pos[1], guess_pos[0]))

        if current_frame % (len(frames)/20) == 0:
            print str(int(current_frame * 100.0 / len(frames))) + '%', time.time()-start_time

    print '> done with ' + base_name
    return locations

# ================================================================================= 
# =================================================================================  

# ----------------------------------------------- 
# takes in a locations dict, writes out scaled 
# data into dict key named text files 

def writeout(filename, positions, scale_factor):
    # open/create output file 
    fo = open(OUTPUT_DIR + '/' + filename + '.txt', "w")

    for pos in positions:
        fo.write(str(int(pos[0] * scale_factor)) + ',' + str(int(pos[1] * scale_factor)) + '\n');

    # close, onwards to next 
    fo.close()

# ================================================================================= 
# =================================================================================  
# main 

# example call: python hexbug_tracker.py -file test01 -debug 1

if __name__ == "__main__":
    start_time = time.time()

    # setup 
    run_setup(True, False, True)
 
    # take in command line args 
    parser = argparse.ArgumentParser(description='Uses various CV techniques to find position of a hexbug in video frames sequence.')
    parser.add_argument('-file', dest='file', required=True, help='Name of file in input directory, i.e. "test01"')
    parser.add_argument('-debug', dest='debug', type=bool, default=False, help='Boolean flag for whether to debug; note that any value evaluates True, otherwise False')
    parser.add_argument('-sample', dest='sample', type=int, default=10, help='Count of sample frames to use to generate the background image')
    parser.add_argument('-scale', dest='scale', type=float, default=1.0, help='Postprocess scale factor if using smaller files; i.e. original 100x100 / input 50x50 should have scale of 2')
    parser.add_argument('-threshold', dest='dist_thres', type=float, default=100.0, help='Distance threshold for radius')
    args = parser.parse_args()

    DIST_THRESHOLD = args.dist_thres

    # run 
    positions = run_bgsub_contours(args.file, args.sample, args.debug)

    # write to file 
    writeout(args.file, positions, args.scale)

    # create video file 
    if args.debug:
        filename = args.file.replace(".mp4", "")
        subprocess.call(["cd", os.path.dirname(os.path.realpath(__file__))])
        subprocess.call(["ffmpeg", "-y", "-i", "debug/"+filename+"_res_color_%04d.jpg", "-pix_fmt", "yuv420p", filename+"_output.m4v"])

