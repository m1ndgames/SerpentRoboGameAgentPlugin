import numpy as np
import serpent.cv
import cv2

def readhp (p1hp_frame, p2hp_frame):
    color_almostblack = np.array([60,14,30])
    color_white = np.array([255,255,255])

    p1hp_frame_big = cv2.resize(p1hp_frame,None,fx=1, fy=10, interpolation = cv2.INTER_LINEAR)
    p1hp_hsv = cv2.cvtColor(p1hp_frame_big, cv2.COLOR_BGR2HSV)
    p1hp_mask = cv2.inRange(p1hp_hsv, color_almostblack, color_white)
    p1_mean_bgr_float = np.mean(p1hp_mask, axis=(0,1))
    p1_mean_bgr_rounded = np.round(p1_mean_bgr_float)
    p1_mean_bgr = p1_mean_bgr_rounded.astype(np.uint8)
    p1_mean_intensity = int(round(np.mean(p1hp_mask)))

    p2hp_frame_big = cv2.resize(p2hp_frame,None,fx=1, fy=10, interpolation = cv2.INTER_LINEAR)
    p2hp_hsv = cv2.cvtColor(p2hp_frame_big, cv2.COLOR_BGR2HSV)
    p2hp_mask = cv2.inRange(p2hp_hsv, color_almostblack, color_white)
    p2_mean_bgr_float = np.mean(p2hp_mask, axis=(0,1))
    p2_mean_bgr_rounded = np.round(p2_mean_bgr_float)
    p2_mean_bgr = p2_mean_bgr_rounded.astype(np.uint8)
    p2_mean_intensity = int(round(np.mean(p2hp_mask)))

    return(p1_mean_intensity, p2_mean_intensity)