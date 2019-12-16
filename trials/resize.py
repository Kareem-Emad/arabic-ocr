import numpy as np
import argparse
import cv2
import os
import shutil
from scipy.signal import argrelextrema
from math import ceil, floor
from scipy.signal import argrelextrema, argrelmin, argrelmax

from utils import convert_to_binary, convert_to_binary_and_invert, display_image, thin_image, most_frequent
from preprocess import get_baseline_y_coord, get_horizontal_projection, get_largest_connected_component
from preprocess import segment_character, get_pen_size, get_vertical_projection, deskew, find_max_transition,\
get_cut_points, contour_seg



if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-o",
                    "--line-segments-path",
                    required=False,
                    help="path to line segments file",
                    default="./segmented_lines")
    ap.add_argument("-i",
                    "--input-path",
                    required=False,
                    help="path to line segments file",
                    default="./inputs")
    
    args = vars(ap.parse_args())
    print(args)
    input_path = args["input_path"]
    line_segmets_path = args["line_segments_path"]

    image = cv2.imread("./segmented_lines/csep1622/segment_1/segment_41.png")
    # image = cv2.imread("word2.png")
    display_image("source", image)

    processed_image = convert_to_binary_and_invert(image)
  
    cv2.imwrite("binary.png", processed_image)

    scale_percent = 600 # percent of original size
    width = int(processed_image.shape[1] * scale_percent / 100)
    height = int(processed_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    processed_image = cv2.resize(processed_image, dim, interpolation = cv2.INTER_AREA)


    image = processed_image.copy() # original copy of the image
    # edged = cv2.Canny(processed_image,10,100)
    # cv2.imwrite("edges.png", edged)
    edged = processed_image
    
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # image_blank = np.zeros((edged.shape[0], edged.shape[1], 3), np.uint8)
    image_blank = np.zeros(edged.shape, np.uint8)

    max_cont = max(contours, key=cv2.contourArea)

    # image_blank = np.zeros(edged.shape, np.uint8)
    img = cv2.drawContours(image_blank, [max_cont], 0, (255, 255, 255), -1) 

    cnt = max_cont

    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])   

    # cv2.circle(img, leftmost, 1, (255,0,0), -1)
    # cv2.circle(img, rightmost, 1, (0,255,0), -1)
    # cv2.circle(img, topmost, 1, (0,0,255), -1)
    # cv2.circle(img, bottommost, 1, (0,0,255), -1)


    # # cv2.imwrite("cnt.png", img)
    # display_image("contour", img)


    index_left = np.where((cnt == leftmost).all(axis=2))
    index_right = np.where((cnt == rightmost).all(axis=2))
    index_top = np.where((cnt == topmost).all(axis=2))
    index_bottom = np.where((cnt == bottommost).all(axis=2))


    # print("left: ", index_left[0][0])
    # print("right: ", index_right[0][0])
    # print("cnt shape", cnt.shape)
    
    # right il 2a5r
    # top lil left
    # img_cnt =np.zeros((edged.shape[0], edged.shape[1], 3), np.uint8)
    img_cnt =np.zeros(edged.shape, np.uint8)

    y_points = []
    x_points = []

    # for i in range(index_left[0][0], -1, -1):
    for i in range(0, cnt.shape[0]):
        point = (cnt[i][0][0], cnt[i][0][1])
        y_points.append(point[1])
        x_points.append(point[0])
        img_cnt[point[1], point[0]] = img[point[1], point[0]]
        # cv2.circle(img, point, 1, (255,0,0), -1)


    count = 0
    flag = False
    # y_diff = []
    point_positions = []
    curr = -1
    for i in range(len(y_points) -1):

        if y_points[i] == y_points[i+1]:
            if not flag:
               count = 1
               flag = True
            #    y_diff.append(y_points[i])
               point_positions.append(i)

            else:
                count += 1
    
        else:
            if flag:
                flag = False
                point_positions.append(i)
                

    print(y_points)
    print(point_positions)
    # print(y_diff)
    y_diff = y_points.copy()
    x_diff = x_points.copy()
    # point_positions.reverse()
    for i in range(0, len(point_positions) -1, 2):
        for j in range(point_positions[i], point_positions[i+1]+1):
            if j == (point_positions[i] + point_positions[i+1]) // 2:
                continue
            y_diff[j] = -1
            x_diff[j] = -1

    y_diff_filtered = list(filter(lambda a: a != -1, y_diff))
    x_diff_filtered = list(filter(lambda a: a != -1, x_diff))


    # print("y diff", y_diff)

    local_min_indices = (np.diff(np.sign(np.diff(y_diff_filtered))) > 0).nonzero()[0] + 1
    # local_min_indices = argrelextrema(np.asarray(y_points), np.less)
    print(local_min_indices)
    # print(local_min_indices)
    for i in local_min_indices:
        print(i)
        cv2.line(img, (x_diff_filtered[i], 0), (x_diff_filtered[i], img.shape[0]), (255, 255, 255), 1)

    cv2.imwrite("img_cnt.png", img_cnt)
    cv2.imwrite("max.png", img)
    print(y_diff[point_positions[0]])
    # # print("baseline: ", baseline)
