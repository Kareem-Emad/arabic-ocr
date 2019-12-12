import numpy as np
import argparse
import cv2
import os
import shutil
from scipy.signal import argrelextrema
from math import ceil, floor


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

    image = cv2.imread("word2.png")
    display_image("source", image)
    processed_image = convert_to_binary_and_invert(image)
  
    cv2.imwrite("binary.png", processed_image)
    image = processed_image.copy() # original copy of the image
    edged = processed_image


    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    image_blank = np.zeros((edged.shape[0], edged.shape[1], 3), np.uint8)
    max_cont = max(contours, key=cv2.contourArea)
    image_blank = np.zeros(edged.shape, np.uint8)
    img = cv2.drawContours(image_blank, [max_cont], 0, (255, 255, 255), 1) 

    cnt = max_cont

    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])   
    

    # cv2.circle(img, leftmost, 1, (255,0,0), -1)
    # cv2.circle(img, rightmost, 1, (0,255,0), -1)
    # cv2.circle(img, topmost, 1, (0,0,255), -1)
    # cv2.circle(img, bottommost, 1, (255,255,0), -1)

    index_left = np.where((cnt == leftmost).all(axis=2))
    index_right = np.where((cnt == rightmost).all(axis=2))
    index_top = np.where((cnt == topmost).all(axis=2))
    index_bottom = np.where((cnt == bottommost).all(axis=2))

    print("left: ", index_left[0][0])
    print("right: ", index_right[0][0])
    print("cnt shape", cnt.shape)
    
    # right il 2a5r
    # top lil left


    img_cnt = np.zeros(image.shape, np.uint8)
    y_points = []
    x_points = []
    upper_cnt =[]
    for i in range(index_right[0][0], cnt.shape[0]):
        point = (cnt[i][0][0], cnt[i][0][1])
        y_points.append(point[1])
        x_points.append(point[0])
        img_cnt[point[1], point[0]] = image[point[1], point[0]]
        cv2.circle(img, point, 1, (255,0,0), -1)

    # img_cnt = cv2.morphologyEx(img_cnt, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

    hp = get_horizontal_projection(img_cnt)
    baseline = get_baseline_y_coord(hp)
    baseline = most_frequent(y_points)
    print("now baseline is: ", baseline)
    pen_size = get_pen_size(img_cnt)
    print("pen_size", pen_size)

    cv2.imwrite("cnt.png", img)
    cv2.imwrite("img_c nt.png", img_cnt)
    min_y = min(y_points)
    print("y_min: ", min_y)
    print("y_points:", y_points)

    count = 0
    flag = False
    length_consective = []
    point_positions = []
    for i in range(len(y_points)):

       if not flag:
           if y_points[i] == baseline:
               count = 1
               flag = True
       else:
            if not(y_points[i] == baseline):
                flag = False
                if count > 1:
                    length_consective.append(count)
                    point_positions.append(i)

            else:
                count += 1

    print("length_consective: ", length_consective)
    print("point_positions: ", point_positions)
    print(list(y_points[x] for x in point_positions))
    print("x_points: ", x_points)
    peak_points = list(x_points[x] for x in point_positions)
    seg_points = []

    for point in peak_points:
        # seg_points.append(point + 1)
        seg_points.append(point - 2)
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # x = np.r_[True, y_points[1:] < y_points[:-1]] & np.r_[y_points[:-1] < y_points[1:], True]
    # b = (np.diff(np.sign(np.diff(y_points))) > 0).nonzero()[0] + 1
    # print("b: ", b)
    # print("y  points: ", y_points)
    # print("local: ", y_points[9], y_points[11], y_points[14], y_points[29])
    # seg_points = argrelextrema(np.asarray(y_points), np.less)
    # print(seg_points[0][0])
    # print(y_points[seg_points[0][0]])
    # print(y_points[seg_points[0][1]])

    # print(seg_points[1])
    # cv2.line(img_cnt, (x_points[point_positions[1]], 0), (x_points[point_positions[1]], image.shape[0]), (255, 255, 255), 1)
    # cv2.line(img_cnt, (x_points[point_positions[2]], 0), (x_points[point_positions[2]], image.shape[0]), (255, 255, 255), 1)

    # display_image("cnt",img_cnt)
    # cv2.imwrite("s.png", img_cnt)
    # for i in range(len(point_positions)):
    #     cv2.line(img_cnt, (x_points[point_positions[i]], 0), (x_points[point_positions[i]], image.shape[0]), (255, 255, 255), 1)



    for i in range(len(seg_points)):
        sub = img_cnt[:baseline, seg_points[i]]
        print("sub: ", sub)
        #need to add some threshold to eliminate too close seg points
        if 255 in sub:
            print("it's true")
            continue
        cv2.line(img_cnt, (seg_points[i], 0), (seg_points[i], image.shape[0]), (255, 255, 255), 1)


    cv2.imwrite("final.png", img_cnt)

    

    

