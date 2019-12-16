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

    image = cv2.imread("./segmented_lines/csep1622/segment_1/segment_41.png")
    display_image("source", image)

    processed_image = convert_to_binary_and_invert(image)
  
    # cv2.imwrite("binary.png", processed_image)
    image = processed_image.copy() # original copy of the image
    img_line = processed_image.copy()

    edged = processed_image
    # cv2.imwrite("img_cnt.png", edged)

    hp = get_horizontal_projection(edged)
    baseline = get_baseline_y_coord(get_horizontal_projection(processed_image))
    
    print("now baseline is: ", baseline)
    h , w = processed_image.shape
    cv2.line(img_line, (0, baseline), (w, baseline), (255, 255, 255), 1)
    # cv2.imwrite("baseline.png", img_line)

    pen_size = get_pen_size(edged)
    print("pen_size", pen_size)

    y = edged[:, baseline]
    print("y", y)
    exit()
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
                if count > 2:
                    length_consective.append(count)
                    point_positions.append(i)

            else:
                count += 1

    print("length_consective: ", length_consective)
    print("point_positions: ", point_positions)
    # print(list(y_points[x] for x in point_positions))
    print("x_points: ", x_points)
    print("y_points: ", y_points)
    sub_x = []
    j = 0

    for i in point_positions:
        sub_x = x_points[i-length_consective[j] : i]
        j += 1
        print("sub x:", sub_x)

        for k in range(len(sub_x)):
            sub = img_cnt[:baseline, sub_x[k]]
            print("sub: ", sub)
            #need to add some threshold to eliminate too close seg points
            if 255 not in  sub:
                cv2.line(img_cnt, (sub_x[k], 0), (sub_x[k], image.shape[0]), (255, 255, 255), 1)
                break

    display_image("final", img_cnt)
    cv2.imwrite("final.png", img_cnt)
