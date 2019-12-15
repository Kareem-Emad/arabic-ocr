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

    image = cv2.imread("./segmented_lines/csep1622/segment_1/segment_28.png")
    display_image("source", image)

    processed_image = convert_to_binary_and_invert(image)
  
    cv2.imwrite("binary.png", processed_image)

    image = processed_image.copy() # original copy of the image
    img_line = processed_image.copy()

    edged = processed_image
    # edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    display_image("after closing", edged)
    cv2.imwrite("closed.png", edged)
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
    
    cv2.circle(img, leftmost, 1, (255,0,0), -1)
    cv2.circle(img, rightmost, 1, (0,255,0), -1)
    cv2.circle(img, topmost, 1, (0,0,255), -1)
    cv2.circle(img, bottommost, 1, (255,255,0), -1)

    cv2.imwrite("cnt.png", img)
    display_image("contour", img)

    index_left = np.where((cnt == leftmost).all(axis=2))
    index_right = np.where((cnt == rightmost).all(axis=2))
    index_top = np.where((cnt == topmost).all(axis=2))
    index_bottom = np.where((cnt == bottommost).all(axis=2))

    print("left: ", index_left[0][0])
    print("right: ", index_right[0][0])
    print("cnt shape", cnt.shape)
    
    # right il 2a5r
    # top lil left


    img_cnt = np.zeros(edged.shape, np.uint8)
    y_points = []
    x_points = []

    for i in range(0, cnt.shape[0]):
        point = (cnt[i][0][0], cnt[i][0][1])
        y_points.append(point[1])
        x_points.append(point[0])
        img_cnt[point[1], point[0]] = image[point[1], point[0]]
        cv2.circle(img, point, 1, (255,0,0), -1)

    # img_cnt = cv2.morphologyEx(img_cnt, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    cv2.imwrite("img_cnt.png", img_cnt)

    hp = get_horizontal_projection(img_cnt)
    baseline_org = get_baseline_y_coord(get_horizontal_projection(processed_image))

    print("original baseline: ", baseline_org)
    
    baseline = get_baseline_y_coord(hp)
    baseline = most_frequent(np.asarray(y_points))
    print("now baseline is: ", baseline)
    h , w = processed_image.shape
    cv2.line(img_line, (0, baseline), (w, baseline), (255, 255, 255), 1)
    cv2.imwrite("baseline.png", img_line)

    pen_size = get_pen_size(img_cnt)
    print("pen_size", pen_size)

    
    min_y = min(y_points)
    print("y_min: ", min_y)
    print("y_points:", y_points)

    count = 0
    flag = False
    length_consective = []
    point_positions = []
    for i in range(len(y_points)):

       if not flag:
           if y_points[i] == baseline or y_points[i] + 1 == baseline or y_points[i] -1 == baseline:
               count = 1
               flag = True
       else:
            if not(y_points[i] == baseline or y_points[i] + 1 == baseline or y_points[i] -1 == baseline):
                flag = False
                if count > 2:
                    length_consective.append(count)
                    point_positions.append(i)
                    print("count: ", count)

            else:
                count += 1

    print("length_consective: ", length_consective)
    print("point_positions: ", point_positions)
    # print(list(y_points[x] for x in point_positions))
    print("x_points: ", x_points)
    print("y_points: ", y_points)
    sub_x = []
    j = 0

    final = img_cnt.copy()
    segment_points = []
    for i in point_positions:
        sub_x = x_points[i-length_consective[j] : i]
        j += 1
        print("sub x:", sub_x)

        # for k in range(len(sub_x)-1 , -1, -1):
        canidatate_points = []
        for k in range(len(sub_x)):
            sub_above = img_cnt[:baseline, sub_x[k]]
            # sub_below = img_cnt[baseline + 2:, sub_x[k]]
            # print("sub_above: ", sub_above)
            # print("sub_below:, ", sub_below)
            #need to add some threshold to eliminate too close seg points
            if 255 not in sub_above:
                # cv2.line(final, (sub_x[k], 0), (sub_x[k], image.shape[0]), (255, 255, 255), 1)
                # segment_points.append(sub_x[k])
                print("there is a point")
                canidatate_points.append(sub_x[k])
        
        if len(canidatate_points) > 0:
            print("can", canidatate_points)
            segment_points.append(canidatate_points[len(canidatate_points) // 2])
            print("seg @@@@@: ", segment_points)
        
    if len(segment_points) < 1:
       #return
       exit()
    delete_point = False
    segment_points.sort()
    print("segment points: ", segment_points)
    for i in range(1, len(segment_points)):
        # cv2.imwrite("sa_" + str(i) + ".png", img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
        if (img_cnt[:baseline, segment_points[i-1]: segment_points[i]] == 0).all():
            delete_point = True
            # print(img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
            print("####seg :", i-1 , i)
            print("and ",  segment_points[i-1],  segment_points[i])
            segment_points[i-1] = -1

    if delete_point:
        segment_points.remove(-1)

    if len(segment_points) > 1:
        next_last_seg_point = segment_points[1]
    else:
        next_last_seg_point = img_cnt.shape[1]

    last_seg_point = segment_points[0]
    last_seg_hp = get_horizontal_projection(img_cnt[:baseline, last_seg_point:next_last_seg_point])
    print(last_seg_hp.shape)

    first_non_zero_index = (last_seg_hp != 0).argmax(axis=0)[0]
    print(first_non_zero_index)


    # print(get_horizontal_projection(img_cnt[baseline - 1:baseline +2, 0:last_seg_point]))
    # print(img[baseline + 3:, 0:last_seg_point])
    # this is for the dall and zal at the end of a sentence
    if (first_non_zero_index / last_seg_hp.shape[0]) < 0.85 and (last_seg_hp[first_non_zero_index:] != 0).all()\
        and (img[baseline - 1:baseline + 2, 0:last_seg_point] != 0).any()\
        and (img[0:baseline - 2, 0:last_seg_point] == 0).all()\
        and (img[baseline + 3:, 0:last_seg_point] == 0).all(): 
        print("here", last_seg_hp[first_non_zero_index:])
        segment_points = segment_points[1:]
        print("this is a dal at the end")

    for seg_point in segment_points:
        if seg_point != -1:
            cv2.line(final, (seg_point, 0), (seg_point, image.shape[0]), (255, 255, 255), 1)
            
    display_image("final", final)
    cv2.imwrite("final.png", final)
