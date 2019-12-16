import numpy as np
import argparse
import cv2
import os
import shutil


from utils import convert_to_binary, convert_to_binary_and_invert, display_image, get_distance_between_words
from preprocess import get_baseline_y_coord, get_horizontal_projection, get_largest_connected_component
from preprocess import segment_character, get_pen_size, get_vertical_projection, deskew, find_max_transition,\
get_cut_points, contour_seg


def segment_lines(image, directory_name, write_to_file):
    (h, w) = image.shape[:2]
    original_image = image.copy()
   
    image = cv2.bitwise_not(image)
    display_image("here", image)
    image = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    
    horizontal_projection = get_horizontal_projection(image)

    y, count = 0, 0
    is_space = False
    ycoords = []
    for i in range(h):
        if not is_space:
            if horizontal_projection[i] == 0:
                is_space = True
                count = 1
                y = i

        else:
            if horizontal_projection[i] > 0:
                is_space = False
                ycoords.append(y / count)

            else:
                y += i
                count += 1

    previous_height = 0

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)

    os.makedirs(directory_name)
    line_images = []

    for i in range(len(ycoords)):
        if i == 0:
            continue

        cv2.line(image, (0, int(ycoords[i])), (w, int(ycoords[i])), (255, 255, 255), 2) 
        image_cropped = original_image[previous_height:int(ycoords[i]), :]
        line_images.append(image_cropped)

        previous_height = int(ycoords[i])
        if write_to_file == 1:
            cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", image_cropped)
    display_image("segmented lines", image_cropped)

    image_cropped = original_image[previous_height:h, :]
    line_images.append(image_cropped)
    if write_to_file == 1:
        cv2.imwrite(directory_name + "/" + "segment_" + str(i + 1) + ".png", image_cropped)
    
    # cv2.imwrite("segmented_lines.png", image)
    return line_images

def segment_words(line_images, path, write_to_file):
    """
    this function keeps the list of word separatation points in word_separation list
    but segments into sub words and saves the sub words segements in their designated directory
    """
    # files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # image = cv2.imread(os.path.join(path, files[0]))
    directory_name = path + "/" + files[0][:-4]

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.makedirs(directory_name)

    max_segment_heights = []
    max_segment_widths = []
    all_segments = []
    avg_max_segment_height = 0
    avg_max_segment_width = 0
    for image in line_images:
        
        original_image = image.copy()
        display_image("inside", image)
        image = cv2.bitwise_not(image)
        # image_with_line = image.copy()
        
        (h, w) = image.shape
        horizontal_projection = get_horizontal_projection(image)
        baseline_y_coord = get_baseline_y_coord(horizontal_projection)
        vertical_projection = get_vertical_projection(image)

        x, count = 0, 0
        is_space = False
        xcoords = []
        distances = []

        for i in range(w):
            if not is_space:
                if vertical_projection[i] == 0:
                    is_space = True
                    count = 1
                    x = i

            else:
                if vertical_projection[i] > 0:
                    is_space = False
                    xcoords.append(x / count)
                    distances.append(count)

                else:
                    x += i
                    count += 1

        distance = get_distance_between_words(distances)

        previous_width = 0
        word_separation = xcoords.copy()
        max_segment_height = 0
        max_segment_width = 0
        for i in range(len(xcoords)):
            if i == 0:
                previous_width = int(xcoords[i])
                continue

            if distances[i-1] >= distance:
                pass
                # cv2.line(image, (previous_width, 0), (previous_width, h), (255, 255, 255), 1)
            else:
                word_separation[i-1] = -1
            sub_word = original_image[:, previous_width:int(xcoords[i])]
            all_segments.append(sub_word)
            print(sub_word.shape)

            if write_to_file == 1:
                cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", sub_word)
                # display_image("sub word", sub_word)

            if i < 2:
                max_segment_height = sub_word.shape[0]
                max_segment_width = sub_word.shape[1]
                previous_width = int(xcoords[i])
                continue
            if max_segment_height < sub_word.shape[0]:
                max_segment_height = sub_word.shape[0]
            if max_segment_width < sub_word.shape[1]:
                max_segment_width = sub_word.shape[1]

            previous_width = int(xcoords[i])

        max_segment_heights.append(max_segment_height)
        max_segment_widths.append(max_segment_width)

    
        if distances[-2] < distance:
            word_separation[-2] = -1
        
        sub_word = original_image[:, int(xcoords[-1]):w]
        all_segments.append(sub_word)
        if write_to_file == 1:
            cv2.imwrite(directory_name + "/" + "segment_" + str(len(xcoords)) + ".png", sub_word)

        #word and sub word segmentation
        # previous_width = 0
        # sub_seg_points = []
        # # word_separation = list(filter(lambda a: a != -1, word_separation))
        # flag = False
        # for i in range(len(word_separation)):
                
        #     if word_separation[i] == -1 and flag == False:
        #         flag = True
        #         sub_seg_points = []
        #         sub_seg_points.append(xcoords[i-1])
        #         # sub_seg_points.append(xcoords[i])
            
        #     if word_separation[i] == -1 and flag:
        #         sub_seg_points.append(xcoords[i])
            
        #     if word_separation[i] != -1 and flag:
        #         sub_seg_points.append(xcoords[i])
        #         flag = False
        #         # print("sub seg: ", sub_seg_points)
        #         sub_image = image[:, int(sub_seg_points[0]): int(sub_seg_points[-1])]
        #         for i in range(1, len(sub_seg_points) -1):
        #             cv2.line(image, (int(sub_seg_points[i]), 0), (int(sub_seg_points[i]), h), (255, 255, 255), 1)
        #         # display_image("display", image)

        #     previous_width = int(word_separation[i])

        # print("word: ", word_separation)
        # print("xcoord: ", xcoords)
        # display_image("final output", image)
        # cv2.imwrite("dis.png", image)

    avg_max_segment_height = sum(max_segment_heights) / len(max_segment_heights)
    avg_max_segment_width = sum(max_segment_widths) / len(max_segment_widths)
    print("avg_max_segment_height: ", avg_max_segment_height)
    print("avg_max_segment_width: ",  avg_max_segment_width)
    

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

    ap.add_argument("-w",
                    "--write-to-file",
                    required=False,
                    help="1  write line segments to disk, 0 otherwise",
                    default="0")
    
    args = vars(ap.parse_args())
    print(args)
    input_path = args["input_path"]
    line_segmets_path = args["line_segments_path"]
    write_to_file = args["write_to_file"]

    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    for f in files:
        print(f)
        image = cv2.imread(os.path.join(input_path, f))
        display_image("source", image)
        processed_image = convert_to_binary_and_invert(image)
        processed_image = deskew(processed_image)

        processed_image = cv2.bitwise_not(processed_image)
        print(processed_image.shape)
        # display_image("after deskew", processed_image)
        line_segmets_path = os.path.join(line_segmets_path, f[:-4])

        lines = segment_lines(processed_image, line_segmets_path, 0)
        segment_words(lines, line_segmets_path, 0)