import numpy as np
import argparse
import cv2
import os
import shutil


from utils import convert_to_binary, convert_to_binary_and_invert, display_image, get_distance_between_words
from preprocess import get_baseline_y_coord, get_horizontal_projection, get_largest_connected_component
from preprocess import segment_character, get_pen_size, get_vertical_projection, deskew, find_max_transition,\
get_cut_points, contour_seg


def segment_lines(image, directory_name):
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

    for i in range(len(ycoords)):
        if i == 0:
            continue

        cv2.line(image, (0, int(ycoords[i])), (w, int(ycoords[i])), (255, 255, 255), 2) 
        image_cropped = original_image[previous_height:int(ycoords[i]), :]
        previous_height = int(ycoords[i])
        cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", image_cropped)
    display_image("segmented lines", image)

    image_cropped = original_image[previous_height:h, :]
    cv2.imwrite(directory_name + "/" + "segment_" + str(i + 1) + ".png", image_cropped)
    print(image.shape)
    cv2.imwrite("segmented_lines.png", image)
    return image


def segment_words(path):
    """
    this function keeps the list of word separatation points in word_separation list
    but segments into sub words and saves the sub words segements in their designated directory
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    image = cv2.imread(os.path.join(path, files[0]))
    directory_name = path + "/" + files[0][:-4]

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.makedirs(directory_name)

    original_image = image.copy()
    image = convert_to_binary_and_invert(image)
    image_with_line = image.copy()
    

    (h, w) = image.shape

    # image_with_line = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)  # needs some tuning
    horizontal_projection = get_horizontal_projection(image)
    baseline_y_coord = get_baseline_y_coord(horizontal_projection)
    cv2.line(image_with_line, (0, baseline_y_coord), (w, baseline_y_coord), (255, 255, 255), 1)

    # display_image("image without dotting", image_without_dotting)
    vertical_projection = get_vertical_projection(image)

    print("shape of vertical projections is: ", len(vertical_projection))

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
    print(word_separation)
    print(len(word_separation))
    for i in range(len(xcoords)):
        if i == 0:
            previous_width = int(xcoords[i])
            continue

        if distances[i-1] >= distance:
            pass
            # cv2.line(image, (previous_width, 0), (previous_width, h), (255, 255, 255), 1)
        else:
            print(i)
            word_separation[i-1] = -1
        sub_word = original_image[:, previous_width:int(xcoords[i])]
        cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", sub_word)
            # display_image("sub word", sub_word)
        previous_width = int(xcoords[i])

    if distances[-2] < distance:
        word_separation[-2] = -1
    
    sub_word = original_image[:, int(xcoords[-1]):w]
    cv2.imwrite(directory_name + "/" + "segment_" + str(len(xcoords)) + ".png", sub_word)
    display_image("sub word", sub_word)


    previous_width = 0
    word_separation = list(filter(lambda a: a != 2, word_separation))
    for i in range(len(word_separation)):
        cv2.line(image, (previous_width, 0), (previous_width, h), (255, 255, 255), 1)
        previous_width = int(word_separation[i])

    display_image("final output", image)
    cv2.imwrite("dis.png", image)


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

    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    for f in files:
        print(f)
        image = cv2.imread(os.path.join(input_path, f))
        display_image("source", image)
        processed_image = convert_to_binary_and_invert(image)
        processed_image = deskew(processed_image)

        processed_image = cv2.bitwise_not(processed_image)
        print(processed_image.shape)
        display_image("after deskew", processed_image)
        cv2.imwrite("binary.png", processed_image)
        line_segmets_path = os.path.join(line_segmets_path, f[:-4])
     
        # processed_image = segment_lines(processed_image, line_segmets_path)
        segment_words(line_segmets_path)