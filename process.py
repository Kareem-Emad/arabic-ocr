import numpy as np
import argparse
import cv2
import os
import shutil

from utils import convert_to_binary, convert_to_binary_and_invert, display_image
from preprocess import get_base_line_y_coord, get_horizontal_projection, get_largest_connected_component
from preprocess import get_pen_size, get_vertical_projection, deskew


def segment_lines(image, directory_name):
    (h, w) = image.shape[:2]
    original_image = image.copy()

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

        cv2.line(image, (0, int(ycoords[i])), (w, int(ycoords[i])), (255, 255, 255), 2)  # for debugging
        image_cropped = original_image[previous_height:int(ycoords[i]), :]

        previous_height = int(ycoords[i])
        cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", image_cropped)

    display_image("segmented lines", image)

    image_cropped = original_image[previous_height:h, :]
    cv2.imwrite(directory_name + "/" + "segment_" + str(i + 1) + ".png", image_cropped)
    print(image.shape)
    return image


def segment_words_dilate(path):
    # should have a loop here on all files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    image = cv2.imread(os.path.join(path, files[0]), cv2.IMREAD_GRAYSCALE)
    image = convert_to_binary(image)
    image_with_line = image.copy()
    original_image = image.copy()

    (h, w) = image.shape
    print("this is image shape: ", image.shape)
    """
    img = image
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    # image = skel
    last_candidate = 0
    cum_sum = np.sum(skel, axis=0)
    separators = []
    for i in range(0, skel.shape[1]):
        if ((skel[20][i] != 0 or skel[22][i] != 0) and (skel[21][i + 1] != 0 or skel[21][i - 1] != 0)):
            print(f"point with x = 20 and y ={i} matched")
            if (i - last_candidate > 3 and cum_sum[i] != cum_sum[last_candidate]):
                separators.append(i)
                last_candidate = i

    for i in separators:
        cv2.line(skel, (i, 0), (i, 32), (255, 255, 255), 1)

    cv2.imshow("skel", skel)
    cv2.waitKey(0)
    """
    # image_with_line = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)  # needs some tuning
    horizontal_projection = get_horizontal_projection(image)
    base_line_y_coord = get_base_line_y_coord(horizontal_projection)
    cv2.line(image_with_line, (0, base_line_y_coord), (w, base_line_y_coord), (255, 255, 255), 1)
    # largest_connected_component = get_largest_connected_component(image_with_line)

    image_without_dotting = original_image  # cv2.bitwise_and(largest_connected_component, original_image)

    display_image("image without dotting", image_without_dotting)
    vertical_projection = get_vertical_projection(image)

    print("shape of vertical projections is: ", len(vertical_projection))

    x, count = 0, 0
    is_space = False
    xcoords = []

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

            else:
                x += i
                count += 1

    print("len of xcoords", len(xcoords))
    previous_width = 0

    for i in range(len(xcoords)):
        if i == 0:
            previous_width = int(xcoords[i])
            continue
        cv2.line(image, (previous_width, 0), (previous_width, h), (255, 255, 255), 1)
        sub_word = image_without_dotting[:, previous_width:int(xcoords[i])]
        get_pen_size(sub_word)
        seg_chars(sub_word)
        # display_image("sub word", sub_word)
        previous_width = int(xcoords[i])

    cv2.line(image, (int(xcoords[-1]), 0), (int(xcoords[-1]), h), (255, 255, 255), 1)
    sub_word = image_without_dotting[:, int(xcoords[-1]):w]
    display_image("sub word", sub_word)
    get_pen_size(sub_word)

    display_image("final output", image)


def calculate_hist_threshold(vertical_sums):
    (values, counts) = np.unique(vertical_sums, return_counts=True)
    ind = np.argmax(counts)
    threshold = values[ind]
    return threshold


def calculate_vertical_transitions(img):
    vertical_transitions_bin = np.zeros(img.shape[1])
    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]):
            if (j != 0 and img[j][i] != img[j - 1][i]):
                vertical_transitions_bin[i] += 1
    return vertical_transitions_bin


def get_first_match_for_a_criteria(vertical_sums, threshold, start, end):
    for i in range(start, end):
        if (vertical_sums[len(vertical_sums) - i - 1] > threshold):
            return len(vertical_sums) - i - 1
    return -1


def get_top_line_for_column(col, img, stop_point, start_point):
    for k in range(start_point, stop_point):
        if (img[k][col] != 0):
            return k
    return 33


def is_matching_b_criteria(junction_line, top_line_i, bottom_line_i, vertical_sums, vertical_transitions_bin,
                           threshold, top_line_j1, i):
    return junction_line >= top_line_i and bottom_line_i >= junction_line and vertical_sums[
        i] >= threshold and vertical_transitions_bin[i] == 2 and (
            bottom_line_i - top_line_i) <= threshold and top_line_i > top_line_j1


def seg_chars(img):
    img = img.copy()
    """
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(img.shape, np.uint8)
    size = np.size(img)
    done = False
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    """

    vertical_sums = np.sum(img, axis=0)
    threshold = calculate_hist_threshold(vertical_sums)

    horizontal_sums = np.sum(img, axis=1)
    vertical_transitions_bin = calculate_vertical_transitions(img)

    junction_line = np.argmax(horizontal_sums)
    j1 = 0
    j2 = -1
    j1_dash = 0
    top_line_j1 = 33
    deb = True
    first_time = True
    while (True):
        if (deb is True):
            deb = False
            j1 = -1
            j1 = get_first_match_for_a_criteria(vertical_sums, threshold, 0, len(vertical_sums))
            j2 = j1 + 1
            top_line_j1 = get_top_line_for_column(j1, img, img.shape[0], 0)
            if (j1 == -1):
                break
        else:
            j1 = j2 - 1
            j1_dash = -1
            """
            for i in range(j1 + 1, len(vertical_sums)):
                if(vertical_sums[i] > threshold):
                    j1_dash = i
                    break
            """
            if (j1_dash == -1):
                j1_dash = 0

            j2 = -1
            for i in range(j1_dash, j1):
                top_line_i = get_top_line_for_column(j1 - i - 1, img, img.shape[0], 0)
                bottom_line_i = 0
                for k in range(0, img.shape[0]):
                    if (k != 0 and img[img.shape[0] - k - 1][j1 - i - 1] != 0):
                        bottom_line_i = img.shape[0] - k - 1
                        break
                # import ipdb; ipdb.set_trace()
                is_matching = is_matching_b_criteria(junction_line, top_line_i, bottom_line_i, vertical_sums,
                                                     vertical_transitions_bin, threshold, top_line_j1, i)
                if (is_matching):
                    j2 = j1 - i - 1
                    j3 = j2
                    for i in range(j1_dash, j2):
                        top_line_i = get_top_line_for_column(j2 - i - 1, img, img.shape[0], 0)

                        bottom_line_i = 0
                        for k in range(0, img.shape[0]):
                            if (k != 0 and img[img.shape[0] - k - 1][j2 - i - 1] != 0):
                                bottom_line_i = img.shape[0] - k - 1
                                break

                        is_matching = is_matching_b_criteria(junction_line, top_line_i, bottom_line_i,
                                                             vertical_sums, vertical_transitions_bin,
                                                             threshold, top_line_j1, i)
                        if (is_matching):
                            j3 = i
                            break

                    if (j2 != j3 and vertical_sums[j3] == vertical_sums[j2]):
                        continue
                    if (first_time is True):
                        first_time = False
                        cv2.line(img, (j1, 0), (j1, 32), (255, 255, 255), 1)
                    cv2.line(img, (j2, 0), (j2, 32), (255, 255, 255), 1)
                    print(f" found character starting at {j1} and ending at {j2}")
                    break
            if (j2 == -1):
                break
    print("End OF CURRENT WORD CHARACTERS")
    display_image("sub word", img)


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
    # ap.add_argument("-f", "--figs-path", required=False, help="path to line segments file", default="./figs") # noqa

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
        processed_image = convert_to_binary(processed_image)
        display_image("after deskew", processed_image)

        processed_image = segment_lines(processed_image, line_segmets_path)
        segment_words_dilate(line_segmets_path)
