import numpy as np
import argparse
import cv2
import os
import shutil
from matplotlib import pyplot as plt
from utils import convert_to_binary, convert_to_binary_and_invert, display_image
from preprocess import get_base_line_y_coord, get_horizontal_projection, get_largest_connected_component
from preprocess import get_pen_size, get_vertical_projection, deskew


def segment_lines(original_image, directory_name):
    (h, w) = original_image.shape[:2]
    image = original_image.copy()

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


def template_match(img, template):
    img2 = img.copy()
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] # noqa

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img2, top_left, bottom_right, 255, 2)
        display_image("template matched image", img2)


def template_match22(img_gray, template):
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    display_image('res.png', img_gray)


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
        # display_image("sub word", sub_word)
        # cv2.imwrite("segmented_chars" + "/" + "char_" + str(i + 1) + ".png", sub_word)

        # seg_chars(sub_word)
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


def calculate_horizonatal_transitions(img):
    horz_transitions_bin = np.zeros(img.shape[0])
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (j != 0 and img[i][j] != img[i][j - 1]):
                horz_transitions_bin[i] += 1
    return horz_transitions_bin


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


def sum_adjancent_values(arr):
    curr_val = arr[0]
    sum = 0
    if(curr_val == 1):
        sum += 1

    for i in range(0, len(arr)):
        if(curr_val == 0 and arr[i] > 0):
            sum += 1
            curr_val = 1
        if(curr_val == 1 and arr[i] <= 0):
            curr_val = 0
    return sum


def get_interest_points(transitions_columns, transitions_rows, img):
    interest_points = []

    for i in range(0, transitions_columns.shape[0]):
        if(transitions_columns[i] >= 4):
            start_row = -1
            end_row = -1
            for j in range(0, img.shape[0]):
                if(j != 0 and img[j][i] != img[j - 1][i]):
                    if(start_row == -1):
                        start_row = j
                    else:
                        end_row = j

            interest_point = (int((start_row + end_row) / 2), i)
            if(img[interest_point[0]][interest_point[1]] == 0):
                print(f'[vertical`]start at {start_row} and end at {end_row} yeild point {interest_point}')
                interest_points.append(interest_point)

    for i in range(0, transitions_rows.shape[0]):
        if(transitions_rows[i] >= 4):
            start_col = -1
            end_col = -1
            for j in range(0, img.shape[1]):
                if(j != 0 and img[i][j] != img[i][j - 1]):
                    if(start_col == -1):
                        start_col = j
                    else:
                        end_col = j
            interest_point = (i, int((start_col + end_col) / 2))
            if(img[interest_point[0]][interest_point[1]] == 0):
                print(f'[horz]start at {start_col} and end at {end_col} yeild point {interest_point}')
                interest_points.append(interest_point)
    return interest_points


def label_interest_points(interest_ponts, w, h, img):
    labeled_points = []
    N = (-1, 0)
    S = (-N[0], -N[1])
    E = (0, 1)
    W = (-E[0], -E[1])

    NE = (N[0] + E[0], N[1] + E[1])
    NW = (N[0] + W[0], N[1] + W[1])
    SE = (S[0] + E[0], S[1] + E[1])
    SW = (S[0] + W[0], S[1] + W[1])

    directions = [N, S, E, W, NE, NW, SE, SW]

    for pt in interest_ponts:
        blocked_dirs = []
        for dir in directions:
            curr_pt = (pt[0] + dir[0], pt[1] + dir[1])
            while(h > curr_pt[0] and w > curr_pt[1] and curr_pt[0] >= 0 and curr_pt[1] >= 0):
                if(curr_pt in interest_ponts):
                    print(f"Point {curr_pt} has been visited by {pt}")
                    interest_ponts.remove(curr_pt)
                if(img[curr_pt[0]][curr_pt[1]] == 255):
                    blocked_dirs.append(dir)
                    break
                curr_pt = (curr_pt[0] + dir[0], curr_pt[1] + dir[1])

        if(len(blocked_dirs) == len(directions)):
            if((pt, 'HOLE') not in labeled_points):
                labeled_points.append((pt, 'HOLE'))
        else:
            if((pt, 'CONC') not in labeled_points):
                labeled_points.append((pt, 'CONC'))

    return labeled_points


def recognize_char(input_path):
    # segmented_char/3een_start.png
    char_img = convert_to_binary(cv2.imread(input_path, 0))
    char_img = (255 - char_img)
    horz_transitions = calculate_horizonatal_transitions(char_img)
    ver_transitions = calculate_vertical_transitions(char_img)

    interest_pts = get_interest_points(ver_transitions, horz_transitions, char_img)

    labeled_pts = label_interest_points(interest_pts, char_img.shape[1], char_img.shape[0], char_img)

    print(horz_transitions)
    print(ver_transitions)
    print(interest_pts)
    print(labeled_pts)
    display_image('character', char_img)


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
                    default="./segmented_char/3een_start.png")

    # ap.add_argument("-f", "--figs-path", required=False, help="path to line segments file", default="./figs") # noqa

    args = vars(ap.parse_args())
    print(args)
    input_path = args["input_path"]
    recognize_char(input_path)
    """
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
    """
