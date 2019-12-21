import numpy as np # noqa
import cv2
from utils import display_image # noqa
from integrator import validation_map, augment_with_compsities


def get_largest_connected_component(image):
    # image = cv2.erode(image, np.ones((2,2), np.uint8), iterations=1)
    # image = cv2.dilate(image,  np.ones((2,2), np.uint8), iterations=1)

    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    # display_image("after erode+dilate", image)
    number_of_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, number_of_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    # print("max label is: ", max_label)
    # image2 = np.zeros(output.shape)

    # image2[output == max_label] = 255
    # image2 = image2.astype(np.uint8)
    # display_image("Biggest component", output)
    output[output == max_label] = 0
    return output, max_label


def remove_dots(image):
    # image = cv2.erode(image, np.ones((2,2), np.uint8), iterations=1)
    # image = cv2.dilate(image,  np.ones((2,2), np.uint8), iterations=1)

    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    # display_image("after erode+dilate", image)
    number_of_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, number_of_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    # print("max label is: ", max_label)
    image2 = np.zeros(output.shape)

    image2[output == max_label] = 255
    image2 = image2.astype(np.uint8)
    # display_image("Biggest component", output)
    return image2


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


def sum_adjancent_values(arr):
    curr_val = arr[0]
    sum = 0
    if (curr_val == 1):
        sum += 1

    for i in range(0, len(arr)):
        if (curr_val == 0 and arr[i] > 0):
            sum += 1
            curr_val = 1
        if (curr_val == 1 and arr[i] <= 0):
            curr_val = 0
    return sum


def get_interest_points(transitions_columns, transitions_rows, img):
    interest_points = []

    for i in range(0, transitions_columns.shape[0]):
        if (transitions_columns[i] >= 4):
            start_row = -1
            end_row = -1
            for j in range(0, img.shape[0]):
                if (j != 0 and img[j][i] != img[j - 1][i]):
                    if (start_row == -1):
                        start_row = j
                    else:
                        end_row = j

            interest_point = (int((start_row + end_row) / 2), i)
            if (img[interest_point[0]][interest_point[1]] == 0):
                interest_points.append(interest_point)

    for i in range(0, transitions_rows.shape[0]):
        if (transitions_rows[i] >= 4):
            start_col = -1
            end_col = -1
            for j in range(0, img.shape[1]):
                if (j != 0 and img[i][j] != img[i][j - 1]):
                    if (start_col == -1):
                        start_col = j
                    else:
                        end_col = j
            interest_point = (i, int((start_col + end_col) / 2))
            if (img[interest_point[0]][interest_point[1]] == 0):
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
            while (h > curr_pt[0] and w > curr_pt[1] and curr_pt[0] >= 0 and curr_pt[1] >= 0):
                if (curr_pt in interest_ponts):
                    # print(f"Point {curr_pt} has been visited by {pt}")
                    interest_ponts.remove(curr_pt)
                if (img[curr_pt[0]][curr_pt[1]] == 255):
                    blocked_dirs.append(dir)
                    break
                curr_pt = (curr_pt[0] + dir[0], curr_pt[1] + dir[1])

        if (len(blocked_dirs) == len(directions)):
            if ((pt, 'HOLE') not in labeled_points):
                labeled_points.append((pt, 'HOLE'))
        else:

            label = 'CONC'
            if (W not in blocked_dirs):
                label = 'L_CONC'
            else:
                if (W in blocked_dirs and S in blocked_dirs and E in blocked_dirs and (N not in blocked_dirs or NE not in blocked_dirs or NW not in blocked_dirs)): # noqa
                    label = 'U_CONC'
                else:
                    if (E not in blocked_dirs):
                        label = 'R_CONIC'
                    else:
                        if ((W in blocked_dirs and N in blocked_dirs and E in blocked_dirs and (S not in blocked_dirs or SE not in blocked_dirs or SW not in blocked_dirs))): # noqa
                            label = 'D_CONIC'

            if ((pt, label) not in labeled_points):
                labeled_points.append((pt, label))

    return labeled_points


def eliminate_extra_padding(img):
    horz_sum = np.sum(img, axis=1)
    ver_sum = np.sum(img, axis=0)
    upper_x = -1
    upper_y = -1
    lower_x = -1
    lower_y = -1
    for i in range(0, horz_sum.shape[0]):
        if (horz_sum[i] != 0):
            if (upper_x == -1):
                upper_x = i
            else:
                lower_x = i

    for i in range(0, ver_sum.shape[0]):
        if (ver_sum[i] != 0):
            if (upper_y == -1):
                upper_y = i
            else:
                lower_y = i
    return img[upper_x:lower_x + 1, upper_y:lower_y + 1]


def is_hamza(dots_img):
    v_t = calculate_vertical_transitions(dots_img)
    if (np.max(v_t) >= 4):
        return True
    else:
        return False


def is_3_dots_connected(dots_img):
    h_t = calculate_horizonatal_transitions(dots_img)
    if (np.max(h_t) >= 4):
        return True
    else:
        return False


def recognize_dots(char_img):
    dots_img, max_label = get_largest_connected_component(char_img)
    max_label = max(np.max(dots_img), max_label)
    if (max_label == 1):
        return -1, 0, 0
    if (max_label == 2):
        if (is_hamza(dots_img)):
            max_label = 5  # hamza label is 4
        else:
            if (is_3_dots_connected(dots_img)):
                max_label = 4

    horizontal_sums = np.sum(char_img, axis=1)

    char_highest_point = -1
    for i in range(0, horizontal_sums.shape[0]):
        if (horizontal_sums[i] != 0):
            char_highest_point = i
            break

    dots_horz_sum = np.sum(dots_img, axis=1)
    lowest_dots_point = -1
    for i in range(0, dots_horz_sum.shape[0]):
        if (dots_horz_sum[i] != 0):
            lowest_dots_point = i

    highest_dots_point = -1
    for i in range(0, dots_horz_sum.shape[0]):
        if (dots_horz_sum[i] != 0):
            highest_dots_point = i
            break

    if (char_highest_point == highest_dots_point):
        return 1, 1, max_label - 1  # upper pos

    char_lowest_point = -1
    for i in range(0, horizontal_sums.shape[0]):
        if (horizontal_sums[i] != 0):
            char_lowest_point = i

    if (char_lowest_point == lowest_dots_point):
        return 3, 1, max_label - 1  # under pos

    return 2, 1, max_label - 1  # mid pos


def add_extra_padding(char_img):
    hpad = np.zeros((char_img.shape[0], 1))

    char_img = np.hstack((char_img, hpad))
    char_img = np.hstack((hpad, char_img))

    vpad = np.zeros((1, char_img.shape[1]))

    char_img = np.vstack((char_img, vpad))
    char_img = np.vstack((vpad, char_img))
    return char_img


def recognize_char(char_img):

    # segmented_char/3een_start.png
    img_dotted = char_img.copy()
    char_img = add_extra_padding(remove_dots(char_img))
    # display_image('no dots', char_img)

    horz_transitions = calculate_horizonatal_transitions(char_img)
    ver_transitions = calculate_vertical_transitions(char_img)

    interest_pts = get_interest_points(ver_transitions, horz_transitions, char_img)

    labeled_pts = label_interest_points(interest_pts, char_img.shape[1], char_img.shape[0], char_img)
    score = 0
    has_hole = 0
    for lpt in labeled_pts:
        label = lpt[1]
        if (label == 'HOLE'):
            score += 1
            has_hole = 1
        if (label == 'L_CONC'):
            score += 4
        if (label == 'R_CONIC'):
            score += 4**2
        if (label == 'U_CONC'):
            score += 4**3
        if (label == 'D_CONIC'):
            score += 4**4

    if (char_img.shape[1] == 0 or char_img.shape[0] == 0):
        return []
    char_img = eliminate_extra_padding(img_dotted)
    if (char_img.shape[0] * char_img.shape[1] < 2):
        return []
    try:
        form_ratio = char_img.shape[0] / char_img.shape[1]
    except Exception:
        return []

    char_form = -1
    if (form_ratio < 0.8):
        char_form = 1
    if (form_ratio >= 0.8 and form_ratio < 1.2):
        char_form = 2
    if (form_ratio > 1.2):
        char_form = 3

    h, w = char_img.shape
    try:
        corvar = (char_img[0][0] / 255) * 1 + (char_img[0][w - 1] / 255) * 2 + (
            char_img[h - 1][w - 1] / 255) * 4 + (char_img[h - 1][0] / 255) * 8  # noqa
    except Exception:
        return []

    pospunc, expunc, numpunc = recognize_dots(img_dotted)
    hmax = np.max(horz_transitions)
    vmax = np.max(ver_transitions)
    if(hmax < 4):
        hmax = 0
    if(vmax < 4):
        vmax = 0
    feature_vector = [score, char_form, corvar, expunc, pospunc, numpunc, hmax, vmax, has_hole]
    return feature_vector


def validate_segment(fv, text_word, current_char_idx):
    validations = validation_map[text_word[current_char_idx]]
    is_valid = True
    for validate in validations:
        if(not validate(fv)):
            is_valid = False
            break
    return is_valid


def batch_get_feat_vectors(word, idxes, text_word):
    # text_word = augment_with_compsities(text_word)
    idxes.append(word.shape[1] - 1)
    feat_vectors = []
    last_idx = 0
    good_cuts = []
    # curr_char_idx = len(text_word) - 1
    for idx in idxes:
        idx = int(idx)
        last_idx = int(last_idx)
        try:
            fv = recognize_char(word[:, last_idx:idx])
            if(fv != []):  # and validate_segment(fv, text_word, curr_char_idx) is True):
                feat_vectors.append(fv)
                last_idx = idx
                good_cuts.append(idx)
                # curr_char_idx -= 1
        except Exception:
            # print(e)
            pass
            # feat_vectors.append([])
    return feat_vectors, good_cuts
