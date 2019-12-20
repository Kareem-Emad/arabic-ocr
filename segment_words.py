import numpy as np
import argparse
import cv2
import os
import shutil
import json
from utils import convert_to_binary, convert_to_binary_and_invert, display_image, get_distance_between_words
from preprocess import get_baseline_y_coord, get_horizontal_projection
from preprocess import get_vertical_projection, deskew, contour_seg
from train_recognition import batch_get_feat_vectors
from integrator import compare_and_assign, get_words_from_text, load_features_map, match_feat_to_char


def segment_lines(image, directory_name, write_to_file):
    (h, w) = image.shape
    image = convert_to_binary(image)
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
    line_images = []

    for i in range(len(ycoords)):
        if i == 0:
            continue

        cv2.line(image, (0, int(ycoords[i])), (w, int(ycoords[i])), (255, 255, 255), 2)
        image_cropped = original_image[previous_height:int(ycoords[i]), :]
        line_images.append(image_cropped)

        # line = image_cropped.copy()
        # baseline = get_baseline_y_coord(get_horizontal_projection(line))
        # cv2.line(line, (0, baseline), (w, baseline), (255, 255, 255), 1)
        # display_image("base",line)
    
        previous_height = int(ycoords[i])
        if write_to_file == 1:
            cv2.imwrite(directory_name + "/" + "segment_" + str(i) + ".png", image_cropped)
    # display_image("segmented lines", image_cropped)

    image_cropped = original_image[previous_height:h, :]
    line_images.append(image_cropped)
    if write_to_file == 1:
        cv2.imwrite(directory_name + "/" + "segment_" + str(i + 1) + ".png", image_cropped)

    # cv2.imwrite("segmented_lines.png", image)
    return line_images


def segment_words(line_images, path, img_name, input_path, train=False):
    """
    this function keeps the list of word separatation points in word_separation list
    but segments into sub words and saves the sub words segements in their designated directory
    """
    # files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # image = cv2.imread(os.path.join(path, files[1]))
    # print(os.path.join(path, files[1]))
    gt_words = get_words_from_text(img_name, input_path)
    char_map = load_features_map()

    recognized_chars = ''
    directory_name = "./segmented_words"

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.makedirs(directory_name)

    curr_word_idx = 0
    wrong_seg_words = 0
    for image in line_images:

        original_image = image.copy()
        image_with_line = image.copy()
        (h, w) = image.shape

        horizontal_projection = get_horizontal_projection(image)
        baseline_y_coord = get_baseline_y_coord(horizontal_projection)
        cv2.line(image_with_line, (0, baseline_y_coord), (w, baseline_y_coord), (255, 255, 255), 1)

        vertical_projection = get_vertical_projection(image)

        # print("shape of vertical projections is: ", len(vertical_projection))

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

        previous_width = 0
        word_separation = xcoords.copy()
        # word_separation = list(filter(lambda a: a != -1, word_separation))

        for i in range(len(word_separation)):

            if distances[i] > 2:
                pass
            else:
                word_separation[i] = -1

        word_separation = list(filter(lambda a: a != -1, word_separation))
        print(word_separation)

        previous_width = image.shape[1]
        seg_point = []
        for i in range(len(word_separation)):
            i = len(word_separation) - i - 1

            word = original_image[:, int(word_separation[i]):previous_width]
            display_image("word", word)
            cv2.line(image, (int(word_separation[i]), 0), (int(word_separation[i]), image.shape[0]), (255, 255, 255),1)
            previous_width = int(word_separation[i])
            seg_points = contour_seg(word, baseline_y_coord)
            # import ipdb; ipdb.set_trace()
            feat_vectors = batch_get_feat_vectors(word, seg_points, gt_words[curr_word_idx])
            if (train):
                if(len(gt_words) > curr_word_idx):
                    aux_map = compare_and_assign(feat_vectors, gt_words[curr_word_idx], char_map)
                    if (aux_map != -1):
                        char_map = aux_map
                    else:
                        wrong_seg_words += 1
                else:
                    wrong_seg_words += 1
            else:
                # import ipdb; ipdb.set_trace()
                recognized_chars += ' ' + match_feat_to_char(char_map, feat_vectors)
            curr_word_idx += 1
        display_image("word sep", image)
    
    if (train):
        try:
            with open('./config_map.json', 'w') as f:
                f.write(json.dumps(char_map, ensure_ascii = False))
                f.close()
                print(char_map)
                return wrong_seg_words, curr_word_idx - 1
        except Exception:
            print(char_map)
            return wrong_seg_words, curr_word_idx - 1
    else:
        print(recognized_chars)
        return 0, 0


def process_image(line_segmets_path, input_path, f):
    image = cv2.imread(os.path.join(input_path, f))
    display_image("source", image)
    processed_image = convert_to_binary_and_invert(image)
    processed_image = deskew(processed_image)

    print(processed_image.shape)
    display_image("after deskew", processed_image)
    cv2.imwrite("binary.png", processed_image)
    line_segmets_path = os.path.join(line_segmets_path, f[:-4])

    lines = segment_lines(processed_image, line_segmets_path, 0)
    curr_ww, curr_tw = segment_words(lines, line_segmets_path, f, input_path, True)
    print(f'we got {curr_ww} wrong out of {curr_tw}')


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
    nthreads = 0
    for f in files:
        process_image(line_segmets_path, input_path, f)
