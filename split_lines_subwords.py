import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import os
import shutil


def most_frequent(List):
    return max(set(List), key=List.count)


def display_image(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_binary(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh


def convert_to_binary_and_invert(image):
    # convert to greyscale then flip black and white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh


def get_base_line_y_coord(horizontal_projection):

    base_line_y_coord = np.where(horizontal_projection == np.amax(horizontal_projection))
    return base_line_y_coord[0][0]


def get_horizontal_projection(image):

    h, w = image.shape
    horizontal_projection = cv2.reduce(src=image, dim=-1, rtype=cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    plt.plot(range(h), horizontal_projection.tolist())
    plt.savefig("./figs/horizontal_projection.png")
    return horizontal_projection


def get_vertical_projection(image):

    h, w = image.shape
    vertical_projection = []
    count = 0
    for i in range(w):
        count = 0
        for j in range(h):
            count += image[j, i]
        vertical_projection.append(count)

    plt.plot(range(len(vertical_projection)), vertical_projection)
    plt.savefig("./figs/vertical_projection.png")
    return vertical_projection


def deskew(image):
    # get all white pixels coords (the foreground pixels)
    coords = np.column_stack(np.where(image > 0))
    # minAreaRect computes the minimum rotated rectangle that contains the entire text region.
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # now rotate the image with the obtained angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


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
    print("max label is: ", max_label)
    image2 = np.zeros(output.shape)

    image2[output == max_label] = 255
    image2 = image2.astype(np.uint8)
    display_image("Biggest component", image2)

    return image2


def get_distances(image, base_line):
    # print("these are the pixels along the baseline",image[base_line,:])
    h, w = image.shape
    cv2.circle(image, (h // 2, w // 2), 20, (0, 0, 0), 5)
    # pixels_along_baseline = image[base_line,:]
    is_first_point = False
    distances = []
    previous_point = 0
    for i in range(w):
        if image[base_line, i] == 255:
            if is_first_point is False:
                is_first_point = True
                previous_point = i
                print("this is the first point")
                continue
            if is_first_point:
                distances.append(i - previous_point)
                print("difference is: ", i - previous_point)
                previous_point = i

                # cv2.circle(image, (i, base_line), 3, (0, 0, 255), -1)
    # display_image("pen_size", image)


def get_pen_size(image):

    vertical_projection = get_vertical_projection(image)
    most_freq_vertical = most_frequent(vertical_projection)

    horizontal_projection = get_horizontal_projection(image)
    (values, counts) = np.unique(horizontal_projection, return_counts=True)
    most_freq_horizontal = np.argmax(counts)

    if most_freq_horizontal > most_freq_vertical:
        return most_freq_vertical
    return most_freq_horizontal


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

    # image_with_line = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)  # needs some tuning
    horizontal_projection = get_horizontal_projection(image)
    base_line_y_coord = get_base_line_y_coord(horizontal_projection)
    cv2.line(image_with_line, (0, base_line_y_coord), (w, base_line_y_coord), (255, 255, 255), 1)
    largest_connected_component = get_largest_connected_component(image_with_line)

    image_without_dotting = cv2.bitwise_and(largest_connected_component, original_image)

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
        # display_image("sub word",sub_word)
        previous_width = int(xcoords[i])

    cv2.line(image, (int(xcoords[-1]), 0), (int(xcoords[-1]), h), (255, 255, 255), 1)
    sub_word = image_without_dotting[:, int(xcoords[-1]):w]
    display_image("sub word", sub_word)
    get_pen_size(sub_word)

    display_image("final output", image)


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

        # processed_image = segment_lines(processed_image, line_segmets_path)
        segment_words_dilate(line_segmets_path)
