import cv2
import numpy as np


def most_frequent(List):
    if(type(List) is np.ndarray):
        List = List.tolist()
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
