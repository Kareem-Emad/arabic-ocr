import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil, floor
from utils import display_image, most_frequent


def get_baseline_y_coord(horizontal_projection):

    baseline_y_coord = np.where(horizontal_projection == np.amax(horizontal_projection))
    return baseline_y_coord[0][0]


def get_horizontal_projection(image):

    h, w = image.shape
    horizontal_projection = cv2.reduce(src=image, dim=-1, rtype=cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    plt.plot(range(h), horizontal_projection.tolist())
    plt.savefig("./figs/horizontal_projection.png")
    return horizontal_projection


def get_vertical_projection(image):

    h, w = image.shape
    vertical_projection = []
    vertical_projection = cv2.reduce(src=image, dim=0, rtype=cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    plt.plot(range(w), vertical_projection[0])
    plt.savefig("./figs/vertical_projection.png")
    return vertical_projection[0]


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


def get_pen_size(image):

    vertical_projection = get_vertical_projection(image)
    most_freq_vertical = most_frequent(vertical_projection)

    horizontal_projection = get_horizontal_projection(image)
    most_freq_horizontal = most_frequent(horizontal_projection)

    
    # print("most frq hor: ", most_freq_horizontal)

    if most_freq_horizontal > most_freq_vertical:
        return most_freq_vertical
    return most_freq_horizontal
    
#call on line image  to find the max transition line, above the baseline
def find_max_transition(image_original):

    image = image_original.copy()
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

    horizontal_projection = get_horizontal_projection(image)
    baseline = get_baseline_y_coord(horizontal_projection)

    max_transitions = 0
    max_transition_line = baseline
    h, w = image.shape

    for i in range(baseline, -1, -1):
        current_transitions = 0
        flag = 0

        for j in range(w-1, -1, -1):

            if image[i,j] == 255 and flag == 0:
                current_transitions += 1
                flag = 1

            elif image[i,j] != 255 and flag == 1:
                flag = 0

        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            max_transition_line = i

    

    cv2.line(image, (0, max_transition_line), (w, max_transition_line), (255, 255, 255), 1)
    cv2.line(image, (0, baseline), (w, baseline), (255, 255, 255), 1)
    display_image("max transitions", image)

    return max_transition_line
    

def get_start_end_points_sr(image, max_transition_index):

    flag = 0
    image_co = image.copy()
    separation_regions = []
    h, w = image.shape
    sr = [-1, -1]  #white to black --> start
    for j in range(w-1, -1, -1): # black to white --> end
        if image[max_transition_index,j] == 255 and flag == 0:
            sr[1] = j
            flag = 1
        elif image[max_transition_index,j] != 255 and flag == 1:
            flag = 0
            sr[0] = j

        if not -1 in sr:
            separation_regions.append(sr)
            sr = [-1, -1]

    for sr in separation_regions:
        cv2.line(image_co, (sr[0], 0), (sr[0], h), (255, 255, 255), 1)  # for debugging
        cv2.line(image_co, (sr[1], 0), (sr[1], h), (255, 255, 255), 1)  # for debugging

    display_image("after ", image_co)
    print(separation_regions)

def get_cut_points(image, max_transition_index, vertical_projection):
    
    get_start_end_points_sr(image, max_transition_index)
    # most_freq_vertical = most_frequent(vertical_projection)
    # flag = 0
    # h, w= image.shape
    # separation_regions = []
    # for j in range(w):
    #     sr = [-1, -1, -1]
    #     if image[max_transition_index, j] == 255 and flag == 0:
    #         sr[1] = j #set the end index of the current sr
    #         flag = 1
    #     elif image[max_transition_index, j] != 255 and flag == 1:
    #         sr[0] = j# set the start
    #         middle_index = (sr[0] + sr[1]) // 2
    #         vp = vertical_projection[sr[0]:sr[1]]
    #         if 0 in vp:
    #             temp = [i for i, e in enumerate(vp) if e == 0]
    #             min_distance_index = min(temp, key= lambda x:abs(x-middle_index))
    #             sr[2] = min_distance_index #line 17
            
    #         if vertical_projection[middle_index] == most_freq_vertical:
    #             sr[2] = middle_index
    #         if any (y <= most_freq_vertical for y in vp):
    #             emp = [i for i, e in enumerate(vp) if e <= most_freq_vertical]
    #             min_distance_index = min(temp, key= lambda x:abs(x-middle_index))
    #             sr[2] = min_distance_index #25
    #         else:
    #             sr[2] = middle_index
            
    #     separation_regions.append(sr)
    #     flag = 0
    
    # print("sr: ", separation_regions, sep='\n')
    


def segment_character(image):

    pen_size = get_pen_size(image)
    vertical_projection = get_vertical_projection(image)

    positions = np.where(vertical_projection == pen_size)
    print("pen size is: ", pen_size)
    print("positions is: ", positions[0], sep='\n')
    positions = positions[0]

    count = 0
    consective = False
    length_consective = []
    point_positions = []
    for i in range(1, len(positions)):

        if not consective:
            if positions[i-1] + 1 == positions[i]:
                count  = 1
                consective = True

        else:
            if positions[i-1] + 1 != positions[i]:
                consective = False
                if(count > (pen_size/255) * 0.4):
                    length_consective.append(count+1)
                    point_positions.append(i)


            else:
                count += 1

    print("point positions is", point_positions)
    print("length_consective is", length_consective)
    print("postions is: ", positions)

    segmenataion_points = []
    for i in range(len(length_consective)):
        temp = positions[point_positions[i] - length_consective[i]:point_positions[i]]
        print("final point positions",temp)
        if len(temp) != 0:
            segmenataion_points.append(ceil(sum(temp)/len(temp)))

    print("final seg points", segmenataion_points)

    previous_width = 0
    (h,w) = image.shape
    for i in segmenataion_points:
        cv2.line(image, (i, 0), (i, h), (255, 255, 255), 1)

    # cv2.line(image, (segmenataion_points[-1], 0), (segmenataion_points[-1], h), (255, 255, 255), 1)
    display_image("char seg",image)