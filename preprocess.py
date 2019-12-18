import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil, floor
from utils import display_image, most_frequent, convert_to_binary_and_invert


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
    print("angle: ", angle)
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

    # if most_freq_horizontal > most_freq_vertical:
    #     return most_freq_vertical
    return most_freq_vertical
    
#call on line image  to find the max transition line, above the baseline
def find_max_transition(image_original):

    image = image_original.copy()
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

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


def template_match(image, path):

    template = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    template = convert_to_binary_and_invert(template)

    if (image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]):
        return [], 0

    img = image.copy()
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where( res >= threshold)
    i = 0
    previous_point = 0
    points = []
    for pt in zip(*loc[::-1]):
        if (len(points) > 0):
            if(pt[0] - points[-1] < template.shape[1]):
                continue
            
        cv2.line(img, (pt[0], 0), (pt[0], img.shape[0]), (255, 255, 255), 1)
        cv2.line(img, (pt[0] + template.shape[1], 0), (pt[0] + template.shape[1], img.shape[0]), (255, 255, 255), 1)

        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
        points.append(pt[0])

    return points, template.shape[1]


def contour_seg(image, baseline_org):

    edged = image.copy()
    original_image = image.copy()
    character_indecies = []
    # seen_points, template_width_seen = template_match(image, "seen_start.png")
    # print("seen points", seen_points)

    # seen_end_points, template_width_seen = template_match(image, "seen_end.png")
    # print("seen end points", seen_points)

    # kaf_points, template_width_kaf = template_match(image, "kaf.png")
    # print("kaf points", kaf_points)

    # fa2_points, template_width_fa2 = template_match(image, "fa2.png")
    # print("fa2 points", fa2_points)

    # sad_points, template_width_sad = template_match(image, "sad.png")
    # print("sad points", sad_points)
    # img_cnt = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        if(cv2.contourArea(cnt) < 4):
            break

        # image_blank = np.zeros((edged.shape[0], edged.shape[1], 3), np.uint8)
        image_blank = np.zeros(edged.shape, np.uint8)
        img = cv2.drawContours(image_blank, [cnt], 0, (255, 255, 255), 1) 

        img_cnt = np.zeros(edged.shape, np.uint8)
        y_points = []
        x_points = []

        for i in range(0, cnt.shape[0]):
            point = (cnt[i][0][0], cnt[i][0][1])
            y_points.append(point[1])
            x_points.append(point[0])
            img_cnt[point[1], point[0]] = image[point[1], point[0]]
            cv2.circle(img, point, 1, (255,0,0), -1)

        hp = get_horizontal_projection(img_cnt)

        seen_points, template_width_seen = template_match(img_cnt, "seen_start.png")
        print("seen points", seen_points)

        kaf_points, template_width_kaf = template_match(img_cnt, "kaf.png")
        print("kaf points", kaf_points)

        fa2_points, template_width_fa2 = template_match(img_cnt, "fa2.png")
        print("fa2 points", fa2_points)

        sad_points, template_width_sad = template_match(img_cnt, "sad.png")
        print("sad points", sad_points)
    
        for point in seen_points:
            img_cnt[:, point:point+ template_width_seen] = 255

        for point in kaf_points:
            img_cnt[:, point:point+ template_width_kaf] = 255

        for point in fa2_points:
            img_cnt[:, point:point+ template_width_fa2] = 255

        for point in sad_points:
            img_cnt[:, point:point+ template_width_fa2] = 255

        
        cv2.imwrite("img_cnt.png", img_cnt)
        print(img_cnt.shape)
        display_image("#######################cnt", img_cnt)


        baseline = most_frequent(np.asarray(y_points))
        print("now baseline is: ", baseline)

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
                        if count > 0:
                            length_consective.append(count)
                            point_positions.append(i)
                            print("count: ", count)

                    else:
                        count += 1

        print("length_consective after: ", length_consective)
        print("point_positions after: ", point_positions)

        sub_x = []
        j = 0
        final = img_cnt.copy()
        segment_points = []
        for i in point_positions:
            sub_x = x_points[i-length_consective[j] : i]
            j += 1
            # print("sub x:", sub_x)

            # for k in range(len(sub_x)-1 , -1, -1):
            canidatate_points = []
            for k in range(len(sub_x)):
                sub_above = img_cnt[:baseline_org -2: sub_x[k]]
                sub_below = img_cnt[baseline_org +2:, sub_x[k]]
                # print("sub_above: ", sub_above)
                # print("sub_below:, ", sub_below)
                #need to add some threshold to eliminate too close seg points
                if 255 not in sub_above and 255 not in sub_below:
                    # cv2.line(final, (sub_x[k], 0), (sub_x[k], image.shape[0]), (255, 255, 255), 1)
                    # segment_points.append(sub_x[k])
                    print("there is a point")
                    canidatate_points.append(sub_x[k])
            
            if len(canidatate_points) > 0:
                print("can", canidatate_points)
                segment_points.append(canidatate_points[len(canidatate_points) // 2])
                # print("seg @@@@@: ", segment_points)
            
        if len(segment_points) < 1:
            print("@@@@@@@@@@@@@@@@@@@@@@no segmentation points")
            return []
        delete_point = False
        segment_points.sort()
        # print("segment points: ", segment_points)
        for i in range(1, len(segment_points)):
            # cv2.imwrite("sa_" + str(i) + ".png", img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
            if (img_cnt[:baseline-1, segment_points[i-1]: segment_points[i]] == 0).all():
                delete_point = True
                # print(img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
                # print("####seg :", i-1 , i)
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
        # print(last_seg_hp.shape)

        first_non_zero_index = (last_seg_hp != 0).argmax(axis=0)[0]
        # print(first_non_zero_index)


        # print(get_horizontal_projection(img_cnt[baseline - 1:baseline +2, 0:last_seg_point]))
        # print(img[baseline + 3:, 0:last_seg_point])
        # this is for the dall and zal at the end of a sentence
        if (first_non_zero_index / last_seg_hp.shape[0]) < 0.85 and (last_seg_hp[first_non_zero_index:] != 0).all()\
            and (img[baseline - 1:baseline + 2, 0:last_seg_point] != 0).any()\
            and (img[0:baseline - 2, 0:last_seg_point] == 0).all()\
            and (img[baseline + 3:, 0:last_seg_point] == 0).all(): 
            # print("here", last_seg_hp[first_non_zero_index:])
            segment_points = segment_points[1:]
            print("this is a dal at the end")

        # if len(segment_points) > 0:
        #     if(255 in image[:baseline-1 , segment_points[0]]):
        #         print("this is a ta2 or kaf at the end")
        #         segment_points[0] = -1

        previous_width = 0  
        char_list = []
        segment_points = list(filter(lambda a: a != -1, segment_points))

        for seg_point in segment_points:
            if i == 0:
                previous_width = seg_point
                continue

            character = original_image[:, previous_width:seg_point]
            char_list.append(character)
            previous_width = seg_point
            cv2.line(final, (seg_point, 0), (seg_point, image.shape[0]), (255, 255, 255), 1)
            character_indecies.append(seg_point)
    
        character = original_image[:, seg_point:original_image.shape[1]]
        char_list.append(character)
       
        print("final",segment_points)
        display_image("final", final)
        cv2.imwrite("final.png",final)
        return character_indecies
        # return(seg_point)
        
