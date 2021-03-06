import numpy as np
import argparse # noqa
import cv2
from utils import convert_to_binary_and_invert, display_image, most_frequent # noqa
from preprocess import get_baseline_y_coord, get_horizontal_projection # noqa
from preprocess import get_vertical_projection # noqa


def template_match(image, path):

    template = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    template = convert_to_binary_and_invert(template)

    if (image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]):
        return [], 0

    img = image.copy()
    w, h = template.shape[::-1]
    recv2.lines = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.65
    loc = np.where(res >= threshold)
    # print("template width", template.shape[1])
    points = []
    for pt in zip(*loc[::-1]):
        if (len(points) > 0):
            if (pt[0] - points[-1] < template.shape[1]):
                print("here")
                continue

        # cv2.line(img, (pt[0], 0), (pt[0], img.shape[0]), (255, 255, 255), 1)
        # cv2.line(img, (pt[0] + template.shape[1], 0), (pt[0] + template.shape[1], img.shape[0]), (255, 255, 255), 1) # noqa

        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 2)
        points.append(pt[0])
        # display_image('res.png', img)

    return points, template.shape[1]


"""
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
    # print(args)
    input_path = args["input_path"]
    line_segmets_path = args["line_segments_path"]

    image = cv2.imread("./contour/segment_5.png", cv2.COLOR_BGR2GRAY)
    display_image("source", image)

    processed_image = image
    edged = processed_image
    vertical_projection = get_vertical_projection(image)

    x, count = 0, 0
    is_space = False
    xcoords = []
    distances = []

    for i in range(edged.shape[1]):
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
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        if (cv2.contourArea(cnt) < 1):
            # print("too small", cv2.contourArea(cnt))
            break

        image_blank = np.zeros(edged.shape, np.uint8)
        img = cv2.drawContours(image_blank, [cnt], 0, (255, 255, 255), 1)

        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        # print("left most", leftmost)
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        cv2.circle(img, leftmost, 1, (255, 0, 0), -1)
        cv2.circle(img, rightmost, 1, (0, 255, 0), -1)
        cv2.circle(img, topmost, 1, (0, 0, 255), -1)
        cv2.circle(img, bottommost, 1, (255, 255, 0), -1)

        cv2.imwrite("cnt.png", img)
        display_image("contour", img)

        index_left = np.where((cnt == leftmost).all(axis=2))
        index_right = np.where((cnt == rightmost).all(axis=2))
        index_top = np.where((cnt == topmost).all(axis=2))
        index_bottom = np.where((cnt == bottommost).all(axis=2))

        # print("cnt shape", cnt.shape)

        # right il 2a5r
        # top lil left

        img_cnt = np.zeros(edged.shape, np.uint8)
        y_points = []
        x_points = []

        for i in range(0, cnt.shape[0]):
            point = (cnt[i][0][0], cnt[i][0][1])
            y_points.append(point[1])
            x_points.append(point[0])
            cv2.circle(img, point, 1, (255, 0, 0), -1)
            img_cnt[point[1], point[0]] = image[point[1], point[0]]

        seen_points, template_width_seen = template_match(img_cnt, "seen_start.png")
        # print("seen points", seen_points)

        seen_mid_points, template_width_seen = template_match(img_cnt, "seen_mid.png")
        # print("seen mid points", seen_mid_points)

        kaf_points, template_width_kaf = template_match(img_cnt, "kaf.png")
        # print("kaf points", kaf_points)

        fa2_points, template_width_fa2 = template_match(img_cnt, "fa2.png")
        # print("fa2 points", fa2_points)

        sad_points, template_width_sad = template_match(img_cnt, "sad.png")
        # print("sad points", sad_points)

        for point in seen_points:
            img_cnt[:, point:point + template_width_seen] = 255

        for point in kaf_points:
            img_cnt[:, point:point + template_width_kaf] = 255

        for point in fa2_points:
            img_cnt[:, point:point + template_width_fa2] = 255

        for point in sad_points:
            img_cnt[:, point:point + template_width_fa2] = 255

        cv2.imwrite("img_cnt.png", img_cnt)

        hp = get_horizontal_projection(img_cnt)
        baseline_org = get_baseline_y_coord(get_horizontal_projection(processed_image))

        img_line = img_cnt.copy()
        baseline = most_frequent(np.asarray(y_points))
        # print("now baseline is: ", baseline)
        # print("now baseline_org is:", baseline_org)
        h, w = processed_image.shape
        cv2.line(img_line, (0, baseline), (w, baseline), (255, 255, 255), 1)
        cv2.line(img_line, (0, baseline_org), (w, baseline_org), (255, 255, 255), 1)

        cv2.imwrite("baseline.png", img_line)
        display_image("baseline_org", img_line)

        count = 0
        flag = False
        length_consective = []
        point_positions = []
        for i in range(len(y_points)):

            if not flag:
                if y_points[i] - 2 == baseline or y_points[
                        i] == baseline or y_points[i] + 1 == baseline or y_points[i] - 1 == baseline:
                    count = 1
                    flag = True
            else:
                if not (y_points[i] - 2 == baseline or y_points[i] == baseline or y_points[i] + 1 == baseline
                        or y_points[i] - 1 == baseline):
                    flag = False
                    if count > 2:
                        length_consective.append(count)
                        point_positions.append(i)

                else:
                    count += 1

        # print("length_consective: ", length_consective)
        # print("point_positions: ", point_positions)
        # print(list(y_points[x] for x in point_positions))
        sub_x = []
        j = 0

        final = img_cnt.copy()
        segment_points = []

        for i in point_positions:
            sub_x = x_points[i - length_consective[j]:i]
            print("sub_x", sub_x)
            j += 1
            # for k in range(len(sub_x)-1 , -1, -1):
            canidatate_points = []
            print(type(img_cnt[0, 0]))
            for k in range(len(sub_x)):
                # sub_above = img_cnt[:baseline, sub_x[k]]
                sub_above = img_cnt[int(baseline / 2):baseline, sub_x[k]]
                sub_below = img_cnt[baseline + 2:, sub_x[k]]
                # print("sub_above: ", sub_above)
                # print("sub_below:, ", sub_below)
                # need to add some threshold to eliminate too close seg points
                if 255 not in sub_above and 255 not in sub_below:
                    # cv2.line(final, (sub_x[k], 0), (sub_x[k], image.shape[0]), (255, 255, 255), 1)
                    # segment_points.append(sub_x[k])
                    # print("there is a point")
                    # print("sub below: ", sub_below)
                    canidatate_points.append(sub_x[k])

            if len(canidatate_points) > 0:
                # print("can", canidatate_points)
                segment_points.append(canidatate_points[len(canidatate_points) // 2])
                # print("seg @@@@@: ", segment_points)

        if len(segment_points) < 1:
            print("&&&&&&&&& no seg points")
            continue
        delete_point = False
        segment_points.sort()
        print("segment points: ", segment_points)
        for i in range(1, len(segment_points)):
            # cv2.imwrite("sa_" + str(i) + ".png", img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
            if (img_cnt[:baseline, segment_points[i - 1]:segment_points[i]] == 0).all():
                delete_point = True
                # print(img_cnt[:baseline, segment_points[i-1]: segment_points[i]])
                # print("####seg :", i - 1, i)
                # print("and ", segment_points[i - 1], segment_points[i])
                segment_points[i - 1] = -1

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
        if (first_non_zero_index / last_seg_hp.shape[0]) < 0.85 and (last_seg_hp[first_non_zero_index:] != 0).all() and (img[baseline - 1:baseline + 2, 0:last_seg_point] != 0).any() and (img[0:baseline - 2, 0:last_seg_point] == 0).all() and (img[baseline + 3:, 0:last_seg_point] == 0).all():
            # print("here", last_seg_hp[first_non_zero_index:])
            segment_points = segment_points[1:]
            # print("this is a dal at the end")

        for seg_point in segment_points:
            if seg_point != -1:
                # print("******************")
                cv2.line(image, (seg_point, 0), (seg_point, image.shape[0]), (255, 255, 255), 1)

        print("segment_points", segment_points)
        display_image("final", image)
        cv2.imwrite("final.png", image)
"""
