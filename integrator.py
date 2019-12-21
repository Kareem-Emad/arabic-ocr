import json


def get_words_from_text(img_name, input_path):
    with open(f'{input_path}_text/{img_name.replace("png","txt")}') as f:
        lines = f.readline()
        words = []
        while(lines.find('  ') != -1):
            lines = lines.replace('  ', ' ')
        lines = lines.replace('\n', '')
        lines = lines.replace('لا', 'L').split(' ')
        while '' in lines:
            lines.remove('')
        words = lines
        return words


def count_feat_vecs(feat_vecs):
    c = 0
    for fv in feat_vecs:
        if(fv != []):
            c += 1
    return c


composities_map = {
    'لد': 'x'
}
composities = ['لد']


def augment_with_compsities(word_text):
    for comp in composities:
        while(word_text.count(comp) != 0):
            word_text = word_text.replace(comp, composities_map[comp])
    return word_text


def replace_composities(word_text):
    for comp in composities:
        while(word_text.count(composities_map[comp]) != 0):
            word_text = word_text.replace(composities_map[comp], comp)
    return word_text


def should_have_one_dot(fv):
    return fv[5] == 1


def should_have_dots(fv):
    return fv[5] > 0


def should_have_no_dots(fv):
    return fv[5] == 0


def should_have_score(fv):
    return fv[0] != 0


def should_have_no_score(fv):
    return fv[0] == 0


def should_be_dotted_bottom(fv):
    return fv[4] == 3


def should_be_dotted_top(fv):
    return fv[4] == 1


def should_be_dotted_middle(fv):
    return fv[4] == 2


def should_have_high_score(fv):
    return fv[0] > 128


def should_have_4_hoz_trans(fv):
    return fv[6] >= 4


def should_have_6_hoz_trans(fv):
    return fv[6] >= 6


def should_have_4_ver_trans(fv):
    return fv[7] >= 4


def should_have_6_ver_trans(fv):
    return fv[7] >= 6


validation_map = {
    'ا': [should_have_no_dots],
    'ب': [should_have_one_dot, should_be_dotted_bottom],
    'ت': [should_have_dots, should_be_dotted_top],
    'ث': [should_have_dots, should_be_dotted_top],
    'ج': [should_have_one_dot, should_have_4_ver_trans],
    'ح': [should_have_no_dots, should_have_4_ver_trans],
    'خ': [should_have_one_dot, should_be_dotted_top, should_have_4_ver_trans],
    'د': [should_have_no_dots],
    'ذ': [should_have_one_dot, should_be_dotted_top],
    'ر': [should_have_no_dots],
    'ز': [should_have_one_dot, should_be_dotted_top],
    'س': [should_have_no_dots, should_have_score, should_have_high_score, should_have_6_hoz_trans],
    'ش': [should_have_score, should_have_dots, should_be_dotted_top, should_have_high_score, should_have_6_hoz_trans], # noqa
    'ص': [should_have_no_dots, should_have_score, should_have_4_ver_trans],
    'ض': [should_have_one_dot, should_have_score, should_be_dotted_top, should_have_4_ver_trans],
    'ط': [should_have_no_dots, should_have_score, should_have_4_ver_trans],
    'ظ': [should_have_one_dot, should_have_score, should_be_dotted_top, should_have_4_ver_trans],
    'ع': [should_have_no_dots, should_have_4_ver_trans],
    'غ': [should_have_one_dot, should_be_dotted_top, should_have_4_ver_trans],
    'ف': [should_have_one_dot, should_be_dotted_top, should_have_4_ver_trans, should_have_4_hoz_trans],
    'ق': [should_have_dots, should_be_dotted_top, should_have_4_ver_trans, should_have_4_hoz_trans],
    'ك': [should_have_score],
    'ل': [should_have_no_dots],
    'م': [should_have_no_dots, should_have_4_ver_trans, should_have_4_hoz_trans],
    'ن': [should_have_one_dot, should_be_dotted_top],
    'ه': [should_have_no_dots, should_have_score],
    'و': [should_have_no_dots],
    'ى': [],
    'ي': [should_have_dots, should_be_dotted_bottom],
    'L': [should_have_no_dots],
    'x': [should_have_no_dots, should_have_score]
}


def compare_and_assign(feat_vects, word_str, char_map):
    word_str = augment_with_compsities(word_str)
    if(len(word_str) != count_feat_vecs(feat_vects)):
        return -1
    feat_vects.reverse()
    for i in range(0, len(word_str)):
        curr_char = word_str[i]
        char_validations = validation_map[curr_char]
        not_valid = False
        for validation in char_validations:
            if(not validation(feat_vects[i])):
                not_valid = True
                break

        if(not_valid is True):
            continue

        score = feat_vects[i][0]
        if(score not in char_map):
            char_map[score] = []

        fc_tup = (curr_char, feat_vects[i])
        if(fc_tup not in char_map[score]):
            char_map[score].append(fc_tup)
    return char_map


def load_features_map():
    feat_map = {}
    try:
        with open('config_map.json') as f:
            feat_map = json.load(f)
        return feat_map
    except Exception:
        return {}


def match_feat_to_char(feat_map, feat_vecs):
    feat_vecs.reverse()
    word_str = ''
    for fv in feat_vecs:
        score = str(fv[0])
        if(score in feat_map):
            for tup in feat_map[score]:
                if(fv in tup):
                    word_str += tup[0]
                    break
    word_str = replace_composities(word_str)
    return word_str
