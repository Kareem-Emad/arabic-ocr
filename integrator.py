import json


def get_words_from_text(img_name, input_path):
    with open(f'{input_path}_text/{img_name.replace("png","txt")}') as f:
        lines = f.readline()
        words = []
        for line in lines:
            while(line.find('  ') != -1):
                line = line.replace('  ', '')
            line = line.replace('\n', '').split(' ')
            while '' in line:
                line.remove('')
            words += line
        return words


def compare_and_assign(feat_vects, word_str, char_map):
    if(len(word_str) != len(feat_vects)):
        return -1
    for i in range(0, len(word_str)):
        curr_char = word_str[i]
        if(not char_map.get(curr_char)):
            char_map[word_str[i]] = []

        if(feat_vects[i] not in char_map[word_str[i]]):
            char_map[word_str[i]].append(feat_vects[i])
    return char_map


def load_features_map():
    feat_map = {}
    with open('config_map.json') as f:
        feat_map = json.load(f)
    return feat_map


def match_feat_to_char(feat_map, feat_vecs):
    return 'سنلانسلنلت'
