from textblob_ar.correction import TextCorrection # noqa


def show_candidate_corrections(word):
    return TextCorrection().correction(word)


def correct_word(word):
    return TextCorrection().correction(word, top=True)
