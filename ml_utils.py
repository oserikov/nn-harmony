import sys


def read_training_data_from_file(filename):
    with open(filename) as f:
        words = []
        word = []
        for line in f.readlines():
            if line.strip() == "":
                words.append(word)
                word = []
            else:
                word.append([[int(c) for c in elem] for elem in line.rstrip().split(' ')])
        words.append(word)

    return words


def read_from_stdin():
    words = []
    word = []
    for line in sys.stdin:
        if line.strip() == "":
            words.append(word)
            word = []
        else:
            word.append([[int(c) for c in elem] for elem in line.rstrip().split(' ')])

    return list(filter(None, words))