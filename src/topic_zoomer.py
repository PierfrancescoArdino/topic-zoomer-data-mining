#!/usr/bin/python3
from nltk.tokenize import RegexpTokenizer
from gensim import models
import gensim
from read_dataset import read_dataset
from utils import create_corpus, create_dict, topic_update
import re
import pickle
import sys
import time
import argparse
from collections import namedtuple
import psutil


def get_squares(topLeft, bottomRight, step):
    # every little square is defined as topLeft and
    # bottomRight angles (as namedtuples)
    Point = namedtuple('Point', ['x', 'y'])
    out = []
    y = 0
    x = 0
    yMin = topLeft.y
    yMax = topLeft.y - step
    while yMax >= bottomRight.y or ((bottomRight.y - yMax) < step):
        xMin = topLeft.x
        xMax = topLeft.x + step
        x = 0
        while xMax <= bottomRight.x or ((xMax - bottomRight.x) < step):
            square = [Point(xMin, yMin), Point(xMax, yMax), (y, x)]
            out.append(square)
            # update x boundaries
            xMin = xMax
            xMax += step
            x += 1
        # update y boundaries
        yMin = yMax
        yMax -= step
        y += 1
    return out


"""
Assign the point to a specific square in the grid
"""


def is_inside(row, topLeft, bottomRight, step, squares):
    x = float(row[0])
    y = float(row[1])
    if topLeft.x <= x and topLeft.y >= y \
            and bottomRight.x >= x and bottomRight.y <= y:
        # now I'm inside the selected area
        # range among all the possible little squares
        for s in squares:
            if point_inside_square(x, y, s):
                return s[2]


"""
Determine if a point is inside a given square or not
"""


def point_inside_square(x, y, square):
    topLeft = square[0]
    bottomRight = square[1]
    if topLeft.x <= x and topLeft.y >= y \
            and bottomRight.x >= x and bottomRight.y <= y:
        return True
    else:
        return False


def compute(filename, s, recomputation):
    start_time = time.time()
    topics_no = 3
    topic_words = 3
    cores = (psutil.cpu_count(logical=False)) - 1
    reg = r"\""
    reg_compiled = re.compile(reg)
    dataset = read_dataset(filename)
    tokenizer = RegexpTokenizer(r"\"\w+\"")
    # Calculate area of the dataset
    Point = namedtuple('Point', ['x', 'y'])
    t_l = [sys.maxsize, -sys.maxsize]
    b_r = [-sys.maxsize, sys.maxsize]
    for i in range(len(dataset)):
        if dataset[i][0] < t_l[0]:
            t_l[0] = dataset[i][0]
        if dataset[i][1] > t_l[1]:
            t_l[1] = dataset[i][1]
        if dataset[i][0] > b_r[0]:
            b_r[0] = dataset[i][0]
        if dataset[i][1] < b_r[1]:
            b_r[1] = dataset[i][1]
    print("Area: {} {}".format(t_l, b_r))
    top_left = Point(x=int(round(t_l[0] - 0.4)),
                     y=int(round(t_l[1] + 0.4)))
    bottom_right = Point(x=int(round(b_r[0] + 0.4)),
                         y=int(round(b_r[1] - 0.4)))
    print("top_left=" + str(top_left))
    print("bottom_right=" + str(bottom_right))
    squares = get_squares(top_left, bottom_right, s)
    print(squares)
    try:
        old_size = pickle.load(open("../data/s.p", 'rb'))
    except:
        old_size = 3.345678134567

    # Here we calculate the number of squares needed
    length = ((bottom_right.x - top_left.x) / s) + 0.4
    height = ((top_left.y - bottom_right.y) / s) + 0.4
    length = int(round(length))
    height = int(round(height))

    # The first corner is the top-left corner,
    # the second corner is the bottom-right corner
    # corpus=[height][length]
    corpus = [[[] for i in range(0, length)] for j in range(0, height)]
    text = [[[] for i in range(0, length)] for j in range(0, height)]
    first_corner = [0.0, 0.0]
    second_corner = [0.0, 0.0]
    print("Dimensions: {} {}".format(height, length))
    print("Dataset pre-processing")
    if s % old_size != 0:
        count = 0
        dict_dataset, clean_dataset = create_dict(dataset)
        print(dict_dataset)
        for page in clean_dataset:
            y, x = is_inside(page, top_left, bottom_right, s, squares)
            text[y][x].append(page[2])
            count += 1
            print(count, "/", len(clean_dataset))
        print("Processing...")
        # generating lda model
        ldamodelmatrix = [[[] for i in range(0, length)]
                          for j in range(0, height)]
        count = 1
        for y in range(height):
            for x in range(length):
                print(count, "/", length * height)
                count += 1
                corpus[y][x] = create_corpus(text[y][x], dict_dataset)
                if corpus[y][x]:
                    if cores > 1:
                        ldamodelmatrix[y][x] = gensim.models.ldamulticore.LdaMulticore(corpus[y][x],
                                                                                       num_topics=topics_no,
                                                                                       id2word=dict_dataset,
                                                                                       workers=cores, passes=50)
                        ldamodelmatrix[y][x] = gensim.models.ldamodel.LdaModel(corpus[y][x], num_topics=topics_no,
                                                                               id2word=dict_dataset, passes=50)
        if recomputation != 0:
            pickle.dump(s, open("../data/s.p", 'wb'))
            pickle.dump(corpus, open("../data/corpus.p", 'wb'))
            pickle.dump(dict_dataset, open("../data/dict_dataset.p", 'wb'))
            for y in range(height):
                for x in range(length):
                    if ldamodelmatrix[y][x]:
                        ldamodelmatrix[y][x].save("../data/lda(" + str(y) + ")(" + str(x) + ").model")
    elif s != old_size:
        corpus_old = pickle.load(open("../data/corpus.p", 'rb'))
        dict_dataset = pickle.load(open("../data/dict_dataset.p", 'rb'))
        to_merge = int(s / old_size)
        length_old = length * to_merge
        height_old = height * to_merge
        print("New dimensions are: {} {} ".format(height_old, length_old))
        for i in range(0, height_old - len(corpus_old)):
            corpus_old.append([[] for i in range(0, length_old)])
        diff = length_old - len(corpus_old[0])
        for element in corpus_old:
            for i in range(0, diff):
                element.append([])
        ldamodelmatrix_old = [[[] for i in range(0, length_old)] for j in range(0, height_old)]
        for y in range(height_old):
            for x in range(length_old):
                try:
                    ldamodelmatrix_old[y][x] = models.LdaModel.load("../data/lda(" + str(y) + ")(" + str(x) + ").model")
                except:
                    ldamodelmatrix_old[y][x] = []
        if recomputation == 1:
            y_old = 0
            y = 0
            count=1
            ldamodelmatrix = [[[] for i in range(0, length)] for j in range(0, height)]
            print("Recomputation....")
            while y_old < height_old:
                x_old = 0
                x = 0
                while x_old < length_old:
                    print(count, "/", length * height)
                    count += 1
                    if (y_old + to_merge > height_old) or (x_old + to_merge > length_old):
                        break
                    corpus_to_merge = []
                    for i in range(0, to_merge):
                        for j in range(0, to_merge):
                            if not ldamodelmatrix[y][x] and corpus_old[y_old + i][x_old + j]:
                                ldamodelmatrix[y][x] = ldamodelmatrix_old[y_old + i][x_old + j]
                            else:
                                corpus_to_merge += corpus_old[y_old + i][x_old + j]
                    if ldamodelmatrix[y][x]:
                        ldamodelmatrix[y][x].update(corpus_to_merge)
                        print(ldamodelmatrix[y][x].show_topics(num_words=topic_words, log=False, formatted=False))
                    x_old += to_merge
                    x += 1
                y_old += to_merge
                y += 1
        elif recomputation == 2:
            y_old = 0
            y = 0
            count=1
            corpus_new = [[[] for i in range(0, length)] for j in range(0, height)]
            ldamodelmatrix = [[[] for i in range(0, length)] for j in range(0, height)]
            print("Recomputation....")
            while y_old < height_old:
                x_old = 0
                x = 0
                while x_old < length_old:
                    print(count, "/", length * height)
                    count += 1
                    if (y_old + to_merge > height_old) or (x_old + to_merge > length_old):
                        break
                    topics_new_cell = []
                    for i in range(0, to_merge):
                        for j in range(0, to_merge):
                            corpus_new[y][x] += corpus_old[y_old + i][x_old + j]
                            if ldamodelmatrix_old[y_old + i][x_old + j]:
                                tmp_topic_cell = ldamodelmatrix_old[y_old + i][x_old + j].show_topics(
                                    num_words=topic_words, log=False, formatted=False)
                                for t in tmp_topic_cell:
                                    for w in t[1]:
                                        topics_new_cell.append(w[0])
                    topics_new_cell = set(topics_new_cell)
                    ldamodelmatrix[y][x] = topic_update(corpus_new[y][x], topics_new_cell, dict_dataset)
                    x_old += to_merge
                    x += 1
                y_old += to_merge
                y += 1
    # Print results
    if s != old_size:
        output = open('../data/results.csv', 'w')
        output.write("Top-Left corner; Bottom-Right corner; Topic with id 1; Topic with id 2; Topic with id 3 \n")
        h = top_left.y
        for y in range(height):
            for x in range(length):
                if ldamodelmatrix[y][x]:
                    res_top_left = [float(s * x) + top_left.x, h - float(s * y)]
                    res_bottom_right = [float(s * (x + 1)) + top_left.x, float(h - s * (y + 1))]
                    topic = []
                    if recomputation != 2:
                        for i in range(0, topics_no):
                            string = ldamodelmatrix[y][x].print_topic(i, topn=topic_words)
                            tokens = tokenizer.tokenize(string)
                            topic.append("".join(tokens) + ";")
                        topic_string = "".join(topic)
                        topic_string = reg_compiled.sub(' ', topic_string)
                        output.write(str(res_top_left[0]) + "," + str(res_top_left[1]) + " ; " + str(
                            res_bottom_right[0]) + "," + str(res_bottom_right[1]) + " ; " + topic_string + "\n")
                    else:
                        for i in range(0, topics_no):
                            if i < len(ldamodelmatrix[y][x]):
                                string = ldamodelmatrix[y][x][i]
                                topic.append("".join(string) + ";")
                        topic_string = "".join(topic)
                        topic_string = reg_compiled.sub(' ', topic_string)
                        output.write(str(res_top_left[0]) + "," + str(res_top_left[1]) + " ; " + str(
                            res_bottom_right[0]) + "," + str(res_bottom_right[1]) + " ; " + topic_string + "\n")

        output.close()
    end_time = time.time()
    print("Computation time is: {}".format(end_time - start_time))

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('step', type=float, help='the square side size')
    parser.add_argument('dataset', type=str, help='the path to the input dataset')
    parser.add_argument('recType', type=int, help='flag to activate recomputation, 0 no recomputation, '
                                                        '1 LdaUpdate recomputation, 2 merging recomputation')
    args = parser.parse_args()
    compute(args.dataset, args.step, args.recType)
