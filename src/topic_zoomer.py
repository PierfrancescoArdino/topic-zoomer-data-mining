#!/usr/bin/python3
from nltk.tokenize import RegexpTokenizer
from gensim import models
import gensim
from read_dataset import read_dataset
from utils import create_corpus, create_dict
import re, pickle, sys, time, argparse
from collections import namedtuple


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
	if topLeft.x <= x and topLeft.y >= y and \
		bottomRight.x >= x and bottomRight.y <= y:
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
	if topLeft.x <= x and topLeft.y >= y and \
		bottomRight.x >= x and bottomRight.y <= y:
		return True
	else:
		return False


def compute(filename, s, recomputation):
	start_time = time.time()
	topics_no = 3
	topic_words = 3
	#Read the dataset
	reg = r"\""
	reg_compiled = re.compile(reg)

	dataset = read_dataset(filename)
	tokenizer = RegexpTokenizer(r"\"\w+\"")
	#Calculate area of the dataset
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
	print("Area: ")
	print(t_l, b_r)
	top_left = Point(x=int(round(t_l[0]-0.4)), y=int(round(t_l[1]+0.4)))
	bottom_right = Point(x=int(round(b_r[0]+0.4)), y=int(round(b_r[1]-0.4)))
	print("top_left=" + str(top_left))
	print("bottom_right=" + str(bottom_right))
	squares = get_squares(top_left, bottom_right, s)
	print(squares)
	try:
		old_size = pickle.load(open("../data/s.p",'rb'))
	except:
		old_size = 3.345678134567

	#Here we calculate the number of squares needed
	length = ((bottom_right.x-top_left.x)/s)+0.4
	height = ((top_left.y-bottom_right.y)/s)+0.4
	length = int(round(length))
	height = int(round(height))

	#The first corner is the top-left corner, the second corner is the bottom-right corner
	#corpus=[height][length]
	corpus = [[[] for i in range(0, length)] for j in range(0, height)]
	dictionary = [[[] for i in range(0, length)] for j in range(0, height)]
	text = [[[] for i in range(0, length)] for j in range(0, height)]
	first_corner = [0.0, 0.0]
	second_corner = [0.0, 0.0]
	print("Dimensions: ", height, length)
	print("Dataset pre-processing")
	if s % old_size != 0:
		count=0
		dict_dataset, clean_dataset = create_dict(dataset)
		for page in clean_dataset:
			y,x = is_inside(page, top_left, bottom_right, s, squares)
			text[y][x].append(page[2])
			count=count +1
			print (count,"/",len(clean_dataset))
		print("Processing...")
		# generating lda model
		ldamodelMatrix = [[[] for i in range(0, length)] for j in range(0, height)]
		count = 1
		print(text)
		for y in range(height):
			for x in range(length):
				print(count, "/", length*height)
				count += 1
				corpus[y][x] = create_corpus(text[y][x], dict_dataset)
				print(corpus[y][x])
				if corpus[y][x]:
					# Multicore variant
					# ldamodelMatrix[h][b] = gensim.models.ldamulticore.LdaMulticore(corpus[y][x], num_topics=topics_no, id2word=dict_dataset, passes=30)
					ldamodelMatrix[y][x] = gensim.models.ldamodel.LdaModel(corpus[y][x], num_topics=topics_no, id2word=dict_dataset, passes=30)
		if recomputation != 0:
			pickle.dump(height,open("../data/height.p", 'wb'))
			pickle.dump(length,open("../data/length.p", 'wb'))
			pickle.dump(s,open("../data/s.p", 'wb'))
			pickle.dump(corpus,open("../data/corpus.p", 'wb'))
			for y in range(height):
				for x in range(length):
					if ldamodelMatrix[y][x]:
						ldamodelMatrix[y][x].save("../data/lda("+str(y)+")("+str(x)+").model")
	elif s != old_size:
		length_old = pickle.load(open("../data/length.p", 'rb'))
		height_old = pickle.load(open("../data/height.p", 'rb'))
		corpus_old = pickle.load(open("../data/corpus.p", 'rb'))
		to_merge = int(s/old_size)
		length_old = length * to_merge
		height_old = height * to_merge
		print(length_old, height_old)
		print(len(corpus_old), len(corpus_old[0]))
		for i in range(0, height_old-len(corpus_old)):
			corpus_old.append([[] for i in range(0, length_old)])
		diff = length_old-len(corpus_old[0])
		for element in corpus_old:
			for i in range(0, diff):
				element.append([])
		print(len(corpus_old), len(corpus_old[0]))
		ldamodelMatrix_old = [[[] for i in range(0, length_old)] for j in range(0, height_old)]
		for y in range(height_old):
			for x in range(length_old):
				try:
					ldamodelMatrix_old[y][x] = models.LdaModel.load("../data/lda("+str(y)+")("+str(x)+").model")
				except:
					ldamodelMatrix_old[y][x] = []
		# START REC STUPID USING LDA UPDATE
		y_old = 0
		x_old = 0
		y = 0
		x = 0
		print(corpus_old)
		ldamodelMatrix = [[[] for i in range(0, length)] for j in range(0, height)]
		while y_old < height_old:
			while x_old < length_old:
				if (y_old + to_merge > height_old) or (x_old + to_merge > length_old):
					break
				corpus_to_merge = []
				for i in range(0,to_merge):
					for j in range(0,to_merge):
						if not ldamodelMatrix[y][x] and corpus_old[y_old + i][x_old + j]:
							print(corpus_old[y_old + i][x_old + j])
							ldamodelMatrix[y][x] = ldamodelMatrix_old[y_old + i][x_old + j]
						else:
							print(corpus_old[y_old+i][x_old+j])
							corpus_to_merge += corpus_old[y_old+i][x_old+j]
				#print(corpus_to_merge)
				if ldamodelMatrix[y][x]:
					ldamodelMatrix[y][x].update(corpus_to_merge)
				x_old += to_merge
				x += 1
			y_old += to_merge
			y += 1

		print(y_old, x_old, y, x)
		print(ldamodelMatrix)
	# Print results
	if s != old_size:
		output = open('../data/results.csv', 'w')
		output.write("Top-Left corner; Bottom-Right corner; Topic with id 1; Topic with id 2; Topic with id 3 \n")
		h = top_left.y
		for y in range(height):
			for x in range(length):
				if ldamodelMatrix[y][x]:
					print(top_left)
					res_top_left = [float(s * x)+top_left.x, h - float(s*y)]
					res_bottom_right = [float(s*(x+1))+top_left.x, float(h - s*(y+1))]
					topic = []
					for i in range(0, topics_no):
						string = ldamodelMatrix[y][x].print_topic(i, topn=topic_words)
						tokens = tokenizer.tokenize(string)
						#tps = print()
						topic.append("".join(tokens)+";")
					topic_string= "".join(topic)
					topic_string = reg_compiled.sub(' ', topic_string)
					output.write(str(res_top_left[0])+","+str(res_top_left[1])+" ; "+str(res_bottom_right[0])+","+str(res_bottom_right[1])+" ; "+topic_string + "\n")
		output.close()
	end_time = time.time()
	print(end_time-start_time)
if __name__ == '__main__':
	# command line arguments
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#parser.add_argument('k', type=int, help='the number of topics to be found')
	parser.add_argument('step', type=float, help='the square side size')
	parser.add_argument('dataset', type=str, help='the path to the input dataset')
	parser.add_argument('recomputation', type=int, help='flag to activate recomputation, 0 no recomputation, '
														'1 LdaUpdate recomputation, 2 merging recomputation')
	args = parser.parse_args()
	compute(args.dataset, args.step, args.recomputation)



