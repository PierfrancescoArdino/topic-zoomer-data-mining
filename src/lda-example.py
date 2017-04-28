from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from read_dataset import read_dataset
from filter_text import clean_text
import re
import pickle

topics_no = 3
topic_words = 3
#Read the dataset
filename = "../data/test.csv"
output = open('../data/results.csv','w')
reg = r"\""
reg_compiled = re.compile(reg)

dataset = read_dataset(filename)
tokenizer = RegexpTokenizer(r"\"\w+\"")
#Calculate area of the dataset
maximum = []
minimum = []
maximum.append(0.0)
maximum.append(0.0)
minimum.append(9999999999999.0)
minimum.append(9999999999999.0)

for i in range(len(dataset)):
	if dataset[i][0] < minimum[0]:
		minimum[0] = dataset[i][0]
	if dataset[i][1] < minimum[1]:
		minimum[1] = dataset[i][1]
	if dataset[i][0] > maximum[0]:
		maximum[0] = dataset[i][0]
	if dataset[i][1] > maximum[1]:
		maximum[1] = dataset[i][1]
print("Area: ")
print(maximum)
print(minimum)

#Here we divide the area in SxS squares
s = 10
try:
	old_size = pickle.load(open("../data/old/s.p",'rb'))
except:
	old_size = 3.345678134567
if (s%old_size != 0):
	#Here we calculate the number of squares needed
	base = ((maximum[0]-minimum[0])/s)+0.5
	height = ((maximum[1]-minimum[1])/s)+0.5
	base = int(round(base))
	height = int(round(height))

	#The first corner is the bottom-left corner, the second corner is the top-right corner
	#corpus=[height][base]
	corpus = [ [[] for i in range(0,base)] for j in range(0,height)]
	dictionary = [ [[] for i in range(0,base)] for j in range(0,height)]
	text = [ [[] for i in range(0,base)] for j in range(0,height)]
	first_corner = [0.0, 0.0]
	second_corner = [0.0, 0.0]
	print("Dimensions: ", height, base)
	print("Dataset pre-processing")
	count=0
	for element in dataset:
		#find x position
		b=int(round(((element[0]-minimum[0])/s)+0.5))
		#find y position
		h=int(round(((element[1]-minimum[1])/s)+0.5))
		if (h==0):
			h=1
		if(b==0):
			b=1
		#print(h-1, b-1)
		text[h-1][b-1].append(element[2])
		count=count +1
		print (count,"/",base*height)

	print ("Processing...")
	# generating lda model
	ldamodelMatrix=[ [[] for i in range(0,base)] for j in range(0,height)]
	emptyMatrix=[ [[] for i in range(0,base)] for j in range(0,height)]
	count = 1
	for b in range(base):
		for h in range(height):
			print (count,"/",base*height)
			count=count +1
			try:
				corpus[h][b], dictionary[h][b] = clean_text(text[h][b])
				#Multicore variant
				#ldamodelMatrix[h][b] = gensim.models.ldamulticore.LdaMulticore(corpus[h][b], num_topics=topics_no, id2word=dictionary[h][b], passes=20)
				ldamodelMatrix[h][b] = gensim.models.ldamodel.LdaModel(corpus[h][b], num_topics=topics_no, id2word=dictionary[h][b], passes=30)
				emptyMatrix[h][b]=1
			except:
				emptyMatrix[h][b]=0

	pickle.dump(height,open("../data/old/height.p",'wb'))
	pickle.dump(base,open("../data/old/base.p",'wb'))
	pickle.dump(s,open("../data/old/s.p",'wb'))
	pickle.dump(corpus,open("../data/old/corpus.p",'wb'))
	pickle.dump(emptyMatrix,open("../data/old/empty.p",'wb'))
	for b in range(base):
		for h in range(height):
			if(emptyMatrix[h][b]==1):
				ldamodelMatrix[h][b].save("../data/old/lda("+str(h)+")("+str(b)+").model")
else:
	base = pickle.load(open("../data/old/base.p",'rb'))
	height = pickle.load(open("../data/old/height.p",'rb'))
	corpus = pickle.load(open("../data/old/corpus.p",'rb'))
	emptyMatrix = pickle.load(open("../data/old/empty.p",'rb'))
	ldamodelMatrix=[ [[] for i in range(0,base)] for j in range(0,height)]
	for b in range(base):
		for h in range(height):
			if(emptyMatrix[h][b]==1):
				ldamodelMatrix[h][b]= models.LdaModel.load("../data/old/lda("+str(h)+")("+str(b)+").model")
	


# Print results
output.write("First corner; Second corner; Topic with id 1; Topic with id 2; Topic with id 3 \n")
for b in range(base):
	for h in range(height):
		if(emptyMatrix[h][b]==1):
			first_corner=[float(s*(b))+minimum[0], float(s*(h))+minimum[1]]
			second_corner=[float(s*(b+1))+minimum[0], float(s*(h+1))+minimum[1]]
			topic = []
			for i in range(0,topics_no):
				string=ldamodelMatrix[h][b].print_topic(i, topn=topic_words)
				tokens = tokenizer.tokenize(string)
				#tps = print()
				topic.append("".join(tokens)+";")
			topic_string= "".join(topic)
			topic_string = reg_compiled.sub(' ', topic_string)
			output.write(str(first_corner[0])+","+str(first_corner[1])+" ; "+str(second_corner[0])+","+str(second_corner[1])+" ; "+topic_string + "\n")
output.close()




