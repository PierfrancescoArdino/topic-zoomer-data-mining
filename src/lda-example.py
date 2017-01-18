from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from read_dataset import read_dataset
from filter_text import clean_text
import re

topics_no = 3
topic_words = 4
#Read the dataset
filename = "data/dataset.csv"
output = open('data/results.csv','w')
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
s = 25
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

count=0
for b in range(0,base):
	for h in range(0,height):
		corpus[h][b], dictionary[h][b] = clean_text(text[h][b])
		count=count +1

#print (height*base)
#print (count)

print ("Processing...")
# generating lda model
ldamodelMatrix=[ [[] for i in range(0,base)] for j in range(0,height)]
emptyMatrix=[ [[] for i in range(0,base)] for j in range(0,height)]
for b in range(base):
	for h in range(height):
		try:
			ldamodelMatrix[h][b] = gensim.models.ldamodel.LdaModel(corpus[h][b], num_topics=topics_no, id2word=dictionary[h][b], passes=20)
			emptyMatrix[h][b]=1
		except:
			emptyMatrix[h][b]=0
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




