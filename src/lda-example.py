from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from read_dataset import read_dataset
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
filename = "data/test.csv"
dataset = read_dataset(filename)
doc_set = []
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
print(maximum)
print(minimum)

first_corner = [10.0, 10.0]
second_corner = [0.0, 0.0]
count = 0
count2 = 0
stop = 0
for i in range(len(dataset)):
	if(first_corner[0] < dataset[i][0] < second_corner[0]) or (first_corner[0]>dataset[i][0]>second_corner[0]):
		if(first_corner[1]<dataset[i][1]<second_corner[1]) or (first_corner[1]>dataset[i][1]>second_corner[1]):
			if stop == 0:
				doc_set.append(dataset[i][2])
				print(count)
				count += 1
print ("Processing...")
# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
# clean and tokenize document string
	raw = i.lower()
	tokens = tokenizer.tokenize(raw)
# remove stop words from tokens
	stopped_tokens = [i for i in tokens if not i in en_stop]
# stem tokens
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
# add tokens to list
	texts.append(stemmed_tokens)
print(texts)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)

print (ldamodel.print_topics(num_topics=2, num_words=10))