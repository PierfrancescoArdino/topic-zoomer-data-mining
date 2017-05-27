from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import operator
import re

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
reg = r"\'[a-zA-Z]+"
reg2 = r"\[a-zA-Z]"
reg_compiled = re.compile(reg)
reg2_compiled = re.compile(reg2)
# create English stop words list
en_stop = get_stop_words('en')
it_stop = get_stop_words('it')
stop_words = en_stop + it_stop
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


def create_dict(dataset):
	texts = []
	texts_for_dict = []
	# loop through document list
	for i in dataset:
		# clean and tokenize document string
		raw = i[2].lower()
		raw = reg_compiled.sub('', raw)
		raw = reg2_compiled.sub('', raw)
		tokens = tokenizer.tokenize(raw)

		# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in stop_words]
		# stem tokens
		# stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		# add tokens to list
		texts_for_dict.append(stopped_tokens)
		texts.append([i[0],i[1],stopped_tokens])
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts_for_dict)
	return dictionary, texts


def create_corpus(doc_set, dictionary):
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in doc_set]
	return corpus


def topic_update(corpus, topics, dict_corpus):
	topic_occ = {}
	for topic in topics:
		for page in corpus:
			for token in page:
				if dict_corpus[token[0]] == topic:
					if topic not in topic_occ:
						topic_occ[topic]=token[1]
					else:
						topic_occ[topic] += token[1]
	total_token = sum(v for k,v in topic_occ.items())
	for key, value in topic_occ.items():
		topic_occ[key] = value/total_token
	topic_occ_sorted = sorted(topic_occ.items(), key=operator.itemgetter(1), reverse=True)
	return [topic_occ_sorted[i][0] for i in range(0, min(3, len(topic_occ_sorted)))]