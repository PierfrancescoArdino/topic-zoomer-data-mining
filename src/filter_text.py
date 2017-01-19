from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
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


def clean_text(doc_set):
	
	texts = []

	# loop through document list
	for i in doc_set:
	# clean and tokenize document string
		raw = i.lower()
		raw = reg_compiled.sub('', raw)
		raw = reg2_compiled.sub('', raw)
		tokens = tokenizer.tokenize(raw)
		
	# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in stop_words]
	# stem tokens
		#stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	# add tokens to list
		texts.append(stopped_tokens)
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
	    
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]
	return corpus, dictionary