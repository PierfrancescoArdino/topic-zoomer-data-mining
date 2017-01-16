import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

url = "http://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers"
headers={'User-Agent':user_agent,} 

def urlToText(url):
	request=urllib.request.Request(url,None,headers)
	#url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
	html = urllib.request.urlopen(request).read()
	soup = BeautifulSoup(html)

	# kill all script and style elements
	for script in soup(["script", "style"]):
	    script.extract()    # rip it out

	# get text
	text = soup.get_text()

	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)

	return text