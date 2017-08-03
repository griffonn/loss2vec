from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.tokenize import RegexpTokenizer

# Instantiate the reader like this
PATH_TO_BNC_TEXTS = '/tmp/2554/2554/download/Texts/'
NEW_BNX_TXT = "/tmp/bnc2txt_new.txt"
BNC2TEXT = '/tmp/bnc2txt.txt'

def BNC2TXT():
	bnc_reader = BNCCorpusReader(root=PATH_TO_BNC_TEXTS, fileids=r'[A-K]/\w*/\w*\.xml')
	tokenizer = RegexpTokenizer(r'\w+')
	# txt = bnc_reader.sents() #all the bnc corpus by sentances

	with open(NEW_BNX_TXT, 'w') as nf:
		i = 0 
		for s in bnc_reader.sents():
			nf.write(' '.join(tokenizer.tokenize(s)))
			i = i + 1
			if i%100000==0:
				print('Joined {} Sentances , {}% Done'.format(i, i/6026276))
		pass

if __name__ == '__main__':
	print('Start Script')
	BNC2TXT()
	print('End Script')



