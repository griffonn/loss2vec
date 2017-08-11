
from collections import Counter
from itertools import dropwhile
CORP_PATH = r"C:\Users\owner\Desktop\CNM\loss2vec\data\text1"
FREQ_MIN_THRESH = 1000


def get_words_with_min_freq(courpus, min_freq):
	# In case there is more than one line
	txt_list = []
	for line in range(0,len(words)):
		txt_list = txt_list + words[line].split()

	# Turn to counter
	txt_counter = Counter(txt_list)

	# Remove from counter all words with freq less than:
	for key, count in dropwhile(lambda key_count: key_count[1] >= min_freq, txt_counter.most_common()):
		del txt_counter[key]

	return list(txt_counter.keys())


if __name__ == '__main__':

	with open(CORP_PATH) as corp:
		words = corp.readlines()

	# Get top words
	top_words = get_words_with_min_freq(words, FREQ_MIN_THRESH)

	# Check for syn:
	


	# Check for ant:

