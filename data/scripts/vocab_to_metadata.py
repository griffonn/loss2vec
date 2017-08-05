import sys

if len(sys.argv) < 3:
	print('Usage: vocab_to_metdata.py <vocab.txt> <metadata.tsv>')
	exit()

with open(sys.argv[1], 'r') as f:
	words = f.readlines()

with open(sys.argv[2], 'w') as output:
	output.write('Word\tFrequency\n')
	for word in words:
		output.write(word[2:].replace('\'', '').replace(' ', '\t').replace('\n', '') + '\n')

