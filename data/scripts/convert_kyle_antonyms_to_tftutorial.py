import itertools

with open(r'../antonyms.txt', 'r') as f:
	ant_pairs = f.readlines()

with open(r'../test-antonyms.txt', 'w') as output:
	for p1, p2 in itertools.product(ant_pairs, repeat=2):
		if (p1 != p2):
			output.write(p1.replace(',', ' ').replace('\n', '') + ' ' + p2.replace(',', ' ').replace('\n', '') + '\n')
