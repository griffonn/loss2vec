def is_ascii(s):
    return all(ord(c) < 128 for c in s)

import re
regex = re.compile(r"\s\s+", re.IGNORECASE)

with open('/tmp/wiki.en.text', 'r') as f:
    with open('/tmp/wiki_clean.en.text', 'w') as g:
        for i, line in enumerate(f.readlines()):
            if i % 10000 == 0: print(i)
            g.write(regex.sub(' ', ' '.join(map(lambda x: x if is_ascii(x) else '', line.split(' ')))))