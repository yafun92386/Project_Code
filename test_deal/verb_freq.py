from collections import Counter
import itertools

def wordlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	wordls = []
	for words in f.read().split(' '):
		# replace \ufeff
		words = words.replace(u'\ufeff',u'')
		words = words.strip()
		if words.find('\n')!=-1:
			words = words.split('\n')
			wordls.extend(words)
		else:
			wordls.append(words)
	f.close()
	return wordls

def relationlist(fname):
    f = open(fname, 'r', encoding = 'utf-8-sig', errors='ignore')
    rells = []
    for rels in f.readlines():
    	rels = rels.replace(u'\xa0', u'')
    	# replace all \n
    	rels = rels.replace('\n','')
    	# skip space
    	if (rels == ' ') or (rels == ''):
    		continue
    	rells.append(rels)
    f.close()
    return rells 


if __name__ == '__main__':

	path = "doc_clr/textbook_data/html_txt/result1_2/"
	word_name = "segment/seg.txt"
	rel_name = "segment/seg_verb.txt"

	wordslist = wordlist(path+word_name)
	relslist = relationlist(path+rel_name)

	c = Counter(wordslist)
	# print(c.most_common(10))

	verb_dict = {}
	for r in relslist:
		if r in c:
			verb_dict.update({r:c[r]})

	print(dict(itertools.islice(verb_dict.items(), 100)))