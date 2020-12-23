# conding=utf-8
import nltk
from nltk.corpus import brown
from nltk import WordNetLemmatizer
from math import log


def wordlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	wordls = []
	for words in f.read().split(' '):
		# replace \ufeff
		words = words.replace(u'\ufeff',u'')
		if words.find('\n')!=-1:
			words = words.split('\n')
			wordls.extend(words)
		else:
			wordls.append(words)
	f.close()
	return wordls


def sentlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	# each sentance
	s = []
	# total sentances
	sentls = []
	for sents in f.read().split(' '):
		# replace \ufeff
		sents = sents.replace(u'\ufeff',u'')
		if sents.find('\n')!=-1:
			# split with \n
			sents = sents.split('\n')
			# add before sentance to sentance list
			s.append(sents[0])
			sentls.append(s)
			# clear temp s
			s = []
			s.append(sents[1])
		else:
			s.append(sents)
	f.close()
	return sentls


def termlist(in_fname):
	f = open(in_fname, 'r', encoding = 'utf-8-sig', errors='ignore')
	termls = []
	for terms in f.readlines():
		terms = terms.replace(u'\xa0', u'')
		# replace all \n
		terms = terms.replace('\n','')
		# skip space
		if (terms == ' ') or (terms == ''):
			continue
		termls.append(terms)
	f.close()
	return termls 


def relationlist(in_fname):
	f = open(in_fname, 'r', encoding = 'utf-8-sig', errors='ignore')
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



# 單字x出現的機率
def p(x):
	return _Fdist[x]/float(len(_Fdist))

# 計算單字x和單字y出現在同一個句子的機率
def pxy(x,y):
	errls = []
	errls = [0.00044943820224719103]
	ipxy = (len(list(filter(lambda s :  (x in s) & (y in s) ,_Sents)))+1)/ float(len(_Sents) )
	if ipxy in errls:
		return 0
	else:
		return ipxy

# 計算單字x和單字y的Pointwise Mutual Information
def pmi(x,y):
	if p(x)==0:
		print(x+"=0")
		return 0
	elif p(y)==0:
		print(y+"=0")
		return 0
	else:
		return log(pxy(x,y)/(p(x)*p(y)),2)


if __name__ == '__main__':

	print("/////////////start/////////////")

	ipath = "doc_clr/textbook_data/html_txt/result1_2/"

	wordslist = wordlist(ipath+"indexRel_merge.txt")
	sentslist = sentlist(ipath+"indexRel_merge.txt")
	termslist = termlist(ipath+"index_merge.txt")
	tfidflist = termlist(ipath+"term_tfidf.txt")
	relslist = relationlist(ipath+"given_relation_terms.txt")

	wnl = WordNetLemmatizer()

	# 單字出現頻率
	_Fdist = nltk.FreqDist([wnl.lemmatize(w.lower()) for w in wordslist])

	# 文章中的所有句子
	_Sents = [[wnl.lemmatize(j.lower()) for j in i] for i in sentslist]

	new_termlist=[]
	for t in termslist:
		if p(t)!=0:
			new_termlist.append(t)

	new_tfidflist=[]
	for t in tfidflist:
		if p(t)!=0:
			new_tfidflist.append(t)

	new_rellist=[]
	for r in relslist:
		if p(r)!=0:
			new_rellist.append(r)

	for r in new_rellist:
		rel_diclist = []
		pmi_dict = {}
		for t in new_termlist:
			if t in pmi_dict:
				continue
			pmi_dict[t] = pxy(r, t)
		for t in new_tfidflist:
			if t in pmi_dict:
				continue
			pmi_dict[t] = pxy(r, t)
		
		sorted_by_value = sorted([items for items in pmi_dict.items() if items[1]!=0], key=lambda kv: kv[1], reverse=True)
		
		rel_diclist.append(sorted_by_value)
		print(r+":")
		print(sorted_by_value[:20])



	print("/////////////end///////////////")


	while True:

		query = input()
		q_list = query.split()
		print(q_list)

		if q_list[0]=='p':
			print(p(q_list[1]))
		elif q_list[0]=='pxy':
			print(pxy(q_list[1], q_list[2]))
		elif q_list[0]=='pmi':
			print(pmi(q_list[1], q_list[2]))
		else:
			break


