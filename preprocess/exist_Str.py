
import os, csv


def sentlist(fname):
	f = open(fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
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
			if sents[0]!='':
				s.append(sents[0])
			sentls.append(s)
			# clear temp s
			s = []
			s.append(sents[1])
		else:
			s.append(sents)
	f.close()
	return sentls


def indexlist(fname):
	f = open(fname, 'r', encoding = 'utf-8-sig', errors='ignore')
	idxls = []
	for idxs in f.readlines():
		idxs = idxs.replace(u'\xa0', u'')
		# replace all \n
		idxs = idxs.replace('\n','')
		# skip space
		if (idxs == ' ') or (idxs == ''):
			continue
		idxls.append(idxs)
	f.close()
	return idxls 


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


if __name__ == "__main__":

	ipath = "../textbook_data/html_txt/result3_6/"
	spath = "existStr/"
	rpath = "existRel/"
	npath = "relStr/"
	# for indexRel_1 into indexRel_1rel
	# spath = "indexRel_1/"
	# rpath = "indexRel_1rel/"

	idxslist = indexlist(ipath+"index_terms.txt")
	relslist = relationlist(ipath+"relation_terms.txt")


	# # from seg into existStr
	# sentslist = sentlist(ipath+"segment/seg.txt")

	# existls = ['具有','含有']

	# for s in sentslist:
	# 	for r in relslist:
	# 		if r in s:
	# 			if ('具有' in s):
	# 				with open(ipath+npath+"具有.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
	# 					wrtf.write(" ".join(s))
	# 					wrtf.write("\n")
	# 			elif ('含有' in s):
	# 				with open(ipath+npath+"含有.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
	# 					wrtf.write(" ".join(s))
	# 					wrtf.write("\n")
	# 			else:
	# 				with open(ipath+npath+r+".txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
	# 					wrtf.write(" ".join(s))
	# 					wrtf.write("\n")


	# print("existStr Finish!")


	# from seg into existStr
	# sentslist = sentlist(ipath+"segment/seg.txt")

	# for s in sentslist:
	# 	for i in idxslist:
	# 		if i in s:
	# 			with open(ipath+spath+i+".txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
	# 				wrtf.write(" ".join(s))
	# 				wrtf.write("\n")

	# print("existStr Finish!")


	# from existStr into existRel
	# or indexRel_1 to indexRel_1rel
	files = os.listdir(ipath+spath)

	for file in files:

		filename = os.path.splitext(file)[0]
		sentslist = sentlist(ipath+spath+file)

		for s in sentslist:
			for r in relslist:
				if r in s:
					with open(ipath+rpath+r+".txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
					# with open(ipath+rpath+r+".txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
						for item in s:
							item = item.strip()
							wrtf.write("%s "%item)
						wrtf.write("\n")
						break

		print(file)
