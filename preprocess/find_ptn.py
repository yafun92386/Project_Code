
import os,csv,re,jieba, math
import jieba.posseg as pseg
from os import walk


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


def readTerm(fname):

	with open(fname, newline='',encoding='utf-8-sig') as csvfile:
		# put term into term Matrix
		wordMatrix = []
		# read row index
		rowindex = 0
		# read csv with dictionary
		dic_csv = csv.DictReader(csvfile,['term','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])

		# read row_csv to list
		for row in dic_csv:
			# 1d termVector append = 2d wordMatrix
			wordMatrix.append([])
			# read row's value into row_list
			for key in dic_csv.fieldnames:
				# skip none value
				if row[key] is None:
					break
				# write term into wordMatrix[rowindex]
				else:
					wordMatrix[rowindex].append(row[key].split(',')[0])
		
			rowindex += 1

		return wordMatrix


def wrtTermMatrix(term_mtx,wrtfname):
	# Matrix append into list
	wrtMatrixls = []
	for r in range(0,len(term_mtx)):
		for c in range(0,len(term_mtx[r])):
			if (term_mtx[r][c] in wrtMatrixls) | (term_mtx[r][c] in idxslist):
				continue
			else:
				wrtMatrixls.append(term_mtx[r][c])
	# write into file
	with open(wrtfname, 'w', encoding = 'utf-8-sig', errors='ignore') as tMf:
		for w in wrtMatrixls:
			tMf.write(w)
			tMf.write('\n')


if __name__ == "__main__":

	jieba.load_userdict('../../jieba_dict/dictionary_merge.txt')

	ipath = "../textbook_data/html_txt/result1_2/"

	# get index list
	idxslist = indexlist(ipath+"index_terms.txt")

	# get relation word list
	rellist = relationlist(ipath+"relation_terms.txt")

	# read term_csv into termMatrix
	termMatrix = readTerm(ipath+'sum_tfidf.csv')

	# write termMatrix into list to tfidf_terms file
	wrtTermMatrix(termMatrix,ipath+"tfidf_terms.txt")
	
	
	for idx in range(0,len(termMatrix)):

		term = termMatrix[idx][0]

		# read existRel into strlist
		strlist = sentlist(ipath+"existRel/"+term+".txt")

		# write relation string into list
		wrtRelation = []
		
		# for col_term (top sum_tfidf)
		for col_index in range(0,len(termMatrix[idx])):

			# skip same term
			if termMatrix[idx][col_index] == term :
				continue

			if termMatrix[idx][col_index] != '':
				# for every strlist
				for every_str in strlist:

					# find all word exist in str's then put into word_index
					word_index = []
					word_index = [w for w, word in enumerate(every_str) if word==termMatrix[idx][col_index]]

					if len(word_index)==0:
						continue

					# find all term exist in str's then put into term_index
					term_index = []
					term_index = [t for t, word in enumerate(every_str) if word==term]

					# find close term
					for ti in term_index:
						for wi in word_index:
							wrtList = []
							# if wi within ti+-2
							# wi ... ti
							if ((ti>wi) & ((ti-2)<=wi)):
								# get substr
								wrtList = every_str[wi:ti+1]
							# ti ... wi
							elif ((ti<wi) & ((ti+2)>=wi)):
								# get substr
								wrtList = every_str[ti:wi+1]

							# skip only 2 word
							if len(wrtList)<=2:
								continue
							# skip no relation word
							rel_flag = 0
							for r in rellist:
								if wrtList[math.floor(len(wrtList)/2)]!=r:
									continue
								else:
									rel_flag=1
									break

							# if has relation word
							if rel_flag==1:
								# if substr doesn't repeat
								if wrtList not in wrtRelation:
									wrtRelation.append(wrtList)


			with open (ipath+"indexRel_1/"+term+".txt",'w', encoding = 'utf-8-sig',newline='') as wrtf:
				for wrt in wrtRelation:
					for w in wrt:
						wrtf.write("%s "%w)
					# wrtf.write(wrt)
					wrtf.write("\n")

			if os.path.getsize(ipath+"indexRel_1/"+term+".txt")==0:
				os.remove(ipath+"indexRel_1/"+term+".txt")

		print(term+" finish")
	