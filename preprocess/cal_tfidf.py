import numpy as np
import string, csv, os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from os.path import join 

def sentlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	# each sentance
	s = []
	# total sentances
	sentls = []
	for sents in f.readlines():
		# replace \ufeff
		sents = sents.replace(u'\ufeff',u'')
		sents = sents.replace(u'\n',u'')
		sents = sents.strip()
		sentls.append(sents)
	f.close()
	return sentls


def wrtfidf(ipath,fname):

	# write each term of tfidf to csv
	if os.path.exists(ipath):
		with open (ipath,'a', encoding = 'utf-8-sig',newline='') as csvfile:
			wrfile = csv.writer(csvfile)
			findtfidf(wrfile)
			csvfile.close()
	else:
		# write new row with index
		with open (ipath,'w', encoding = 'utf-8-sig',newline='') as csvfile:
			wrfile = csv.writer(csvfile)
			wrfile.writerow([fname,'1','2','3','4','5','6','7','8','9','10'])
			findtfidf(wrfile)
			csvfile.close()


def findtfidf(wrt):

	n = 20 # print top 10 tfidf
	listindex = 0;
	
	for s in sentslist:
		# witch document (listindex)
		listindex = listindex + 1

		# sort 
		loc = np.argsort(-weight[listindex-1])

		wrlist=[]
		wrlist.append(str(listindex))

		# print range n tfidf 
		for i in range(n):
			try:
				# exception
				if weight[listindex-1][loc[i]]==0 : break
				if weight[listindex-1][loc[i]]==1.0: continue
				wrstr = words[loc[i]]+','+str(weight[listindex-1][loc[i]])
				wrlist.append(wrstr)
					# print (u'-{}: {} {}'.format(str(i + 1), words[loc[i]], weight[listindex-1][loc[i]]))
				# print ('\n')
			except IndexError:
				continue

		if len(wrlist)!=1:
			wrt.writerow(wrlist)


if __name__ == "__main__":


	ipath = "../textbook_data/html_txt/result1_2/"
	iipath = ipath+"existRel/"
	opath = ipath+"csv/"

	files = os.listdir(iipath)

	for file in files:

		filename = os.path.splitext(file)[0]

		sentslist = sentlist(iipath+file)

		# calculate tf-idf
		vectorizer = CountVectorizer()
		transformer = TfidfTransformer()
		tfidf = transformer.fit_transform(vectorizer.fit_transform(sentslist))

		words = vectorizer.get_feature_names()
		weight = tfidf.toarray()

		# write tf-idf
		wrtfidf(opath+filename+".csv",filename)

		print(filename+' finish !')


	
