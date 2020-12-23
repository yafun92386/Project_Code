import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from gensim import models
from os import listdir, remove, makedirs
from os.path import isfile, isdir, join, splitext, getsize, exists
import pickle, random, os
import numpy as np

# split training_data into 2-dim list to find w2v
def sentlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	# total sentances (sents into list)
	sentls = []
	# for each sents in f_file
	for sents in f.readlines():
		# ignore space & \ufeff & \n 
		sents = sents.strip()
		sents = sents.replace(u'\ufeff',u'')
		sents = sents.replace(u'\n',u'')
		# split sents with space then append to sentls
		tmp_s = sents.split(" ")
		sentls.append(tmp_s)

	f.close()
	return sentls

# transform pair_word into pair_vector
def vec_preprocesser(v_dict,p_pair):

	pair_vec = []
	for p in p_pair:
		p_head = []
		p_tail = []
		try:
			p_head = v_dict[p[0]].tolist()
			p_tail = v_dict[p[1]].tolist()
			pair_vec.append(np.array(p_tail)-np.array(p_head))
		except ValueError:
			print(p,"ValueError!")

	return pair_vec


if __name__ == "__main__":

	path = "textbook_data/html_txt/result1_2/"
	ipath = "indexRel_1rel/"
	ppath = "b_pairs_index/"

	clsnames = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Gaussian_Process",
		 "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
		 "Naive_Bayes", "QDA"]

	# load word2vec model
	w2v_model = word2vec.Word2Vec.load(path+"models_seg/seg.model")

	# get vocab vector into dictionary
	key_dict = {k: w2v_model.wv[k] for k, v in w2v_model.wv.vocab.items()}


	for clsname in clsnames:

		if not exists(path+ppath+clsname):
			makedirs(path+ppath+clsname)
			print("Create %s"%clsname)

		# load cls_model from pickle
		cls_model = pickle.load(open(path+ipath+"models_1rel/"+clsname+".sav", 'rb'))

		files = os.listdir(path+ppath)
		for file in files:

			inpath = join(path+ppath, file)
			filename = splitext(file)[0]

			# if os.path.isfile(path+ppath+file):
			if isfile(inpath) & (splitext(file)[-1] == '.txt') & (filename!="pair_index"):

				# read pair file into 2-dim list
				# pair_word = sentlist(inpath)

				# transform word pair into word vector
				# pair_vec = vec_preprocesser(key_dict, pair_word)
				# pickle.dump(pair_vec, open(path+ppath+filename+".pickle", 'wb'))
				datavec = pickle.load(open(path+ppath+filename+".pickle", 'rb'))

				# throw pair_vec into cls_model to predict
				pred_result = cls_model.predict(list(datavec))

				# write result into each file
				f = open(path+ppath+file, mode='r', encoding = 'utf-8-sig', errors='ignore')
				for idx, sents in enumerate(f):
					sents = sents.replace(u'\n',u'')
					try:
						with open(path+ppath+clsname+"/"+filename+"_result.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
							wrtf.write('%s %s\n' % (sents,str(pred_result[idx].tolist())))
					except IndexError:
						print(sents,"IndexError!")

					# write result 1 into one file
					if(int(pred_result[idx])==1):
						with open(path+ppath+clsname+"/_result1.txt",'a', encoding = 'utf-8-sig',newline='') as wrt1:
							wrt1.write('%s\n' %sents)

				print(file)
