import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from gensim.models import word2vec
from gensim import models
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
			for k, v in v_dict.items():
				if p[0] == k:
					p_head = v.tolist()
					break
			for k, v in v_dict.items():
				if p[-1] == k:
					p_tail = v.tolist()
					break
			pair_vec.append(np.array(p_tail)-np.array(p_head))
		except ValueError:
			print(p,"ValueError!")

	return pair_vec


if __name__ == "__main__":

	path = "textbook_data/html_txt/result1_2/"
	ipath = "indexRel_1rel/"
	ppath = "b_pairs/"
	# clsname = 'AdaBoost'

	# load word2vec model
	# w2v_model = word2vec.Word2Vec.load(path+"models_seg/seg.model")

	# get vocab vector into dictionary
	# vocab_dict = {k: w2v_model.wv[k] for k, v in w2v_model.wv.vocab.items()}

	
	# read pair file into 2-dim list
	# pair_word = sentlist(path+ppath+"pair_words.txt")

	# transform word pair into word vector
	# pair_vec = vec_preprocesser(vocab_dict, pair_word)
	# pickle.dump(pair_vec, open(path+ppath+"pair_words.pickle", 'wb'))

	print("start KMeans")
	k_clusters = 30
	alldata = pickle.load(open(path+ppath+"pair_words.pickle", 'rb'))
	kmeans_clr = KMeans(n_clusters=k_clusters, random_state=0).fit(np.array(alldata))
	pickle.dump(kmeans_clr, open(path+ppath+"kmeans_clu.sav", 'wb'))
	print("finish KMeans")

	kmeans_clr = pickle.load(open(path+ppath+"kmeans_clu.sav", 'rb'))
	print('n_clusters:%s\n%s' % (k_clusters,Counter(kmeans_clr.labels_)))

	print("start wrt clu")
	f = open(path+ppath+"pair_words.txt", mode='r', encoding = 'utf-8-sig', errors='ignore')
	fo = open(path+ppath+"clu_result/pair_words_result.txt",'a', encoding = 'utf-8-sig',newline='')
	for idx, sents in enumerate(f):
		sents = sents.replace(u'\n',u'')
		try:
			# with open(path+ppath+"clu_result/pair_words_result.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
			fo.write('%s %s\n' % (sents,str(kmeans_clr.labels_[idx])))
		except IndexError:
			print(sents,"IndexError!")

		with open(path+ppath+"clu_result/_result"+str(kmeans_clr.labels_[idx])+".txt",'a', encoding = 'utf-8-sig',newline='') as wrt:
			wrt.write('%s\n' %sents)
	print("finish wrt clu")