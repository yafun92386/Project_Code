
import os, logging, pickle, numpy as np
from gensim.models import word2vec
from keras.layers.embeddings import Embedding

def sentlist(fname):
	f = open(fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	lines = [line.strip().split(' ') for line in f]
	return lines

def termlist(fname):
	f = open(fname, 'r', encoding = 'utf-8-sig', errors='ignore')
	lines = [line.strip() for line in f]
	return lines


def mergexist(ipath,existls):

	txtlist = []
	files = os.listdir(ipath)
	fo = open(ipath+'1_data.txt', 'a', encoding = 'utf-8', errors='ignore')
	
	# read file into list with rule
	for file in files: 
		filename = os.path.splitext(file)[0]
		# 1_data rule
		if filename in existls:
			f = open(ipath+file, 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				txtls = txt.split(' ')
				if txtls not in txtlist:
					txtlist.append(txtls)

	# write txt into fo file
	for txt in txtlist:
		fo.write(" ".join(txt))

	fo.close()

def get_label(ipath):

	# label = []
	files = os.listdir(ipath)

	if os.path.exists(ipath+'training_data.txt'):
		os.remove(ipath+'training_data.txt')
	if os.path.exists(ipath+'y_label.txt'):
		os.remove(ipath+'y_label.txt')
	fo = open(ipath+'training_data.txt', 'a', encoding = 'utf-8', errors='ignore')
	fl = open(ipath+'y_label.txt', 'a', encoding = 'utf-8', errors='ignore')
	# for label_1 
	for file in files:
		if os.path.splitext(file)[0][0] == '1':
			f = open(ipath+file, 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					fl.write("1\n")
			f.close()
	# for label_0
	for file in files:
		if os.path.splitext(file)[0][0] == '0':
			f = open(ipath+file, 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					fl.write("0\n")
			f.close()
	fo.close()
	fl.close()

def get_vec(ipath):

	# word into index table
	word2idx = {"_PAD": 0}

	# load word2vec model
	w2v_model = word2vec.Word2Vec.load(ipath+"w2vModel/seg_long_w2v.model")

	# get vocab vector into dictionary
	# vocab_list = {k: w2v_model.wv[k] for k, v in w2v_model.wv.vocab.items()}
	vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]

	# map embedding matrix
	embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
	for i, vocab in enumerate(vocab_list):
		word, vec = vocab
		embedding_matrix[i + 1] = vec
		word2idx[word] = i + 1

	embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)

	# save with pickle
	pickle.dump(word2idx, open(ipath+"dealString/word2idx.pkl", 'wb'))
	pickle.dump(embedding_matrix, open(ipath+"dealString/embmatrix.pkl", 'wb'))
	pickle.dump(embedding_layer, open(ipath+"dealString/emblayer.pkl", 'wb'))

	# training_data into 2-dim list
	sents_list = sentlist(ipath+"dealString/training_data.txt")

	# # set training_data(sents_list) into vector 2-dim list(sents_vector)
	# sents_vector = []
	# for idx, sent in enumerate(sents_list):
	# 	sents_vector.append([])
	# 	for word in sent:
	# 		try:
	# 			sents_vector[idx].append(vocab_list[word])
	# 		except KeyError:
	# 			print(word+" not in vocab")
	# 			continue
	# # save inputvec with pickle
	# pickle.dump(sents_vector, open(ipath+"dealString/X_data.txt", 'wb'))

	# set training_data(sents_list) into vector 2-dim list(sents_vector)
	sents_idx = []
	for idx, sent in enumerate(sents_list):
		sents_idx.append([])
		for word in sent:
			try:
				sents_idx[idx].append(word2idx[word])
			except KeyError:
				print(word+" not in vocab")
				sents_idx[idx].append(0)
	# save inputvec with pickle
	pickle.dump(sents_idx, open(ipath+"dealString/X_data.txt", 'wb'))

	


# if __name__ == "__main__":
def madeTrainset(ipath,existls):

	# set logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
	log = logging.getLogger('deal_trainset')
	# tmp for deal counting
	tmp = 0


	# idxslist = termlist(ipath+"dictionary/index_terms.txt")
	relslist = termlist(ipath+"dictionary/relation_terms.txt")

	# from seg into existStr
	sentslist = sentlist(ipath+"segment/seg_long.txt")
	sen_len = len(sentslist)

	pairls = sentlist(ipath+"dealString/pair_list.txt")

	# for no, s in enumerate(sentslist):
	# 	pair_tag = False
	# 	if (any(e in s for e in existls)==True & all(e!=s[0] for e in existls)==True):
	# 		with open(ipath+"dealString/1has_data.txt",'a', encoding = 'utf-8',newline='') as wrtf:
	# 			wrtf.write(" ".join(s))
	# 			wrtf.write("\n")
	# 		continue

	# 	for pair in pairls:
	# 		pair_tag =  all(p in s  for p in pair)
	# 		if (pair_tag==True):
	# 			with open(ipath+"dealString/1_data.txt",'a', encoding = 'utf-8',newline='') as wrtf:
	# 				wrtf.write(" ".join(s))
	# 				wrtf.write("\n")
	# 			break
	# 	# if not match pair, write into 0_data
	# 	if (pair_tag==False):
	# 		if (any(r in s  for r in relslist)==True):
	# 			with open(ipath+"dealString/0_data.txt",'a', encoding = 'utf-8',newline='') as wrtf:
	# 				wrtf.write(" ".join(s))
	# 				wrtf.write("\n")

	# 	# for ele in existls:
	# 	# 	if(ele in s):
	# 	# 		exs_tag = True
	# 	# 		with open(ipath+"dealString/"+ele+".txt",'a', encoding = 'utf-8',newline='') as wrtf:
	# 	# 			wrtf.write(" ".join(s))
	# 	# 			wrtf.write("\n")
	# 	# 		break
	# 	# if exs_tag==False:
	# 	# 	# for r in relslist:
	# 	# 		# if r in s:
	# 	# 	with open(ipath+"dealString/0_data.txt",'a', encoding = 'utf-8',newline='') as wrtf:
	# 	# 		wrtf.write(" ".join(s))
	# 	# 		wrtf.write("\n")

	# 	# set logging for processing
	# 	prec = int(no/sen_len*100)
	# 	if (prec%10==0) & (tmp != prec):
	# 		tmp = prec
	# 		log.info('Deal '+str(prec)+'% lines')

	# log.info('0_data finish')

	# # merge 1_data
	# mergexist(ipath+"dealString/",existls)
	# log.info('1_data finish')

	# get label with training data
	get_label(ipath+"dealString/")
	log.info('label finish')

	# get training_data embedding vector
	get_vec(ipath)
	log.info('data_vec finish')

