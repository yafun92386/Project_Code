import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec
from gensim import models
from numpy import array
from os.path import join
import pickle, random, os
import numpy as np

# get training data file (return label) then merge into "training_data.txt"
def get_data(path,name):

	label = []
	files = os.listdir(path)

	if os.path.exists(path+'training_data.txt'):
  		os.remove(path+'training_data.txt')
	fo = open(path+'training_data.txt', 'a', encoding = 'utf-8', errors='ignore')
	# for label_1 
	for file in files:
		if os.path.splitext(file)[0][0] == '1':
			f = open(join(path,file), 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					label.append(1)
			f.close()
	# for label_0
	for file in files:
		if os.path.splitext(file)[0][0] == '0':
			f = open(join(path,file), 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					label.append(0)
			f.close()
	fo.close()

	# save label_data with pickle
	pickle.dump(label,open(path+"label.txt","wb"))

	return label

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

# mlp_classifier
def mlp_clf(ipath,name,train_X,train_y):

	# set classifier parameter
	clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, max_iter=200,
						hidden_layer_sizes=(50,10), random_state=1)

	# fit our data to classifier
	clf.fit(train_X, train_y)

	# save model with pickle
	pickle.dump(clf, open(ipath+"models_1rel/"+name, 'wb'))


	return clf



if __name__ == "__main__":

	# path = "textbook_data/html_txt/result1_2/"
	path = "Case/doc_clr/1_2/dealString/"
	# ipath = "relStr/"
	lbname = "label.txt"
	clsname = 'ArB_cls.sav'

	# get label with training data
	label = get_data(path,lbname)

	# get label with pickle
	label = pickle.load(open(path+lbname, 'rb'))

	# training_data into 2-dim list
	sents_list = sentlist(path+ipath+"training_data.txt")

	# load word2vec model
	w2v_model = word2vec.Word2Vec.load(path+"models_seg/seg.model")

	# get vocab vector into dictionary
	key_dict = {k: w2v_model.wv[k] for k, v in w2v_model.wv.vocab.items()}

	# set training_data(sents_list) into vector 2-dim list(sents_vector)
	sents_vector = []
	for idx, sent in enumerate(sents_list):
		sents_vector.append([])
		for word in sent:
			sents_vector[idx].append(key_dict[word])

	# vector preprocesser X(inputvec:word pair), y(label)
	inputvec = []
	for idx in range(len(sents_vector)):
		try:
			tmp_vec = []
			tmp_vec = sents_vector[idx][0].tolist()
			tmp_vec.extend(sents_vector[idx][2])
			inputvec.append(tmp_vec)
		except IndexError:
			print("! %d IndexError !"%idx)
			pass


	# train_test_split
	X, y = array(inputvec), array(label)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, 
											random_state=None, shuffle=True, stratify=y)

	# get testing word, print word pair
	test_word = []
	for vec in X_test.tolist():
		# each pair word
		tmp = []
		for word, value in key_dict.items():
			if vec[:250] in value:
				tmp.append(word)
			if vec[-250:] in value:
				tmp.append(word)
		test_word.append(tmp)
	print(test_word)

	# calculate train_vec - relation_vec
	relation_vec = ((w2v_model.wv['具有']).tolist())*2

	X_train_dif = []
	for xtrain in X_train:
		X_train_dif.append(xtrain-relation_vec)

	X_test_dif = []
	for xtest in X_test:
		X_test_dif.append(xtest-relation_vec)


	# throw it to classifier then predict
	# clf = mlp_clf(path+ipath, clsname, X_train_dif, y_train)

	# load cls_model from pickle
	cls_model = pickle.load(open(path+ipath+"models_1rel/"+clsname, 'rb'))

	# print score
	print("Training score: %f" % cls_model.score(X_train_dif, y_train))
	print("Test score: %f" % cls_model.score(X_test_dif, y_test))

	# print predict and correct ans
	print("predict : ",cls_model.predict(X_test_dif))
	print("label   : ",y_test)

