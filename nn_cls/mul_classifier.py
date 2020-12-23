import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from gensim.models import word2vec
from gensim import models
from numpy import array
from os.path import join
import pickle, random, os
import numpy as np

# get training data file (return label) then merge into "training_data.txt"
def get_data(ipath,name):

	label = []
	files = os.listdir(ipath)

	if os.path.exists(ipath+'training_data.txt'):
		os.remove(ipath+'training_data.txt')
	fo = open(ipath+'training_data.txt', 'a', encoding = 'utf-8', errors='ignore')
	# for label_1 
	for file in files:
		if os.path.splitext(file)[0][0] == '1':
			f = open(join(ipath,file), 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					label.append(1)
			f.close()
	# for label_0
	for file in files:
		if os.path.splitext(file)[0][0] == '0':
			f = open(join(ipath,file), 'r', encoding = 'utf-8', errors='ignore')
			for txt in f:
				if txt!="":
					fo.write(txt)
					label.append(0)
			f.close()
	fo.close()

	# save label_data with pickle
	# pickle.dump(label,open(ipath+"models_1rel/"+name,"wb"))
	pickle.dump(label,open(ipath+name,"wb"))

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


if __name__ == "__main__":

	path = "textbook_data/html_txt/result1_2/"
	ipath = "indexRel_1rel/"
	iipath = "relStr/"
	dataname = "X_data.txt"
	lbname = "y_label.txt"

	names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Gaussian_Process",
		 "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
		 "Naive_Bayes", "QDA"]

	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		GaussianProcessClassifier(1.0 * RBF(1.0)),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		MLPClassifier(alpha=1),
		AdaBoostClassifier(),
		GaussianNB(),
		QuadraticDiscriminantAnalysis()]

	# get label with training data
	# label = get_data(path+iipath,lbname)

	# get label with pickle
	# label = pickle.load(open(path+iipath+"models_1rel/"+lbname, 'rb'))
	label = pickle.load(open(path+iipath+lbname, 'rb'))

	# training_data into 2-dim list
	sents_list = sentlist(path+iipath+"training_data.txt")

	# load word2vec model
	w2v_model = word2vec.Word2Vec.load(path+"models_seg/seg.model")

	# get vocab vector into dictionary
	key_dict = {k: w2v_model.wv[k] for k, v in w2v_model.wv.vocab.items()}

	# set training_data(sents_list) into vector 2-dim list(sents_vector)
	sents_vector = []
	for idx, sent in enumerate(sents_list):
		sents_vector.append([])
		for word in sent:
			try:
				sents_vector[idx].append(key_dict[word])
			except KeyError:
				# print(word)
				# with open(path+iipath+"nonVec.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
				# 	wrtf.write(word)
				# 	wrtf.write("\n")
				continue

	print(sents_vector[0][1])

	# # vector preprocesser X(inputvec:word pair), y(label)
	# inputvec = []
	# for idx in range(len(sents_vector)):
	# 	try:
	# 		tmp_vec = []
	# 		tmp_vec = sents_vector[idx][0].tolist()
	# 		tmp_vec.extend(sents_vector[idx][2])
	# 		inputvec.append(tmp_vec)
	# 	except IndexError:
	# 		print("! %d IndexError !"%idx)
	# 		pass

	# save inputvec with pickle
	# pickle.dump(inputvec, open(path+ipath+"models_1rel/"+dataname, 'wb'))
	# pickle.dump(sents_vector, open(path+iipath+dataname, 'wb'))
	# load inputvec by using pickle
	# datavec = pickle.load(open(path+ipath+"models_1rel/"+dataname, 'rb'))
	datavec = pickle.load(open(path+iipath+dataname, 'rb'))


	# # train_test_split
	# X, y = array(datavec), array(label)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
	# 										random_state=None, shuffle=True, stratify=y)

	# # get testing word, print word pair
	# test_word = []
	# for vec in X_test.tolist():
	# 	# each pair word
	# 	tmp = []
	# 	for word, value in key_dict.items():
	# 		if vec[:250] in value:
	# 			tmp.append(word)
	# 		if vec[-250:] in value:
	# 			tmp.append(word)
	# 	test_word.append(tmp)
	# print(test_word)

	# # calculate train_vec - relation_vec
	# # relation_vec = ((w2v_model.wv['具有']).tolist())*2

	# X_train_dif = []
	# for xtrain in X_train:
	# 	X_train_dif.append(xtrain[-250:]-xtrain[:250])

	# X_test_dif = []
	# for xtest in X_test:
	# 	X_test_dif.append(xtest[-250:]-xtest[:250])


	# # # throw it to classifier then predict
	# # for name, clf in zip(names, classifiers):
	# # 	clf.fit(X_train_dif, y_train)
	# # 	pickle.dump(clf, open(path+ipath+"models_1rel/"+name+".sav", 'wb'))


	# for clsname in names:

	# 	# load cls_model from pickle
	# 	cls_model = pickle.load(open(path+ipath+"models_1rel/"+clsname+".sav", 'rb'))

	# 	print("%s : "%clsname)

	# 	try:
	# 		# print score
	# 		print("Training score: %f" % cls_model.score(X_train_dif, y_train))
	# 		print("Test score: %f" % cls_model.score(X_test_dif, y_test))

	# 		# print predict and correct ans
	# 		print("predict : ",cls_model.predict(X_test_dif))
	# 		print("label   : ",y_test)

	# 	except ValueError:
	# 		print("%s ValueError !"%clsname)

