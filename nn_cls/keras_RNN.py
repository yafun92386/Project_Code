
# %%
import warnings
warnings.filterwarnings(action='ignore')
import os, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from gensim import models
from gensim.models import word2vec
from keras.preprocessing import sequence
# from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Input, Bidirectional, Concatenate, Dense, Dropout, Flatten
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from sklearn.model_selection import train_test_split

# %%
# split training_data into 2-dim list to find w2v
def sentlist(in_fname):
	f = open(in_fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	lines = [line.strip().split(' ') for line in f]
	return lines

# %%
path = "Case/doc_clr/merge_/"
dataname = "X_data.txt"
lbname = "y_label.txt"
clsname = "kerasbiLSTMa"

# training_data into 2-dim list
sents_list = sentlist(path+"dealString/"+dataname)
# load inputvec by using pickle
datavec = pickle.load(open(path+"dealString/"+dataname, 'rb'))
# load trained emb. layer
embedding_layer = pickle.load(open(path+"dealString/emblayer.pkl", 'rb'))
# get label with pickle
label = np.loadtxt(path+"dealString/"+lbname, delimiter='\n')

# token = Tokenizer(num_words=6000)
# token.fit_on_texts(sents_list)
# data_seq = token.texts_to_sequences(sents_list)
# pad_data = sequence.pad_sequences(data_seq, maxlen=20)

# %%
# train_test_split
X, y = np.array(datavec), np.array(label)
X = sequence.pad_sequences(X, maxlen=20, padding='post')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
													random_state=None, shuffle=True, stratify=y)

#%%
from keras import backend as K
from keras.engine.topology import Layer
# from .base_layer import Layer
from keras import initializers, regularizers, constraints
class Attention(Layer):
	# def __init__(self, step_dim, **kwargs):
	def __init__(self, step_dim,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):
		# self.supports_masking = True
		# # preset supports_masking = False

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)
		self.bias = bias

		self.step_dim = step_dim
		self.features_dim = 0

		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		# assert len(input_shape) == 3
		self.features_dim = input_shape[-1]
		self.W = self.add_weight(name='{}_W'.format(self.name),
								 shape = (input_shape[-1],),
								 initializer="glorot_normal",
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint, 
								 trainable=True)
		if self.bias:
			self.b = self.add_weight(name='{}_b'.format(self.name),
									shape = (input_shape[1],),
									initializer='zero',
									regularizer=self.b_regularizer,
									constraint=self.b_constraint, 
									trainable=True)
		else:
			self.b = None					
		super(Attention, self).build(input_shape)

	# def compute_mask(self, input, input_mask=None):
	# 	return None

	def call(self, x, mask=None):
		features_dim = self.features_dim
		step_dim = self.step_dim
		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
						K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b
		eij = K.tanh(eij)

		a = K.exp(eij)
		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)
		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.features_dim

#%%

from keras.layers import Layer
from keras import backend as K

class selfAttention(Layer):
	def __init__(self, n_head, hidden_dim, penalty=0.1, **kwargs):
		self.n_head = n_head
		self.P = penalty
		
		self.hidden_dim = hidden_dim
		super(selfAttention, self).__init__(**kwargs)
	
	def build(self, input_shape):
		self.W1 = self.add_weight(name='w1', shape=(input_shape[2], self.hidden_dim), initializer='uniform',
								  trainable=True)
		self.W2 = self.add_weight(name='W2', shape=(self.hidden_dim, self.n_head), initializer='uniform',
								  trainable=True)
		super(selfAttention, self).build(input_shape)
	
	def call(self, x, **kwargs):
		d1 = K.dot(x, self.W1)
		tanh1 = K.tanh(d1)
		d2 = K.dot(tanh1, self.W2)
		softmax1 = K.softmax(d2, axis=0)
		A = K.permute_dimensions(softmax1, (0, 2, 1))
		emb_mat = K.batch_dot(A, x, axes=[2, 1])
		reshape = K.batch_flatten(emb_mat)
		eye = K.eye(self.n_head)
		prod = K.batch_dot(softmax1, A, axes=[1, 2])
		self.add_loss(self.P * K.sqrt(K.sum(K.square(prod - eye))))
		return reshape
	
	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[-1] * self.n_head


#%%

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(embedding_layer)

# # throw it to classifier then predict
# model = Sequential()
# # model.add(keras.layers.Embedding(output_dim=32,
# # 					input_dim=6000,
# # 					input_length=20))
# model.add(embedding_layer)
# model.add(Bidirectional(LSTM(units=32,return_sequences=True,dropout=0.35)))
# model.add(Attention(20))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dropout(0.35))
# model.add(Dense(units=1, activation='sigmoid'))

inp = Input(shape=(20, ))
x = embedding_layer(inp)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.35))(x)
x = Attention(20)(x)
# x = selfAttention(1, 20)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.35)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

model.summary()

model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


# %%
from keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = path+"dealString/"+clsname+".model.hdf5"
ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
						   save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience=3)
train_history = model.fit(X_train, y_train, batch_size=100, epochs=30, validation_split=0.4, callbacks=[ckpt,early])

# Save model
# model_json = model.to_json()
# with open(path+"dealString/"+clsname+".json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights(path+"dealString/"+clsname+".model.h5")
# model.save(path+"dealString/"+clsname)
# pickle.dump(model, open(path+"dealString/"+clsname, 'wb'))

#%%

def cal_att_weights(output, att_w):
	eij = np.tanh(np.dot(output[0], att_w[0]))
	# eij = np.dot(eij, att_w[2])
	eij = eij.reshape((eij.shape[0], eij.shape[1]))
	ai = np.exp(eij)
	weights = ai / np.sum(ai)

	return weights

def get_attention(sent_model, sequences, topN=5):
	sent_before_att = K.function([sent_model.layers[0].input, K.learning_phase()],
								[sent_model.layers[2].output])
	cnt_reviews = sequences.shape[0]
	
	sent_att_w = sent_model.layers[3].get_weights()
	sent_all_att = []
	for i in range(cnt_reviews):
		sent_each_att = sent_before_att([[sequences[i]], 0])
		sent_each_att = cal_att_weights(sent_each_att, sent_att_w)
		sent_each_att = sent_each_att.ravel()
		sent_all_att.append(sent_each_att)
	sent_all_att = np.array(sent_all_att)

	return sent_all_att


# %%
def show_train_history(train_history, train, validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# %%
# model.load_weights(file_path)
# with open(path+"dealString/"+clsname+".json",'r') as f:
#     json = f.read()
# cls_model = model_from_json(json, custom_objects={'Attention(20)':Attention(20)})
# cls_model.load_weights(path+"dealString/"+clsname+".model.h5")
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# with CustomObjectScope({'Attention(20)': Attention(20)}):
#     cls_model = load_model(path+"dealString/"+clsname)
# cls_model.load_weights(path+"dealString/"+clsname+".model.hdf5")

cls_model = model
# cls_model = pickle.load(open(path+"dealString/"+clsname, 'rb'))

# print score
# print("Training score: %f" % cls_model.score(X_train_dif, y_train))
scores = cls_model.evaluate(X_test, y_test, verbose=1)
print("Test score: %f" % scores[1])

# print predict and correct ans
# (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool
# print(cls_model.predict_proba(X_test))
print(cls_model.predict(X_test))
# print("predict : ", np.array((cls_model.predict_proba(X_test) >= 0.02).astype(int)).flatten())
print("predict : ", np.array((cls_model.predict(X_test)>=0.05).astype(int)).flatten())
print("label   : ", y_test)

# %%

def pred_in2idx(ipath,file,sents_list):

	word2idx = pickle.load(open(ipath+"dealString/word2idx.pkl", 'rb'))
	# wrtV = open(ipath+"dealString/noneVocab.txt", 'w', encoding='utf-8', newline='')
	sents_idx = []
	for idx, sent in enumerate(sents_list):
		sents_idx.append([])
		for word in sent:
			try:
				sents_idx[idx].append(word2idx[word])
			except KeyError:
				# print(word+" not in vocab")
				sents_idx[idx].append(0)
				# wrtV.write(word+"\n")
	# save inputvec with pickle
	pickle.dump(sents_idx, open(ipath+file+".pkl", 'wb'))

# %%

def pred_seg(ipath,file,cls_model,flag):

	input_list = sentlist(ipath+file+".txt")
	pred_in2idx(ipath,file,input_list)
	# input_seq = token.texts_to_sequences(input_list)
	input_seq = pickle.load(open(ipath+file+".pkl", 'rb'))
	# pad_input_seq = sequence.pad_sequences(input_seq, maxlen=20)
	pad_input_seq = sequence.pad_sequences(np.array(input_seq), maxlen=20, padding='post')
	# predict_result = cls_model.predict_classes(pad_input_seq)
	predict_result = (cls_model.predict(pad_input_seq)>=0.05).astype(int)
	result = np.array(predict_result).flatten()

	print("0 : %d"%(list(result).count(0)))
	print("1 : %d"%(list(result).count(1)))

	wrt0 =  open(path+"dealString/0_Aresult.txt", 'a', encoding='utf-8-sig', newline='')
	wrt1 =  open(path+"dealString/1_Aresult.txt", 'a', encoding='utf-8-sig', newline='')

	for i in range(len(result)):
		if result[i] == 1:
			wrt1.write(" ".join(input_list[i]))
			wrt1.write("\n")
		else:
			wrt0.write(" ".join(input_list[i]))
			wrt0.write("\n")

	if flag==1:
		np.savetxt(path+"dealString/1_Adatares_new.txt", result, delimiter='\n', fmt="%d")
	elif flag==0:
		np.savetxt(path+"dealString/0_Adatares_new.txt", result, delimiter='\n', fmt="%d")

	wrt0.close()
	wrt1.close()

	return pad_input_seq


# %%
path = "Case/doc_clr/merge_/"
# cls_model = pickle.load(open(path+"dealString/"+clsname, 'rb'))
return_sen0 = pred_seg(path,"dealString/0_data",cls_model,0)
return_sen1 = pred_seg(path,"dealString/1_data",cls_model,1)

#%%

import seaborn as sns 
import pandas as pd
import tensorflow as tf
from keras import backend as K
from matplotlib.collections import QuadMesh

def show_weifig(fig_x, fig_y, file_ver, max_size, itersize):

	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus']=False

	sns.set(font=['SimHei'], font_scale=1.5)
	sns.set_style('whitegrid',{'font.sans-serif':['SimHei']})

	fig, ax = plt.subplots(figsize=(fig_x,fig_y))

	ipath = "Case/doc_clr/merge_/"
	file = "dealString/"+str(file_ver)+"_data"
	sent_list = sentlist(ipath+file+".txt")

	res = np.array(np.loadtxt(path+"dealString/"+str(file_ver)+"_Adatares_new.txt", delimiter='\n', dtype=int)).tolist()
	strings = np.array(sent_list)
	for no, string in enumerate(strings):
		if len(string)<20:
			strings[no].extend(" " for _ in range(20-len(string)))
		else:
			strings[no] = strings[no][:20]
		strings[no].append(res[no])

	for no in range(0,max_size,itersize):
		get_att = get_attention(cls_model, return_sen1[no:no+itersize], 5)
		# print(get_att)
		df = pd.DataFrame(get_att)
		df['label'] = pd.DataFrame([0]*itersize)
		labels = np.array(sent_list[no:no+itersize])

		try:
			fig = sns.heatmap(df, annot=labels, cbar=True, linewidths=0.2, square=False, cmap="YlGnBu", fmt = '')
		except:
			pass
		# plt.show()

		figure = fig.get_figure()    
		figure.savefig(path+'dealString/A_weight/'+str(file_ver)+'_att('+str(no)+').png', facecolor='w')
		plt.clf()


#%%

fig_x, fig_y = 35, 28
file_ver = 0
max_size = 25000
itersize = 100

show_weifig(fig_x, fig_y, file_ver, max_size, itersize)


# %%

path = "Case/doc_clr/other/"
# cls_model = pickle.load(open(path+"dealString/"+clsname, 'rb'))
pred_seg(path,"segment/seg_long",cls_model,2)

# %%

def error_seg(flag):

	sents_list = sentlist(path+"dealString/"+str(flag)+"_Aresult.txt")

	with open(path+"dealString/"+str(flag)+"error_Aresult"+".txt", 'w', encoding='utf-8-sig', newline='') as wrtf:
		if flag == 1:
			for s in sents_list:
				if ('具有' not in s) & ('含有' not in s):
					for w in s:
						wrtf.write("%s " % w)
					wrtf.write("\n")
		elif flag == 0:
			for s in sents_list:
				if ('具有' in s) | ('含有' in s):
					for w in s:
						wrtf.write("%s " % w)
					wrtf.write("\n")

error_seg(1)
error_seg(0)


#%%
