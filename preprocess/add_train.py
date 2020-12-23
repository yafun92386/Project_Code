

def sentlist(in_fname,mode):
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
		if mode == 0:
			sentls.append(tmp_s)
		else:
			tmp = []
			tmp.append(tmp_s[0])
			tmp.append(tmp_s[2])
			sentls.append(tmp)


	f.close()
	return sentls


if __name__ == '__main__':

	path = "../textbook_data/html_txt/result1_2/"
	ipath = "indexRel_1rel/"
	ppath = "b_pairs_index/RBF_SVM/"
	
	# read pair file into 2-dim list
	train_word = sentlist(path+ipath+"1_data.txt",1)
	test_word = sentlist(path+ppath+"_result1.txt",0)

	for i in range(0,len(test_word)):
		if test_word[i] in train_word:
			print(train_word.index(test_word[i])+1)
		else:
			print(test_word[i])