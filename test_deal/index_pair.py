from os import listdir, remove
from os.path import isfile, isdir, join, splitext, getsize

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

if __name__ == "__main__":

	ipath = "../doc_clr/textbook_data/html_txt/result1_2/"
	ppath = "b_pairs/"
	pipath = "b_pairs_index/"

	idxs_list = indexlist(ipath+"index_merge.txt")

	files = listdir(ipath+ppath)

	for file in files:

		inpath = join(ipath+ppath, file)
		outpath = join(ipath+pipath, file)
		filename = splitext(file)[0]

		if isfile(inpath) & (splitext(file)[-1] == '.txt') & (filename!="pair_index") & (filename!="pair_words"):

			f = open(inpath, 'r', encoding = 'utf-8', errors='ignore')
			fo = open(outpath, 'a', encoding = 'utf-8', errors='ignore')
			for txt in f:
				txt = txt.replace('\n','')
				txtls = txt.split(' ')
				if (txtls[0] in idxs_list) & (txtls[1] in idxs_list):
					fo.write(txt)
					fo.write("\n")
			
			fo.close()
			if getsize(outpath)==0:
				remove(outpath)
			f.close()
			
			print(file)

	


	# # pair_words into 2-dim list
	# sents_list = sentlist(path+ppath+"pair_words.txt")

	# idxs_list = indexlist(path+"index_merge.txt")

	# fo = open(join(path, ppath)+'/pair_index.txt', 'a', encoding = 'utf-8', errors='ignore')

	# for pair in sents_list:
	# 	if (pair[0] in idxs_list) & (pair[1] in idxs_list):
	# 		wrt = ' '.join(pair)
	# 		wrt = wrt+"\n"
	# 		fo.write(wrt)

	# print("finish!")


		
	