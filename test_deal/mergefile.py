# coding=UTF-8
import os, pickle
from os import listdir
from os.path import isfile, isdir, join, splitext


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



# ipath = "../textbook_data/html_txt/result1_2/"
ipath = "Case/doc_clr/merge/dealString/"
# opath = "relStr/"

files = listdir(ipath)

fo = open(ipath+'1_data.txt', 'a', encoding = 'utf-8', errors='ignore')
# fo = open(ipath+'0_data.txt', 'a', encoding = 'utf-8', errors='ignore')
txtlist = []

# read file into list with rule
for file in files: 
	fullpath = join(ipath, file)
	filename = splitext(file)[0]
	# 1_data rule
	rule = (filename=="含有")|(filename=="具有")|(filename=="1_data")


	if rule==True:

		f = open(fullpath, 'r', encoding = 'utf-8', errors='ignore')
		for txt in f:
			txtls = txt.split(' ')
			if txtls not in txtlist:
				txtlist.append(txtls)
		
		print(file)

# write txt into fo file
for txt in txtlist:
	fo.write(" ".join(txt))

fo.close()




# get label with training data
label = get_data(ipath,"label.txt")

# get label with pickle
# label = pickle.load(open(path+iipath+"models_1rel/"+lbname, 'rb'))
# label = pickle.load(open(path+iipath+lbname, 'rb'))