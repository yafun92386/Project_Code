# -*- coding: UTF-8 -*-

import os,csv,jieba
import jieba.posseg as pseg


def readSource(fname):

	with open(fname, newline='',encoding='utf-8-sig') as csvfile:

		col_list=[]
		dic_csv = csv.DictReader(csvfile,[fname,'1','2','3','4','5','6','7','8','9','10'])

		row_count = 0
		# read csv to list
		for row in dic_csv:
			# skip fieldnames
			if row_count==0:
				row_count = row_count+1
				continue
			# read row's value into col_list
			else:
				for key in dic_csv.fieldnames:
					# skip first col
					if key==fname:
						continue
					# skip none value
					elif row[key] is None:
						continue
					# write [word,tfidf] into list
					else:
						col_list.append(row[key].split(','))
	return col_list


def intoDictionary(lst):

	dictionary = {}
	# read col_list into dictionary
	for index in lst:
		# if exists that add tfidf
		if index[0] in dictionary:
			dictionary[index[0]] = str(float(dictionary.get(index[0]))+float(index[1]))
		# else add new key,value
		else:
			dictionary[index[0]] = index[1]

	return dictionary


def wrtSumTfidf(ipath,fname,dic):

	jieba.load_userdict('../../jieba_dict/dictionary_merge.txt')

	# sort tfidf
	sorted_by_value = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)

	item_list=[]
	item_list.append(fname)
	# store info. into list
	# -- for range --
	for item in range(len(sorted_by_value)):
		if item >= 20:
			break
		if sorted_by_value[item][0] == "產生":
			continue
		# find n then continue, else skip
		wflag = 0
		for key in pseg.cut(sorted_by_value[item][0]):
			if key.flag in ['n','ns','nt','nz','nrt','x']:
				wflag = 0
				break
			else:
				wflag = 1
		if wflag == 1:
			continue
		else:
			item_list.append(sorted_by_value[item][0]+','+sorted_by_value[item][1])
	# -- for limit --
	# for item in sorted_by_value:
	#     # set sum_tfidf limit
	#     if float(item[1])>0.6:
	#         item_list.append(item[0]+','+item[1])


	# del file
	# my_file = Path(openfile)
	# if my_file.exists():
	#     os.remove('sum_tfidf.csv')

	# write term's sum_tfidf
	with open (ipath+"sum_tfidf.csv",'a', encoding = 'utf-8-sig',newline='') as csvfile:
		wrfile = csv.writer(csvfile)
		wrfile.writerow(item_list)
		print(fname+' finish')


if __name__ == "__main__":

	ipath = "../textbook_data/html_txt/result1_2/"
	iipath = ipath+"csv/"

	files = os.listdir(iipath)

	for file in files:

		filename = os.path.splitext(file)[0]
		# read file info.
		sourse_list = readSource(iipath+file)

		if sourse_list!=[]:
			# read sourse_list into dictionary
			dic_word = intoDictionary(sourse_list)

			# write sum of tfidf int csv
			wrtSumTfidf(ipath,filename, dic_word)
