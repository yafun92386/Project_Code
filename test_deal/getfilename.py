
import os,csv

if __name__ == "__main__":

	ipath = "../doc_clr/textbook_data/html_txt/result1_2/"
	# iipath = ipath+"existStr/"
	# opath = "existRel/"
	iipath = ipath+"tmpexistStr/"

	# idxslist = indexlist(ipath+"index_terms.txt")
	# relslist = relist(ipath+"relation_terms.txt")

	files = os.listdir(iipath)

	with open(ipath+"tmp_existStr_idx.txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
		for file in files: #遍历文件夹
			filename = os.path.splitext(file)[0]
		# sentslist = sentlist(iipath+file)

			# wrtline = s.split(' ')
			# for item in s:
				# item = item.strip()
			wrtf.write("%s\n"%filename)
			# wrtf.write("\n")

		# for s in sentslist:
		# 	for r in relslist:
		# 		if r in s:
		# 			with open(ipath+opath+r+".txt",'a', encoding = 'utf-8-sig',newline='') as wrtf:
		# 					# wrtline = s.split(' ')
		# 				for item in s:
		# 					# item = item.strip()
		# 					wrtf.write("%s "%item)
		# 				wrtf.write("\n")
		# 				break
			print(file)