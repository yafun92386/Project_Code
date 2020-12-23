
import os, sys
import remove_html
import segment
import find_str

if __name__ == "__main__":

	setfile = "merge_/"

	# html_2_txt
	txt_ipath = "Case/doc_clr/"+setfile+"html/"
	txt_opath = "Case/doc_clr/"+setfile+"htmltxt/"

	# remove_html.html2txt(txt_ipath,txt_opath)

	# txt_segment
	jieba_path = r"C:/Users/KDD_yafun/pythonwork/jieba_dict/"
	filepath = 'Case/doc_clr/'+setfile
	setence_min_len = 20

	# segment.deal_seg(jieba_path,filepath,setence_min_len)

	root_path = "Case/doc_clr/"+setfile
	existls = ['具有','含有']

	find_str.madeTrainset(root_path,existls)
	
	



	pass