import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import word2vec
from gensim import models
import jieba
import jieba.posseg as pseg

def relationlist(fname):
    f = open(fname, 'r', encoding = 'utf-8-sig', errors='ignore')
    rells = []
    for rels in f.readlines():
    	rels = rels.replace(u'\xa0', u'')
    	# replace all \n
    	rels = rels.replace('\n','')
    	# skip space
    	if (rels == ' ') or (rels == ''):
    		continue
    	rells.append(rels)
    f.close()
    return rells 


if __name__ == '__main__':

	path = "doc_clr/textbook_data/html_txt/result1_2/"
	rel_name = "given_relation_terms.txt"
	model_sel = "models_seg/seg"

	model = models.Word2Vec.load(path+model_sel+".model")

	relslist = relationlist(path+rel_name)

	seg_list = []
	for r in relslist:
		print(r+"=========================")
		seg_list.append(r)
		try:
			result = model.most_similar(r,topn = 10)
			for items in result:
				for i in pseg.cut(items[0]):
					if i.flag =='v':
						print(items[0]+","+str(items[1]))
						if items[0] in seg_list:
							continue
						else:
							seg_list.append(items[0])
				
		except KeyError:
			print("word "+r+" not in vocabulary!")
			

	with open(path+"given_relation_terms_new.txt",'w', encoding = 'utf-8-sig',newline='') as wrtf:
			# wrtline = s.split(' ')
		for item in seg_list:
			# item = item.strip()
			wrtf.write("%s\n"%item)