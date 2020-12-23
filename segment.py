
import os, jieba, logging
import jieba.posseg as pseg

def preset(jieba_path,filepath):

	# set logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
	log = logging.getLogger('deal_segment')

	# jieba custom setting
	# jieba_path = r"C:/Users/KDD_yafun/pythonwork/jieba_dict/"
	jieba.set_dictionary(jieba_path+'dict.txt.big')  
	jieba.load_userdict(jieba_path+'dictionary_merge.txt')
	jieba.dt.cache_file = 'jieba.cache.new'

	# load stopwords set
	stopword_set = set()
	with open(jieba_path+'stopwords_.txt','r', encoding='utf-8') as stopwords:
		for stopword in stopwords:
			stopword_set.add(stopword.strip('\n'))

	return log,stopword_set


def deal_seg(jieba_path,filepath,set_len=10):

	# dynamic announce globals variable
	outputs = globals()
	tags = globals()

	# precessing jieba dict
	log, stopword_set = preset(jieba_path,filepath)

	#  read file len
	filelen = len(open(filepath+'htmltxt/merge.txt','r', encoding='utf-8').readlines())

	# segment file list
	seglist = ['segment/seg.txt','segment/seg_long.txt','segment/seg_tag.txt','segment/seg_fre_n.txt','segment/seg_fre_v.txt']
	# delete exist file
	for name in seglist:
		if os.path.exists(filepath+name):
			os.remove(filepath+name)

	# set file
	for i in range(1,6):
		outputs["output%i"%i] = open(filepath+seglist[i-1], 'w', encoding='utf-8')

	# tmp for deal counting
	tmp = 0
	# freq dict for noun/verb counting 
	noun_freq = {}
	verb_freq = {}
	
	log.info('Start segment')
	# start semgment enumerate file content
	content = open(filepath+'htmltxt/merge.txt', 'r', encoding='utf-8')
	for texts_num, line in enumerate(content):
		# precessing
		line = line.strip()
		line = line.strip('\n')
		line = ''.join(line)  

		# tag for check wether write
		for i in range(1,4):
			tags["tag%i"%i] = False
		
		# segment all
		words = pseg.cut(line)
		for word, flag in words:
			if (word not in stopword_set) & ((flag!="m" )&( flag!="M")): 

				# segment all setence
				tags["tag1"] = True
				outputs["output1"].write(word + ' ')
				# segment long sentence
				if len(line)>set_len:
					tags["tag2"] = True
					outputs["output2"].write(word + ' ')
				# segment with tag
				tags["tag3"] = True
				outputs["output3"].write(word +':'+flag+' ')
				
				# counting noun & verb freqence
				# n名詞 nr人名 ns地名 nt機構團體明 nz其他專名 nl名詞慣用 ng名詞性語素
				if flag in ['n','ns','nz','x']:
					if word not in noun_freq:
						noun_freq[word]=1                     
					else:                         
						noun_freq[word]= noun_freq.get(word)+1
				# v動詞 vd副動詞 vn動名詞 vi不及物動詞 vt及物動詞
				if flag in ['v','vd','vn','Vi','Vt','Nv']:
					if word not in verb_freq:
						verb_freq[word]=1                 
					else:
						verb_freq[word]= verb_freq.get(word)+1

		# if write, write "\n" into file 
		for i in range(1,4):
			if (tags["tag%i"%i] == True):
				outputs["output%i"%i].write("\n")
		
		# set logging for processing
		prec = int(texts_num/filelen*100)
		if (prec%10==0) & (tmp != prec):
			tmp = prec
			log.info('Deal '+str(prec)+'% lines')
	
	log.info('Deal segment finish')

		
	# ouput noun freq
	noun = sorted(noun_freq.items(), key=lambda d: d[1], reverse=True)
	for term in noun:
		outputs["output4"].write(term[0]+' '+str(term[1])+'\n')
	log.info('Noun finish')   

	# output verb freq
	verb = sorted(verb_freq.items(), key=lambda d: d[1], reverse=True)
	for term in verb:
		outputs["output5"].write(term[0]+' '+str(term[1])+'\n')
	log.info('Verb finish') 

	# close file
	for i in range(1,6):
		outputs["output%i"%i].close()
