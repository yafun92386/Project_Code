
import os, sys, re, logging
from bs4 import BeautifulSoup


def remove_html(file,html,log):
	# using BeautifulSoup to parse html into soup object
	soup = BeautifulSoup(html, 'html.parser')
	res = soup.find_all(text=True)

	# remove title
	title_tag = soup.title
	res.remove(title_tag.text)

	# remove page link
	hlink = soup.find_all('a')
	try:
		for s in hlink:
			res.remove(s.text)
	except ValueError:
		log.error(file+" Link Error")
		pass

	# remove duplicate <br>
	while res.count('\n') > 0:
		res.remove('\n')

	return res


def getChinese(context):

	# non-Chinese unicode range ，\uFF0C ; \uFF1B 、\u3001 。\u3002 ?\uFF1F !\uFF01 圖\u5716 表\u8868
	filtrate = re.compile(
		u'[^\u4E00-\u9FA5\uFF0C\uFF1B\u3001\u3002\uFF1F\uFF01]')
	context = filtrate.sub(r'', context)  # remove all non-Chinese characters
	return context


def merge_lines(res):
	# merge ，|: into one line
	res = '\n'.join(res)
	res = res.replace('?', '?\n')
	res = res.replace('!', '!\n')
	res = res.replace(';', ';\n')
	res = res.replace('。', '。\n')
	res = res.replace('？', '？\n')
	res = res.replace('！', '！\n')
	res = res.replace('；', '；\n')
	res = res.replace('\n圖', '\n')
	res = res.replace('\n表', '\n')
	res = res.split('\n')
	return res


def cut2long(res_list):
	new_list = []
	for line in res_list:
		if len(line)>=50 :
			find_iter = [m.start() for m in re.finditer('，', line)]
			if len(find_iter)>=1 :
				pos = int(find_iter[len(find_iter)//2])
				new_list.append(line[:pos])
				new_list.append(line[pos:])
			else:
				new_list.append(line)
		else:
			new_list.append(line)

	return new_list


# if __name__ == "__main__":
def html2txt(ipath,opath):

	# set logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
	log = logging.getLogger('deal_html')

	# merge file
	ofile = opath+'merge.txt'
	if os.path.exists(ofile):
		os.remove(ofile)
	merge_f = open(ofile, 'a', encoding='utf-8', errors='ignore')

	# list files
	files = os.listdir(ipath)
	filelen = len(files)
	tmp = 0

	for fileno, file in enumerate(files):
		if not os.path.isdir(file):
			# remove html tag
			with open(ipath+file, 'r', encoding='utf-8', errors='ignore') as f:
				res_remove_tag = remove_html(file,f,log)
			
			# remove symbol
			res_filter_sym = []
			for s in res_remove_tag:
				res_filter_sym.append(getChinese(str(s)))

			# merge comma into one sentence
			res_merge = merge_lines(res_filter_sym)

			res_merge = cut2long(res_merge)


			# write into txt file
			with open(opath+file+".txt", 'w', encoding='utf-8', errors='ignore') as wrt:
				for w in res_merge:
					if (w=='')|(w=='\n'):
						continue
					else:
						wrt.write(w+'\n')
						merge_f.write(w+'\n')
			
			# set logging for processing
			prec = int(fileno/filelen*100)
			if (prec%10==0) & (tmp != prec):
				tmp = prec
				log.info('Deal '+str(prec)+'% lines')
	
	log.info('Deal html finish')

	pass
