
import os, numpy as np

def sentlist(fname):
	f = open(fname, mode='r', encoding = 'utf-8-sig', errors='ignore')
	lines = [line.strip().split(' ') for line in f]
	return lines


if __name__ == "__main__":

	ipath = "Case/doc_clr/"

	# training_data into 2-dim list
	pair_list = sentlist(ipath+"pair_data.txt")
	sents_list = sentlist(ipath+"merge_/segment/seg_long.txt")

	sents_idx = []
	for idx, sent in enumerate(pair_list):
		sents_idx.append([])
		sents_idx[idx].append(sent[0])
		sents_idx[idx].append(sent[2])

	for s in sents_list:
		for pair in sents_idx:
			exs_tag =  all(p in s  for p in pair)
			if (exs_tag==True):
				print_pair = " ".join(pair)
				print("{%s}:%s"%(print_pair,s))
				break
			
	
	# f = open(ipath+'pair_list.txt', 'a', encoding = 'utf-8', errors='ignore')

	# for pair in sents_idx:
	# 	pair=' '.join(pair)
	# 	f.write(pair)
	# 	f.write("\n")
	# f.close()

