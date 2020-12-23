
import logging, pickle

from gensim.models import word2vec
from keras.models import load_model

def train_word2vec(ver):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    sentences = word2vec.LineSentence("Case/doc_clr/merge_/segment/"+ver+".txt")
    model = word2vec.Word2Vec(sentences, alpha=0.025, min_alpha=0.0001, window=3, min_count=1,
                                         size=100, iter=8, batch_words=100, cbow_mean=0)
    # (sentences=None, size=100, alpha=0.025, window=5, min_count=5,
    #  max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001,
    #  sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,
    #  iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

    pickle.dump(model, open("Case/doc_clr/merge_/w2vModel/"+ver+"_w2v.model", 'wb'))


def termlist(fname):
	f = open(fname, 'r', encoding = 'utf-8-sig', errors='ignore')
	lines = [line.strip() for line in f]
	return lines


if __name__ == "__main__":

    ver = "seg_long"

    train_word2vec(ver)

    model = pickle.load(open("Case/doc_clr/merge_/w2vModel/"+ver+"_w2v.model", 'rb'))

    termlist = termlist("Case/doc_clr/merge_/dictionary/index_terms.txt")

    wrtf = open("Case/doc_clr/merge_/w2vModel/"+ver+"_w2v_similar.txt", 'w', encoding='utf-8')
    nonf = open("Case/doc_clr/merge_/w2vModel/"+ver+"_non_vec.txt", 'w', encoding='utf-8')

    for t in termlist:
        try:
            # print(model.most_similar(t,topn=5))
            wrtf.write(','.join(map(str, model.most_similar(t,topn=5)))+"\n")
        except KeyError:
            wrtf.write(t+" none\n")
            # print(t+" not in vocabulary")
            nonf.write(t+"\n")

    wrtf.close()
    nonf.close()

    
