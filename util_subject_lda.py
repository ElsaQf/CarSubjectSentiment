import pandas as pd
import jieba
from gensim import corpora, models, similarities
import gensim

stopwords = pd.read_csv('stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopwords'], encoding='utf-8')
stopwords = stopwords['stopwords'].values

def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]
            segs = list(filter(lambda x: x.strip(), segs))
            segs = list(filter(lambda x: len(x)>1, segs))
            segs = list(filter(lambda x: x not in stopwords, segs))
            sentences.append(segs)
        except Exception:
            print(line)
            continue
def preprocess_sentence(content_sentence, sentences):
    segs = jieba.cut(content_sentence)
    segs = [v for v in segs if not str(v).isdigit()]
    segs = list(filter(lambda x: x.strip(), segs))
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    sentences.append(" ".join(segs))


def subject_lda(fileName):
    comment_df = pd.read_csv(fileName, encoding='gbk')
    comment_df.dropna(inplace=True)
    content = comment_df['content'].values.tolist()

    sentences = []
    preprocess_text(content, sentences)

    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

    return lda, dictionary
