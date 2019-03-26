import util_subject_lda
import jieba
import pandas as pd
import gensim

lda_subject_jiage, dictionary_jiage = util_subject_lda.subject_lda('car_jiage.csv')
lda_subject_jiage.save('lda_jiage.model')

print(lda_subject_jiage.print_topic(1, topn=5))

for topic in lda_subject_jiage.print_topics(num_topics=10, num_words=8):
    print(topic)

content = pd.read_csv('car_test.csv', encoding='gbk')
content.dropna(inplace=True)
content = content['content'].values.tolist()



content_00 = "欧蓝德，价格便宜，森林人太贵啦！"
sentences_test = []
util_subject_lda.preprocess_sentence(content_00, sentences_test)

content_doc2bow = dictionary_jiage.doc2bow(sentences_test)

lda_test = gensim.models.ldamodel.LdaModel.load('lda_jiage.model')


content_test = lda_test[content_doc2bow]

print(content_test)
