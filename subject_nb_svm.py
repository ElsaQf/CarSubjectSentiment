import pandas as pd
import jieba
import random

# 载入停用词
stopwords = pd.read_csv('stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopwords'], encoding='utf-8')
stopwords = stopwords['stopwords'].values
# 载入文本
comment_df = pd.read_csv('train.csv', encoding='utf-8')
comment_df.dropna(inplace=True)
# 将content分成不同的subject
content = comment_df['content'].values
subject = comment_df['subject'].values

# 主题词转换成类别0，1，2
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
subject_toDigit = labelencoder.fit_transform(subject)
# 分词
def preprocess_text(content_lines, sentences, category):
    i = 0
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]
            segs = list(filter(lambda x: x.strip(), segs))
            segs = list(filter(lambda x: len(x)>1, segs))
            segs = list(filter(lambda x: x not in stopwords, segs))
            sentences.append((" ".join(segs), category[i]))
            i += 1
        except Exception:
            print(line)
            continue

sentences = []
preprocess_text(content, sentences, subject_toDigit)
random.shuffle(sentences)

# 抽取词向量特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word',
    max_features=4000,
)
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 模型：NB
from sklearn.naive_bayes import MultinomialNB
classifier_NB = MultinomialNB()
classifier_NB.fit(X_train, y_train)
y_pred_NB = classifier_NB.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_NB)

sum = 0
for i in range(10):
    sum += cm[i][i]

# 模型：SVM
from sklearn.svm import SVC
classifier_SVM = SVC(kernel='linear')
classifier_SVM.fit(X_train, y_train)
y_pred_SVM = classifier_SVM.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_SVM)

