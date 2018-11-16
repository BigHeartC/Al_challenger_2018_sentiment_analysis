import numpy as np
import pickle


def get_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(line.strip())
    return data


train_content_ori = get_data(
    '../data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset_cut_word_rst.txt')
val_content_ori = get_data(
    '../data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset_cut_word_rst.txt')
test_content_ori = get_data(
    '../data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa_cut_word_rst.txt')

print(len(train_content_ori), len(val_content_ori), len(test_content_ori))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(train_content_ori)

train_content = vectorizer.transform(train_content_ori)
val_content = vectorizer.transform(val_content_ori)
test_content = vectorizer.transform(test_content_ori)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20 * 4, n_iter=7, random_state=2018)
svd.fit(train_content)

train_svd = svd.transform(train_content)
val_svd = svd.transform(val_content)
test_svd = svd.transform(test_content)

prefix = 'svd_tfidf_withP_80'
np.save('../data/%s_train' % prefix, train_svd)
np.save('../data/%s_val' % prefix, val_svd)
np.save('../data/%s_test' % prefix, test_svd)
pickle.dump(svd, open('../data/%s.pk' % prefix, 'wb'))
