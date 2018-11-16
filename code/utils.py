import os
import pandas as pd
import numpy as np
import tensorflow as tf
import jieba

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]

def get_stop_word_set(only_punctuation=False):
    words_set = set()
    fname = '../data/哈工大停用标点表.txt' if only_punctuation else '../data/哈工大停用词表扩展.txt' 
    with open(fname) as f_r:
        for line in f_r:
            words_set |= set(line.strip())
    if only_punctuation:
        words_set |= set([' '])
    return words_set

def get_word2id(data_path, all_subjects, train_fname, val_fname, test_fname, w2v, pre_processed, save_fname, suffix='_cut_word_rst.txt'):
    '''构造 word id 映射'''
    save_fname = data_path + save_fname + '.txt'
    print(save_fname)
    word2id = {}
    max_len, max_aspect_len = 0, 0
    if pre_processed:
        crt_cnt=0
        with open(save_fname) as f_r:
            for line in f_r:
                crt_cnt += 1
                if crt_cnt == 1:
                    max_len, max_aspect_len = line[:-1].split(' ')
                    max_len, max_aspect_len = int(max_len), int(max_aspect_len)
                else:
                    tmp = line[:-1].split(' ')
                    word2id[tmp[0]] = int(tmp[1])
    else:
        word2id['<pad>'] = 0
        for s in all_subjects:
            crt_len = 0
            for word in s.split(' '):
                if word in w2v:
                    crt_len += 1
                    if word not in word2id:
                        word2id[word] = len(word2id)
            max_aspect_len = max(crt_len, max_aspect_len)
                
        for file_path in [train_fname, val_fname, test_fname]:
            file_path = data_path + file_path + suffix
            with open(file_path) as f_r:
                for line in f_r:
                    crt_len = 0
                    for word in line.strip().split(' '):
                        if word in w2v:
                            crt_len += 1
                            if word not in word2id:
                                word2id[word] = len(word2id)
                    max_len = max(crt_len, max_len)
        
        with open(save_fname, 'w') as fsave:
            fsave.write('%d %d\n' % (max_len, max_aspect_len))
            for item in sorted(word2id.items(), key=lambda x:x[1]):
                fsave.write(item[0]+' '+str(item[1])+'\n')
                
    return word2id, max_len, max_aspect_len
    
def build_nn_context(data_path, file_name, word2id, pre_processed, context_max_len, suffix='_cut_word_rst.txt'):
    if pre_processed:
        pass
    else:
        contexts, context_lens = [], []
        file_path = data_path + file_name + suffix
        with open(file_path) as f_r:
            for line in f_r:
                words = [word2id[w] for w in filter(lambda x:x in word2id, line.strip().split(' '))]
                if len(words) < context_max_len:
                    crt_content_lens = len(words)
                    words = words + [0] * (context_max_len-len(words))
                else:
                    crt_content_lens = context_max_len
                    words = words[:context_max_len]
                contexts.append(words)
                context_lens.append(crt_content_lens)
        return np.asarray(contexts), np.asarray(context_lens)
    
    
def load_word_embeddings(word2id, com_w2v, word_char_emb=False):
    com_w2v_embedding_dim = com_w2v.vector_size
    word2vec = {}
    fnl_word2vec = np.random.uniform(-0.01, 0.01,
                                     [len(word2id), com_w2v_embedding_dim])

    contain_w_cnt = 0
    for w, w_id in word2id.items():
        if w in com_w2v:
            contain_w_cnt += 1
            crt_word_vec = com_w2v[w]
        elif w in word2vec:
            crt_word_vec = word2vec[w]
        else:
            tmp_word_vec = np.random.uniform(-0.01, 0.01,
                                             [com_w2v_embedding_dim])
            word2vec[w] = tmp_word_vec
            crt_word_vec = tmp_word_vec
        fnl_word2vec[word2id[w]] = crt_word_vec

    print(len(word2vec))
    print('contain rate:%d/%d' % (contain_w_cnt, len(word2id)))
    fnl_word2vec[word2id['<pad>'], :] = 0
    return fnl_word2vec, com_w2v_embedding_dim    
    
def build_aspect(word2id, all_subjects, aspect_max_len):
    subject_rst, subject_lens = [], []
    for subject in all_subjects:
        words = []
        for w in subject.split(' '):
            words.append(word2id[w])
        if len(words) > aspect_max_len:
            subject_lens.append(aspect_max_len)
            words = words[:aspect_max_len]
        else:
            subject_lens.append(len(words))
            words = words + [0] * (aspect_max_len - len(words))
        subject_rst.append(words)
    return np.asarray(subject_rst), np.asarray(subject_lens)    

def build_labels_ws(data_path, file_name, all_subjects, cost_w):
    data = pd.read_csv(data_path + file_name + '.csv', usecols=all_subjects).values
    ans, ws = [], []
    for items in data:
        crt_ans, crt_ws = [], []
        for i, c in zip(items, cost_w):
            tmp = [0, 0, 0, 0]
            tmp[i+2] = 1
            crt_ans.append(tmp)
            crt_ws.append(c[i+2])
        ans.append(crt_ans)
        ws.append(crt_ws)
    return np.asarray(ans), np.asarray(ws)

def get_char2id(data_path, all_subjects, train_fname, val_fname, test_fname, w2v, pre_processed, save_fname, suffix='_cut_char_rst.txt'):
    '''构造 char id 映射'''
    save_fname = data_path + save_fname + '.txt'
    print(save_fname)
    word2id = {}
    max_len, max_aspect_len = 0, 0
    if pre_processed:
        crt_cnt=0
        with open(save_fname) as f_r:
            for line in f_r:
                crt_cnt += 1
                if crt_cnt == 1:
                    max_len, max_aspect_len = line[:-1].split(' ')
                    max_len, max_aspect_len = int(max_len), int(max_aspect_len)
                else:
                    tmp = line[:-1].split(' ')
                    word2id[tmp[0]] = int(tmp[1])
    else:
        word2id['<pad>'] = 0
        for s in all_subjects:
            crt_len = 0
            for word in s:
                if len(word.strip()) == 0: continue
                if word in w2v:
                    crt_len += 1
                    if word not in word2id:
                        word2id[word] = len(word2id)
            max_aspect_len = max(crt_len, max_aspect_len)
                
        for file_path in [train_fname, val_fname, test_fname]:
            file_path = data_path + file_path + suffix
            with open(file_path) as f_r:
                for line in f_r:
                    crt_len = 0
                    for word in line.strip().split(' '):
                        if word in w2v:
                            crt_len += 1
                            if word not in word2id:
                                word2id[word] = len(word2id)
                    max_len = max(crt_len, max_len)
        
        with open(save_fname, 'w') as fsave:
            fsave.write('%d %d\n' % (max_len, max_aspect_len))
            for item in sorted(word2id.items(), key=lambda x:x[1]):
                fsave.write(item[0]+' '+str(item[1])+'\n')
    return word2id, max_len, max_aspect_len    


def re_weigth(x, factor=200):
    x = np.asarray(x)
    x = x * factor
    x = np.exp(x)
    x /= sum(x)
    return x


def get_model_predict(model,
                      score_detail,
                      predict_params,
                      max_save_num=7):
    all_preds = []
    for score in score_detail[-max_save_num:]:
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(model.sess, score[0])
        del saver
        all_preds.append(model.predict_data(predict_params))
    all_preds = np.asarray(all_preds)
    return all_preds


def ensemble_a_model(all_preds,
                     score_detail,
                     subjects_eng,
                     factor=200,
                     max_save_num=7,
                     ensemble_aspects=True):

    submit = pd.DataFrame()
    ensemble_prob = []
    if ensemble_aspects:
        for i, subject in enumerate(subjects_eng):
            all_aspect_f1 = [item[2][i] for item in score_detail[-max_save_num:]]
            crt_ws = re_weigth(all_aspect_f1, factor)
            #             print([(i,j) for i,j in zip(all_aspect_f1, crt_ws)])
            crt_re_w = np.sum(
                [preds[:, i, :] * w for w, preds in zip(crt_ws, all_preds)], 0)
            submit[subject] = np.argmax(crt_re_w, 1) - 2
            ensemble_prob.append(crt_re_w)
        ensemble_prob = np.asarray(ensemble_prob)
    else:
        crt_ws = re_weigth(
            [np.asarray(item[1])[:, 0].mean() for item in all_preds],
            factor)
        print(crt_ws)
        all_preds2 = np.sum([w * p for w, p in zip(crt_ws, all_preds)], axis=0)
        for i, subject in enumerate(subjects_eng):
            submit[subject] = np.argmax(all_preds2[:, i, :], axis=1) - 2
        ensemble_prob = all_preds2
    return submit, ensemble_prob


def cal_local_f1(df_val_data, df_val_preds, subjects_eng):
    from sklearn.metrics import f1_score
    f1, f1_detail = 0, []
    for col in subjects_eng:
        tmp_f1 = f1_score(df_val_data[col], df_val_preds[col], average='macro')
        f1 += tmp_f1
        f1_detail.append(tmp_f1)
        print(col, tmp_f1)
    print(f1 / len(subjects_eng))
    return f1 / len(subjects_eng), f1_detail

def ensemble_multi_model(all_preds,
                     f1s,
                     subjects_eng,
                     factor=200,
                     ensemble_aspects=True):

    submit = pd.DataFrame()
    ensemble_prob = []
    if ensemble_aspects:
        for i, subject in enumerate(subjects_eng):
            all_aspect_f1 = [item[1][i] for item in f1s]
            crt_ws = re_weigth(all_aspect_f1, factor)
            #             print([(i,j) for i,j in zip(all_aspect_f1, crt_ws)])
#             print(all_aspect_f1)
            crt_re_w = np.sum(
                [preds[i, :, :] * w for w, preds in zip(crt_ws, all_preds)], 0)
            submit[subject] = np.argmax(crt_re_w, 1) - 2
            ensemble_prob.append(crt_re_w)
        ensemble_prob = np.asarray(ensemble_prob)
    else:
        crt_ws = re_weigth(
            [item[0] for item in f1s],
            factor)
        print(crt_ws)
        all_preds2 = np.sum([w * p for w, p in zip(crt_ws, all_preds)], axis=0)
        for i, subject in enumerate(subjects_eng):
            submit[subject] = np.argmax(all_preds2[i, :, :], axis=1) - 2
        ensemble_prob = all_preds2
    return submit, ensemble_prob