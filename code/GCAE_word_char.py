import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
import random
import numpy as np
import pandas as pd
from utils import get_batch_index
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
from keras.layers import Bidirectional, CuDNNLSTM, CuDNNGRU, SpatialDropout1D
import pickle


class GCAE_expand_model(object):
    def __init__(self, config, sess):
        self.sess = sess
        self.seed = 2018
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.n_sub_class = config.n_sub_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout_keep = config.dropout_keep
        self.max_context_len = config.max_context_len
        self.max_aspect_len = config.max_aspect_len
        self.embedding_matrix2 = config.embedding_matrix
        self.early_stop = config.early_stop
        self.id2word = config.id2word
        self.val_num = config.val_num
        self.random_del_prob = config.random_del_prob
        self.feature_eng_size = config.feature_eng_size
        self.use_char_emb = config.use_char_emb
        if config.use_char_emb:
            self.embedding_dim_ch = config.embedding_dim_ch
            self.embedding_matrix_ch = config.embedding_matrix_ch
            self.char_hidden_size = config.char_hidden_size
            self.max_char_len = config.max_char_len

        self.kernel_num = config.kernel_num
        self.kernel_sizes = config.kernel_sizes

    def build_model(self):
        with tf.name_scope('inputs'):
            self.aspects = tf.placeholder(tf.int32, [None, self.n_class, self.max_aspect_len])
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class, self.n_sub_class])
            self.context_lens = tf.placeholder(tf.int32, None)
            self.embedding_matrix = tf.get_variable(name='embedding', initializer=self.embedding_matrix2,
                                                    trainable=True, dtype=tf.float32)
            if self.use_char_emb:
                self.aspects_ch = tf.placeholder(tf.int32, [None, self.n_class, self.max_aspect_len, self.max_char_len])
                self.aspects_ch_lens = tf.placeholder(tf.int32, [None])
                self.contexts_ch = tf.placeholder(tf.int32, [None, self.max_context_len, self.max_char_len])
                self.contexts_ch_lens = tf.placeholder(tf.int32, [None])
                self.embedding_matrix_ch = tf.get_variable(name='embedding_ch', initializer=self.embedding_matrix_ch,
                                                           trainable=True, dtype=tf.float32)

            self.dropout_keep_prob = tf.placeholder(tf.float32)
            self.cost_ws = tf.placeholder(tf.float32, [None, self.n_class])
            self.feature_eng = tf.placeholder(tf.float32, [None, self.feature_eng_size])

            batch_size = tf.shape(self.contexts)[0]
        with tf.name_scope('emb'):
            if self.use_char_emb:
                with tf.variable_scope("char"):
                    context_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.embedding_matrix_ch, self.contexts_ch), [-1, self.max_char_len, self.embedding_dim_ch])
                    context_emb = tf.nn.dropout(context_emb, keep_prob=self.dropout_keep_prob)

                    aspect_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.embedding_matrix_ch, self.aspects_ch), [-1, self.max_char_len, self.embedding_dim_ch])
                    aspect_emb = tf.nn.dropout(aspect_emb, keep_prob=self.dropout_keep_prob)

                    cell_fw = tf.contrib.rnn.GRUCell(self.char_hidden_size)
                    cell_bw = tf.contrib.rnn.GRUCell(self.char_hidden_size)

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, aspect_emb, self.aspects_ch_lens, dtype=tf.float32)
                    aspect_emb = tf.concat([state_fw, state_bw], axis=1)

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, context_emb, self.contexts_ch_lens, dtype=tf.float32)
                    context_emb = tf.concat([state_fw, state_bw], axis=1)

                    context_emb = tf.reshape(context_emb, [-1, self.max_context_len, 2 * self.char_hidden_size])
                    aspect_emb = tf.reshape(aspect_emb,
                                            [-1, self.n_class, self.max_aspect_len, 2 * self.char_hidden_size])
            with tf.variable_scope("word"):
                aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.aspects)
                aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

                context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
                context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

            if self.use_char_emb:
                aspect_inputs = tf.concat([aspect_emb, aspect_inputs], axis=-1)
                context_inputs = tf.concat([context_emb, context_inputs], axis=-1)
                fnl_embedding_dim = self.embedding_dim + 2 * self.char_hidden_size
            else:
                fnl_embedding_dim = self.embedding_dim

        with tf.name_scope('weights'):
            weights = {
                'aspect': tf.get_variable(
                    name='aspect_w',
                    shape=[fnl_embedding_dim, self.kernel_num],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[len(self.kernel_sizes) * self.kernel_num * self.n_class + self.feature_eng_size,
                           self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax0': tf.get_variable(
                    name='W_0',
                    shape=[len(self.kernel_sizes) * self.kernel_num * self.n_class + self.feature_eng_size,
                           len(self.kernel_sizes) * self.kernel_num * self.n_class + self.feature_eng_size],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax_aspects': [tf.get_variable(
                    name='W_l_%d' % i,
                    shape=[len(self.kernel_sizes) * self.kernel_num + self.feature_eng_size, self.n_sub_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],

            }

        with tf.name_scope('biases'):
            biases = {
                'aspect': tf.get_variable(
                    name='aspect_b',
                    shape=[self.kernel_num],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class * self.n_sub_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax0': tf.get_variable(
                    name='B_0',
                    shape=[len(self.kernel_sizes) * self.kernel_num * self.n_class + self.feature_eng_size],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax_aspects': [tf.get_variable(
                    name='B_l_%d' % i,
                    shape=[self.n_sub_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'softmax_fnl': tf.get_variable(
                    name='B_fnl',
                    shape=[self.n_sub_class * self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('conv1d'):
            conv_size = self.n_hidden
            conv1d = {
                'convs1_ws': [tf.get_variable(
                    name='convs1_w%d' % k,
                    shape=[k, conv_size, self.kernel_num],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
                              for k in self.kernel_sizes],
                'convs2_wss': [[tf.get_variable(
                    name='convs2_w%d_%d' % (i, k),
                    shape=[k, conv_size, self.kernel_num],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
                                for k in self.kernel_sizes] for i in range(self.n_class)],
                'convs3_wss': [[tf.get_variable(
                    name='convs3_w%d_%d' % (i, k),
                    shape=[2, self.kernel_num, self.kernel_num],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
                                for k in self.kernel_sizes] for i in range(self.n_class)],
            }

        with tf.name_scope('gate-cnn'):
            aspect_avg = tf.reduce_mean(aspect_inputs, axis=2)
            aspect_avg = tf.reshape(aspect_avg, [-1, fnl_embedding_dim])
            new_aspects = tf.matmul(aspect_avg, weights['aspect']) + biases['aspect']
            new_aspects = tf.reshape(new_aspects, [-1, self.n_class, self.kernel_num])
            all_aspects = tf.split(new_aspects, [1] * self.n_class, 1)

            context_outputs = CuDNNLSTM(self.n_hidden, return_sequences=True)(context_inputs)
            content_reps = [tf.nn.tanh(tf.nn.conv1d(context_outputs, conv, 1, 'SAME')) for conv in conv1d['convs1_ws']]
            self.represent_reps = []
            for idx, a_aspect in enumerate(all_aspects):
                aspect_rel_reps = [tf.nn.conv1d(context_outputs, conv, 1, 'SAME') for conv in conv1d['convs2_wss'][idx]]
                x = [i + a_aspect for i in aspect_rel_reps]
                x = [tf.nn.relu(x_i) for x_i in x]

                x = [i * j for i, j in zip(content_reps, x)]
                x = [tf.nn.conv1d(x_i, conv, 1, 'SAME') for x_i, conv in zip(x, conv1d['convs3_wss'][idx])]
                x = [tf.squeeze(tf.layers.max_pooling1d(i, i.get_shape().as_list()[1],
                                                        i.get_shape().as_list()[1], 'SAME')) for i in
                     x]  # [(bs,kn), ...]*len(Ks)
                x = tf.concat(x, 1)
                x = tf.concat([x, self.feature_eng], 1)
                x = tf.matmul(x, weights['softmax_aspects'][idx]) + biases['softmax_aspects'][idx]
                self.represent_reps.append(x)

            self.predict = tf.concat(self.represent_reps, 1)
            self.predict_watch = self.predict
            self.predict = tf.reshape(self.predict, [-1, self.n_class, self.n_sub_class])
            self.predict_sm = tf.nn.softmax(self.predict, dim=-1)
        with tf.name_scope('loss'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.labels)
            self.cost = tf.reduce_sum(self.cost * self.cost_ws)
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer_ori = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = self.optimizer_ori.minimize(self.cost, global_step=self.global_step)
        writer = tf.summary.FileWriter('./tf_graph', tf.get_default_graph())
        writer.close()

    def cal_f1(self, labels, predicts):
        labels = np.argmax(labels, axis=-1)
        predicts = np.argmax(predicts, axis=-1)
        f1, f1_details = 0, []
        for col_idx in range(self.n_class):
            crt_f1 = f1_score(labels[:, col_idx], predicts[:, col_idx], average=None)
            crt_mean_f1 = np.mean(crt_f1)
            print('col_idx={}, mean_f1={}, f1={}'.format(col_idx, crt_mean_f1, crt_f1))
            f1_details.append(crt_mean_f1)
            f1 += crt_mean_f1
        return f1 / self.n_class, f1_details

    def train(self, data):
        aspects, contexts, labels, context_lens, cost_ws, feature_eng, aspects_ch, contexts_ch, aspects_ch_lens, contexts_ch_lens = data
        cost, cnt = 0., 0
        for sample, num in self.get_batch_data(aspects, contexts, labels, context_lens, cost_ws, feature_eng,
                                               aspects_ch, contexts_ch, aspects_ch_lens, contexts_ch_lens,
                                               self.batch_size, True, self.dropout_keep):
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict=sample)
            cost += loss * num
            cnt += num
        return cost / cnt, 0

    def test(self, data):
        aspects, contexts, labels, context_lens, cost_ws, feature_eng, aspects_ch, contexts_ch, aspects_ch_lens, contexts_ch_lens = data
        new_labels, new_predicts = [], []
        cost, cnt = 0., 0
        first = False
        for sample, num in self.get_batch_data(aspects, contexts, labels, context_lens, cost_ws, feature_eng,
                                               aspects_ch, contexts_ch, aspects_ch_lens, contexts_ch_lens,
                                               self.batch_size, False, 1.0):
            predict, labels, loss, predict_watch = self.sess.run(
                [self.predict_sm, self.labels, self.cost, self.predict_watch], feed_dict=sample)
            if first:
                first = False
                print(predict_watch[:1])
            cost += loss * num
            cnt += num
            new_labels += list(labels)
            new_predicts += list(predict)

        f1, f1_details = self.cal_f1(new_labels, new_predicts)
        return cost / cnt, f1, f1_details

    def predict_data(self, data):
        aspects, contexts, context_lens, cost_ws, feature_eng, aspects_ch, contexts_ch, aspects_ch_lens, contexts_ch_lens = data
        predicts = []
        for sample, num in self.get_batch_data(aspects, contexts, [], context_lens, cost_ws, feature_eng, aspects_ch,
                                               contexts_ch, aspects_ch_lens, contexts_ch_lens, self.batch_size, False,
                                               1.0):
            pred = self.sess.run([self.predict_sm], feed_dict=sample)[0].tolist()
            predicts += pred
        return predicts

    def run(self, train_data, test_data, model_suffix='model_iter', max_to_keep=7):
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=max_to_keep)
        print('Training(val_num=%d) ...' % (self.val_num))
        self.sess.run(tf.global_variables_initializer())
        max_f1, step, sub_step = 0., -1, -1
        crt_tol, all_val_details = 0, []
        for i in range(self.n_epoch):
            sub_val_cnt = 0
            while (len(train_data[0]) - sub_val_cnt * self.val_num > 3000):
                train_loss, train_acc = self.train(
                    [item[sub_val_cnt * self.val_num: min((sub_val_cnt + 1) * self.val_num, len(train_data[0]))] for
                     item in train_data])
                sub_val_cnt += 1
                test_loss, test_f1, test_f1_details = self.test(test_data)
                model_name = 'models/%s_iter_%d_%d' % (model_suffix, i, sub_val_cnt)
                all_val_details.append((model_name, test_f1, test_f1_details))
                saver.save(self.sess, model_name)
                print(
                    '>>>>>>>>>> epoch %s: train-loss=%.6f; train-f1=%.6f; test-loss=%.6f; test-f1=%.6f; <<<<<<<<<<' % (
                        '%d_%d' % (i, sub_val_cnt), train_loss, train_acc, test_loss, test_f1))
                if test_f1 > max_f1:
                    max_f1 = test_f1
                    step, sub_step, crt_tol = i, sub_val_cnt, 0
                else:
                    if (crt_tol < self.early_stop):
                        crt_tol += 1
                    else:
                        if step >= 0:
                            print('The max f1 of val data is %s of step %d_%d' % (max_f1, step, sub_step))
                            saver.restore(self.sess, 'models/%s_iter_%d_%d' % (model_suffix, step, sub_step))
                            pickle.dump(all_val_details, open('../data/f1_log_%s' % model_suffix, 'wb'))
                        else:
                            print('No best model!!!')
                        return all_val_details

    def get_batch_data(self, aspects, contexts, labels, context_lens, cost_ws, feature_eng, aspects_ch, contexts_ch,
                       aspects_ch_lens, contexts_ch_lens, batch_size, is_shuffle, keep_prob):
        total = int(len(context_lens) / batch_size) + 1 if len(context_lens) % batch_size != 0 else int(
            len(context_lens) / batch_size)
        for index in tqdm_notebook(get_batch_index(len(aspects), batch_size, is_shuffle), total=total):
            feed_dict = {
                self.aspects: aspects[index],
                self.contexts: contexts[index],
                self.context_lens: context_lens[index],
                self.feature_eng: feature_eng[index],
                self.cost_ws: cost_ws[index],
                self.dropout_keep_prob: keep_prob
            }
            if self.use_char_emb:
                feed_dict[self.aspects_ch] = aspects_ch[index]
                feed_dict[self.contexts_ch] = contexts_ch[index]
                feed_dict[self.aspects_ch_lens] = aspects_ch_lens[index].reshape(-1)
                feed_dict[self.contexts_ch_lens] = contexts_ch_lens[index]
            if len(labels) > 0: feed_dict[self.labels] = labels[index]
            if keep_prob < 1 and self.random_del_prob > 0:  # train mode
                cnt_per_row = int(self.random_del_prob * self.max_context_len)
                zero_rows = np.concatenate([np.ones(cnt_per_row).astype(int) * i for i in range(len(index))])
                zero_cols = np.random.choice(self.max_context_len, len(index) * cnt_per_row)
                if self.use_char_emb:
                    feed_dict[self.contexts_ch_lens][zero_rows, zero_cols] = 0
                # 这里不一定是0，可以试试别的
                feed_dict[self.contexts][zero_rows, zero_cols] = 0
            if self.use_char_emb:
                feed_dict[self.contexts_ch_lens] = feed_dict[self.contexts_ch_lens].reshape(-1)
            yield feed_dict, len(index)
