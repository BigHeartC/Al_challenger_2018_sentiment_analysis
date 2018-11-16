import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
import random
import numpy as np
import pandas as pd
from utils import get_batch_index
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
from keras.layers import Bidirectional, CuDNNLSTM
import pickle
import tensorflow as tf


class SynAtt(object):
    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.embedding_dim_ch = config.embedding_dim_ch
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_sub_class = config.n_sub_class
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout_keep = config.dropout_keep
        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.embedding_matrix2 = config.embedding_matrix
        self.embedding_matrix_ch = config.embedding_matrix_ch
        self.early_stop = config.early_stop
        self.id2word = config.id2word
        self.val_num = config.val_num
        self.random_del_prob = config.random_del_prob
        self.feature_eng_size = config.feature_eng_size
        self.min_cos_sim = config.min_cos_sim
        self.encoder_t_dim = config.encoder_t_dim
        self.aspects_val = config.aspects_val
        self.aspect_lens_val = config.aspect_lens_val
        self.aspects_ch_val = config.aspects_ch_val
        self.aspects_ch_lens_val = config.aspects_ch_lens_val
        self.char_hidden_size = config.char_hidden_size
        self.max_char_len = config.max_char_len
        self.is_bi_rnn = config.is_bi_rnn
        self.sess = sess
        self.seed = 2018

    def build_model(self):
        with tf.name_scope('inputs'):
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class, self.n_sub_class])
            self.context_lens = tf.placeholder(tf.int32, None)
            self.cost_ws = tf.placeholder(tf.float32, None)
            self.feature_eng = tf.placeholder(tf.float32, [None, self.feature_eng_size])
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            self.aspects = tf.get_variable(name='aspects', initializer=self.aspects_val,
                                           dtype=tf.int32)  # n_class * max_aspect_len
            self.aspect_lens = tf.get_variable(name='aspect_lens', initializer=self.aspect_lens_val, dtype=tf.int32)

            self.embedding_matrix = tf.get_variable(name='embedding', initializer=self.embedding_matrix2,
                                                    dtype=tf.float32, trainable=True)
            self.embedding_matrix_ch = tf.get_variable(name='embedding_ch', initializer=self.embedding_matrix_ch,
                                                       trainable=True, dtype=tf.float32)
            self.aspects_ch = tf.get_variable(name='aspects_ch', initializer=self.aspects_ch_val,
                                              dtype=tf.int32)  # n_class * max_aspect_len * max_char_len
            self.aspects_ch_lens = tf.get_variable(name='aspects_ch_lens', initializer=self.aspects_ch_lens_val,
                                                   dtype=tf.int32)
            self.contexts_ch = tf.placeholder(tf.int32, [None, self.max_context_len, self.max_char_len])
            self.contexts_ch_lens = tf.placeholder(tf.int32, [None])

        with tf.name_scope('emb'):
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
                aspect_emb = tf.reshape(aspect_emb, [self.n_class, self.max_aspect_len, 2 * self.char_hidden_size])
            with tf.variable_scope("word"):
                aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.aspects)
                aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

                context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
                context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

            aspect_inputs = tf.concat([aspect_emb, aspect_inputs], axis=-1)
            context_inputs = tf.concat([context_emb, context_inputs], axis=-1)
            fnl_embedding_dim = self.embedding_dim + 2 * self.char_hidden_size

        hidden_times = 2 if self.is_bi_rnn else 1
        with tf.name_scope('weights'):
            weights = {
                'context_score_list': [tf.get_variable(
                    name='W_c_%d' % i,
                    shape=[self.n_hidden, fnl_embedding_dim],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'context_score_list2': [tf.get_variable(
                    name='W_c2_%d' % i,
                    shape=[self.n_hidden, fnl_embedding_dim],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_class * fnl_embedding_dim * hidden_times + self.feature_eng_size,
                           self.n_class * self.n_sub_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax0': tf.get_variable(
                    name='W_0',
                    shape=[self.n_class * fnl_embedding_dim * hidden_times + self.feature_eng_size, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'encoder_t': tf.get_variable(
                    name='encoder_t',
                    shape=[self.encoder_t_dim, fnl_embedding_dim],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                ),
                'encoder': tf.get_variable(
                    name='encoder_w',
                    shape=[fnl_embedding_dim, self.encoder_t_dim],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax_aspects': [tf.get_variable(
                    name='W_l_%d' % i,
                    shape=[fnl_embedding_dim + self.feature_eng_size, self.n_sub_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'softmax_fnl': tf.get_variable(
                    name='W_fnl',
                    shape=[self.n_sub_class * self.n_class, self.n_sub_class * self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'context_score_list': [tf.get_variable(
                    name='B_c_%d' % i,
                    shape=[1, self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'context_score_list2': [tf.get_variable(
                    name='B_c2_%d' % i,
                    shape=[1, self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class * self.n_sub_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax0': tf.get_variable(
                    name='B_0',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'encoder': tf.get_variable(
                    name='encoder_b',
                    shape=[self.encoder_t_dim],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax_aspects': [tf.get_variable(
                    name='B_l_%d' % i,
                    shape=[self.n_sub_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ) for i in range(self.n_class)],
            }

        with tf.name_scope('conv1d'):
            conv1d = {
                'convs1_wss': [tf.get_variable(
                    name='convs1_w%d' % k,
                    shape=[2, fnl_embedding_dim, fnl_embedding_dim],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
                               for k in range(self.n_class)],
            }

        with tf.name_scope('dynamic_rnn'):
            batch_size = tf.shape(context_inputs)[0]
            if self.is_bi_rnn:
                context_outputs = Bidirectional(CuDNNLSTM(self.n_hidden, return_sequences=True), merge_mode=None)(
                    context_inputs)
                context_outputs2 = context_outputs[1]
                context_outputs = context_outputs[0]
            else:
                context_outputs = CuDNNLSTM(self.n_hidden, return_sequences=True)(context_inputs)

            aspect_avg = tf.reduce_sum(aspect_inputs, 1) / tf.expand_dims(tf.cast(self.aspect_lens, tf.float32),
                                                                          dim=-1)  # n_class*K
            aspect_avg = tf.expand_dims(aspect_avg, 0)  # 1*n_class*K
            context_avg = tf.reduce_sum(context_inputs, 1) / tf.expand_dims(tf.cast(self.context_lens, tf.float32),
                                                                            dim=-1)  # bs*K
            context_avg = tf.expand_dims(context_avg, 1)  # bs*1*K
            c_s = (context_avg + aspect_avg) / 2  # bs*n_class*K
            q_t = tf.matmul(tf.reshape(c_s, (-1, fnl_embedding_dim)), weights['encoder']) + biases['encoder']
            self.aspect_reps = tf.matmul(q_t, weights['encoder_t'])
            self.aspect_reps = tf.reshape(self.aspect_reps, (-1, self.n_class, fnl_embedding_dim))  # bs*n_class*K
            all_aspects = tf.split(self.aspect_reps, self.n_class, 1)

            def body(i, max_len, len_iter, att_iter, fnl_att):
                l = len_iter.read(i)
                a = att_iter.read(i)
                crt_att = tf.concat([tf.nn.softmax(tf.slice(a, [0, 0], [1, l])), tf.zeros([1, max_len - l])], 1)
                fnl_att = fnl_att.write(i, crt_att)
                return (i + 1, max_len, len_iter, att_iter, fnl_att)

            def condition(i, max_len, len_iter, att_iter, fnl_att):
                return i < batch_size

            self.represent_reps = []
            for crt_idx, crt_aspect in enumerate(all_aspects):
                context_outputs_conv = tf.nn.tanh(
                    tf.nn.conv1d(context_outputs, conv1d['convs1_wss'][crt_idx], 1, 'SAME'))
                crt_aspect = tf.transpose(crt_aspect, [0, 2, 1])  # bs*K*1
                context_outputs_conv = tf.reshape(context_outputs_conv, (-1, self.n_hidden))  # (n*context_max_len) * K
                self.context_atts = tf.reshape(tf.matmul(context_outputs_conv, weights['context_score_list'][crt_idx]),
                                               (-1, self.max_context_len, self.n_hidden))  # n*context_max_len*K
                self.context_atts = tf.transpose(
                    tf.nn.relu(tf.matmul(self.context_atts, crt_aspect) + biases['context_score_list'][crt_idx]),
                    [0, 2, 1])  # bs*1*context_max_len
                _, _, _, _, self.context_atts = tf.while_loop(cond=condition, body=body,
                                                              loop_vars=(0, self.max_context_len,
                                                                         tf.TensorArray(tf.int32, 1, dynamic_size=True,
                                                                                        infer_shape=False).unstack(
                                                                             self.context_lens),
                                                                         tf.TensorArray(tf.float32, 1,
                                                                                        dynamic_size=True,
                                                                                        infer_shape=False).unstack(
                                                                             self.context_atts),
                                                                         tf.TensorArray(size=batch_size,
                                                                                        dtype=tf.float32)))
                self.context_atts = tf.transpose(self.context_atts.stack(), [0, 2, 1])  # bs*context_max_len*1
                context_outputs_conv = tf.reshape(context_outputs_conv, (-1, self.max_context_len, self.n_hidden))
                self.context_reps = tf.squeeze(
                    tf.matmul(tf.transpose(context_outputs_conv, [0, 2, 1]), self.context_atts), -1)  # bs*K
                self.context_reps = tf.concat([self.context_reps, self.feature_eng], 1)

                self.context_reps = tf.matmul(self.context_reps, weights['softmax_aspects'][crt_idx]) + \
                                    biases['softmax_aspects'][crt_idx]
                self.represent_reps.append(self.context_reps)

            self.predict = tf.concat(self.represent_reps, -1)  # bs*(n_class*K)

            if self.is_bi_rnn:
                self.represent_reps2 = []
                for crt_idx, crt_aspect in enumerate(all_aspects):
                    crt_aspect = tf.transpose(crt_aspect, [0, 2, 1])
                    context_outputs2 = tf.reshape(context_outputs2, (-1, self.n_hidden))  # (n*context_max_len) * K
                    self.context_atts = tf.reshape(tf.matmul(context_outputs2, weights['context_score_list2'][crt_idx]),
                                                   (-1, self.max_context_len, self.n_hidden))  # n*context_max_len*K
                    self.context_atts = tf.transpose(
                        tf.nn.relu(tf.matmul(self.context_atts, crt_aspect) + biases['context_score_list2'][crt_idx]),
                        [0, 2, 1])  # bs*1*context_max_len
                    _, _, _, _, self.context_atts = tf.while_loop(cond=condition, body=body,
                                                                  loop_vars=(0, self.max_context_len,
                                                                             tf.TensorArray(tf.int32, 1,
                                                                                            dynamic_size=True,
                                                                                            infer_shape=False).unstack(
                                                                                 self.context_lens),
                                                                             tf.TensorArray(tf.float32, 1,
                                                                                            dynamic_size=True,
                                                                                            infer_shape=False).unstack(
                                                                                 self.context_atts),
                                                                             tf.TensorArray(size=batch_size,
                                                                                            dtype=tf.float32)))
                    self.context_atts = tf.transpose(self.context_atts.stack(), [0, 2, 1])  # bs*context_max_len*1
                    context_outputs2 = tf.reshape(context_outputs2, (-1, self.max_context_len, self.n_hidden))
                    self.context_reps = tf.squeeze(
                        tf.matmul(tf.transpose(context_outputs2, [0, 2, 1]), self.context_atts), -1)  # bs*K
                    self.represent_reps2.append(self.context_reps)
                self.represent_reps2 = tf.concat(self.represent_reps2, -1)  # bs*(n_class*K)

            self.predict = tf.reshape(self.predict, [-1, self.n_class, self.n_sub_class])
            self.predict_sm = tf.nn.softmax(self.predict, dim=-1)

        with tf.name_scope('loss'):
            self.r_theta = tf.nn.l2_normalize(weights['encoder_t'], dim=1)
            self.r_theta = tf.reduce_sum(
                tf.square(tf.matmul(self.r_theta, tf.transpose(self.r_theta)) - tf.eye(self.encoder_t_dim)))
            self.cos_sim_dist = -tf.minimum(self.min_cos_sim, tf.losses.cosine_distance(c_s, self.aspect_reps, dim=0))
            self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.labels)
            self.cost = tf.reduce_sum(self.cost * self.cost_ws)
            self.cost = self.cost + 1 * self.r_theta + self.cos_sim_dist * 0.1

            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                               global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))
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
        contexts, labels, context_lens, cost_ws, feature_eng, contexts_ch, contexts_ch_lens = data
        cost, cnt = 0., 0
        for sample, num in self.get_batch_data(contexts, labels, context_lens, cost_ws, feature_eng, contexts_ch,
                                               contexts_ch_lens, self.batch_size, True, self.dropout_keep):
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict=sample)
            cost += loss * num
            cnt += num
        return cost / cnt, 0

    def test(self, data):
        contexts, labels, context_lens, cost_ws, feature_eng, contexts_ch, contexts_ch_lens = data
        new_labels, new_predicts = [], []
        cost, cnt = 0., 0
        first = False
        for sample, num in self.get_batch_data(contexts, labels, context_lens, cost_ws, feature_eng, contexts_ch,
                                               contexts_ch_lens, self.batch_size, False, 1.0):
            predict, labels, loss = self.sess.run([self.predict_sm, self.labels, self.cost], feed_dict=sample)
            if first:
                first = False
                print(predict[:3])
            cost += loss * num
            cnt += num
            new_labels += list(labels)
            new_predicts += list(predict)
        f1, f1_details = self.cal_f1(new_labels, new_predicts)
        return cost / cnt, f1, f1_details

    def predict_data(self, data):
        contexts, context_lens, cost_ws, feature_eng, contexts_ch, contexts_ch_lens = data
        predicts = []
        for sample, num in self.get_batch_data(contexts, [], context_lens, cost_ws, feature_eng, contexts_ch,
                                               contexts_ch_lens, self.batch_size, False, 1.0):
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

    def get_batch_data(self, contexts, labels, context_lens, cost_ws, feature_eng, contexts_ch, contexts_ch_lens,
                       batch_size, is_shuffle, keep_prob):
        total = int(len(context_lens) / batch_size) + 1 if len(context_lens) % batch_size != 0 else int(
            len(context_lens) / batch_size)
        for index in tqdm_notebook(get_batch_index(len(contexts), batch_size, is_shuffle), total=total):
            feed_dict = {
                self.contexts: contexts[index],
                self.context_lens: context_lens[index],
                self.feature_eng: feature_eng[index],
                self.cost_ws: cost_ws[index],
                self.dropout_keep_prob: keep_prob,
                self.contexts_ch: contexts_ch[index],
                self.contexts_ch_lens: contexts_ch_lens[index],
            }
            if len(labels) > 0: feed_dict[self.labels] = labels[index]
            if keep_prob < 1 and self.random_del_prob > 0:  # train mode
                cnt_per_row = int(self.random_del_prob * self.max_context_len)
                zero_rows = np.concatenate([np.ones(cnt_per_row).astype(int) * i for i in range(len(index))])
                zero_cols = np.random.choice(self.max_context_len, len(index) * cnt_per_row)
                feed_dict[self.contexts_ch_lens][zero_rows, zero_cols] = 0
                # 这里不一定是0，可以试试别的
                feed_dict[self.contexts][zero_rows, zero_cols] = 0
            feed_dict[self.contexts_ch_lens] = feed_dict[self.contexts_ch_lens].reshape(-1)
            yield feed_dict, len(index)
