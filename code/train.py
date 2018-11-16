import pandas as pd
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from utils import *
import os
from gensim.models import Word2Vec

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df_train_data = pd.read_csv(
    '../data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')
df_val_data = pd.read_csv(
    '../data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv')
df_test_data = pd.read_csv('../data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv')
print(df_train_data.shape, df_val_data.shape, df_test_data.shape)

config = tf.contrib.training.HParams(
    batch_size=512,
    n_epoch=100,
    n_hidden=300,
    n_class=20,
    n_sub_class=4,
    learning_rate=0.001,
    l2_reg=0.001,
    dropout_keep=0.5,
    max_context_len=500,
    embedding_matrix=None,
    early_stop=1,
    id2word=None,
    val_num=int(len(df_train_data) / 2),
    kernel_num=100,
    kernel_sizes=[3]
)
# 定义 aspect 关键词
# 这些关键词由20个二分类的 LightGBM 根据特征重要性得到
subjects = ['地铁站 地铁 地理位置 位置 公交车 公交车站 公交站',
            '百货 商圈 商场 广场 购物中心 城 商业街',
            '容易 位置 醒目 找到 找 地理位置 显眼',
            '小时 排队 等 排 排号 队 号',
            '态度 服务员 热情 服务态度 老板 服务 服务生',
            '开车 停车费 停车位 停 停车场 车位 泊车',
            '很快 催 慢 速度 分钟 上菜 等',
            '小贵 不贵 价位 原价 块钱 价格 性价比',
            '不划算 物有所值 不值 物美价廉 超值 性价比 实惠',
            '活动 团 霸王餐 代金券 团购 优惠 券',
            '装修 布置 灯光 古色古香 装饰 优雅 情调',
            '安静 环境 装修 氛围 嘈杂 吵闹 音乐',
            '大 宽敞 空间 面积 装修 拥挤 店面',
            '整洁 干净 环境 卫生 苍蝇 不错 脏',
            '吃不完 一份 量 量足 个头 好大 少',
            '入味 吃 不错 味道 好吃 口味 好喝',
            '造型 颜色 精致 卖相 好看 色香味 食欲',
            '推荐 强烈推荐 值得 强推 一试 极力推荐 菜品',
            '好 满意 纪念品 内地 之 肠 灌',
            '还会 机会 再 不会 来 值得 推荐']

subjects_eng = ['location_traffic_convenience',
                'location_distance_from_business_district', 'location_easy_to_find',
                'service_wait_time', 'service_waiters_attitude',
                'service_parking_convenience', 'service_serving_speed', 'price_level',
                'price_cost_effective', 'price_discount', 'environment_decoration',
                'environment_noise', 'environment_space', 'environment_cleaness',
                'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                'others_overall_experience', 'others_willing_to_consume_again']
subjects_dict = OrderedDict(zip(subjects_eng, subjects))
config.max_aspect_len = len(subjects[0].split(' '))

print('building word vector...')
print('构造 word2id 映射')
config.w2v_path, config.w2v_word2id_txt = '../data/all_content_no_punc_100_8_mc2_fnl.w2v', 'word2id_map_mc2_fnl'
w2v = Word2Vec.load(config.w2v_path)
print(len(w2v.wv.vocab))
word2id, max_context_len, max_aspect_len = get_word2id('../data/', subjects,
                                                       'ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset',
                                                       'ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset',
                                                       'ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa',
                                                       w2v, pre_processed=False, save_fname=config.w2v_word2id_txt,
                                                       suffix='_cut_word_rst.txt')
print(len(word2id), max_context_len, max_aspect_len)
config.max_context_len, config.max_aspect_len = max_context_len, max_aspect_len

print('对评论编码')
train_context, train_context_lens = build_nn_context('../data/',
                                                     'ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset',
                                                     word2id, pre_processed=False,
                                                     context_max_len=config.max_context_len,
                                                     suffix='_cut_word_rst.txt')
val_context, val_context_lens = build_nn_context('../data/',
                                                 'ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset',
                                                 word2id, pre_processed=False, context_max_len=config.max_context_len,
                                                 suffix='_cut_word_rst.txt')
test_context, test_context_lens = build_nn_context('../data/',
                                                   'ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa',
                                                   word2id, pre_processed=False, context_max_len=config.max_context_len,
                                                   suffix='_cut_word_rst.txt')
print(train_context.shape, val_context.shape, test_context.shape)

print('计算不同类别、标签的权重')
cost_w = []
for col in subjects_eng:
    crt_w = [item[1] for item in sorted(dict(df_train_data[col].value_counts()).items(), key=lambda x: x[0])]
    crt_w = np.exp(1 * np.min(crt_w) / np.asarray(crt_w))
    crt_w /= sum(crt_w)
    cost_w.append(crt_w)
print(np.asarray(cost_w))

print('构造每个sample的label，权重')
train_labels, train_ws = build_labels_ws('../data/',
                                         'ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset',
                                         subjects_eng, cost_w)
val_labels, val_ws = build_labels_ws('../data/',
                                     'ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset',
                                     subjects_eng, cost_w)
print(train_labels.shape, train_ws.shape, val_labels.shape, val_ws.shape)

print('根据word2id构造对应的词向量')
config.embedding_matrix, config.embedding_dim = load_word_embeddings(word2id, w2v)
config.embedding_matrix = config.embedding_matrix.astype(np.float32)
print(config.embedding_matrix.shape)

print('对aspect编码')
aspect_input, aspect_lens = build_aspect(word2id, subjects, config.max_aspect_len)
print(aspect_input.shape, aspect_lens.shape)

print('building char vector...')
config.max_char_len = 7
config.ch2v_path, config.ch2v_char2id_txt = '../data/all_char_no_punc_100_15_mc2_fnl.w2v', 'char2id_map_mc2_fnl'

print('构造 char2id 映射')
ch2v = Word2Vec.load(config.ch2v_path)
print(len(ch2v.wv.vocab))
char2id, max_context_len_ch, max_aspect_len_ch = get_char2id('../data/', subjects,
                                                             'ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset',
                                                             'ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset',
                                                             'ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa',
                                                             ch2v, pre_processed=False,
                                                             save_fname=config.ch2v_char2id_txt,
                                                             suffix='_cut_char_rst.txt')
print(len(char2id), max_context_len_ch, max_aspect_len_ch)

print('将每个词拆分为字并记录')
wid2char = {}
for word, w_id in word2id.items():
    add_item = []
    if len(word) > config.max_char_len:
        add_item = (
            np.asarray([char2id[c] if c in char2id else 0 for c in list(word)[:config.max_char_len]]),
            config.max_char_len)
    else:
        add_item = (
            np.asarray(
                [char2id[c] if c in char2id else 0 for c in list(word)] + [0] * (config.max_char_len - len(word))),
            len(word))
    wid2char[w_id] = add_item
wid2char[0] = (np.zeros(config.max_char_len), 0)

print('对 aspect、评论 进行字编码')
aspect_input_ch = np.asarray([[wid2char[i][0] for i in asp] for asp in aspect_input])
aspect_input_ch_lens = np.asarray([[wid2char[i][1] for i in asp] for asp in aspect_input])
print(aspect_input_ch.shape, aspect_input_ch_lens.shape)

train_context_ch = np.asarray([[wid2char[i][0] for i in asp] for asp in train_context])
train_context_ch_len = np.asarray([[wid2char[i][1] for i in asp] for asp in train_context])
print(train_context_ch.shape, train_context_ch_len.shape)

val_context_ch = np.asarray([[wid2char[i][0] for i in asp] for asp in val_context])
val_context_ch_len = np.asarray([[wid2char[i][1] for i in asp] for asp in val_context])
print(val_context_ch.shape, val_context_ch_len.shape)

test_context_ch = np.asarray([[wid2char[i][0] for i in asp] for asp in test_context])
test_context_ch_len = np.asarray([[wid2char[i][1] for i in asp] for asp in test_context])
print(test_context_ch.shape, test_context_ch_len.shape)

print('根据 char2id 构造对应的字向量')
config.embedding_matrix_ch, config.embedding_dim_ch = load_word_embeddings(char2id, ch2v)
config.embedding_matrix_ch = config.embedding_matrix_ch.astype(np.float32)
print(config.embedding_matrix_ch.shape)

print('loading feature...')
prefix = 'svd_tfidf_withP_80'
train_feas = np.load('../data/%s_train.npy' % prefix)
val_feas = np.load('../data/%s_val.npy' % prefix)
test_feas = np.load('../data/%s_test.npy' % prefix)
config.feature_eng_size = train_feas.shape[1]
print(config.feature_eng_size)


def train_GCAE():
    train_aspect = np.tile(aspect_input, [len(train_context), 1, 1])
    val_aspect = np.tile(aspect_input, [len(val_context), 1, 1])

    train_aspect_ch = np.tile(aspect_input_ch, [len(train_context_ch), 1, 1, 1])
    val_aspect_ch = np.tile(aspect_input_ch, [len(val_context_ch), 1, 1, 1])

    train_aspect_input_ch_lens = np.tile(aspect_input_ch_lens, [len(train_context_ch), 1, 1])
    val_aspect_input_ch_lens = np.tile(aspect_input_ch_lens, [len(train_context_ch), 1, 1])

    config.batch_size = 32
    config.n_class = 20
    config.use_char_emb = True

    config.kernel_sizes = [3]
    config.kernel_num = 100
    config.random_del_prob = 0.08
    config.char_hidden_size = 64
    config.n_hidden = 300
    config.dropout_keep = 0.5

    config.val_num = int(len(train_context) / 2)
    config.early_stop = int(len(train_context) / config.val_num)
    print(config.val_num, config.early_stop)
    model_name = 'GCAE_expand_wc_%d_dim%d' % (config.max_context_len,
                                              config.embedding_dim)
    print(model_name)
    from GCAE_word_char import GCAE_expand_model

    # model.sess.close()
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    model = GCAE_expand_model(config, tf.Session(config=tf_config))
    model.build_model()

    score_detail = model.run(
        [train_aspect, train_context, train_labels, train_context_lens, train_ws, train_feas, train_aspect_ch,
         train_context_ch, train_aspect_input_ch_lens, train_context_ch_len],
        [val_aspect, val_context, val_labels, val_context_lens, val_ws, val_feas, val_aspect_ch, val_context_ch,
         val_aspect_input_ch_lens, val_context_ch_len],
        model_suffix=model_name)

    print('模型内集成')
    preds = get_model_predict(model, score_detail,
                              [val_aspect, val_context, val_context_lens, val_ws, val_feas, val_aspect_ch,
                               val_context_ch, val_aspect_input_ch_lens, val_context_ch_len])
    df_ensemble, df_ensemble_val = ensemble_a_model(preds, score_detail, subjects_eng, factor=200, max_save_num=5,
                                                    ensemble_aspects=True)
    local_f1 = cal_local_f1(df_val_data, df_ensemble, subjects_eng)


def train_syn_att():
    config.aspects_val = aspect_input.astype(np.int32)
    config.aspect_lens_val = aspect_lens.astype(np.int32)
    config.aspects_ch_val = aspect_input_ch.astype(np.int32)
    config.aspects_ch_lens_val = aspect_input_ch_lens.astype(np.int32).reshape([-1])

    # 2h 39m 18s
    config.batch_size = 32
    config.val_num = int(len(df_train_data))
    config.n_class = 20

    config.learning_rate = 0.001
    config.min_cos_sim = 10 ** -7
    config.encoder_t_dim = 20
    config.is_bi_rnn = False
    config.dropout_keep = 0.5

    config.random_del_prob = 0.16
    config.char_hidden_size = 50

    config.n_hidden = config.embedding_dim + 2 * config.char_hidden_size
    config.val_num = int(len(train_context) / 3)
    config.early_stop = int(len(train_context) / config.val_num)
    print(config.val_num, config.early_stop)

    model_name = 'SYN_expand_wc1_%d_dim%d' % (config.max_context_len,
                                              config.embedding_dim)

    from SynAtt_expand_model import SynAtt

    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    model = SynAtt(config, tf.Session(config=tf_config))
    model.build_model()

    score_detail = model.run(
        [train_context, train_labels, train_context_lens, train_ws, train_feas, train_context_ch, train_context_ch_len]
        , [val_context, val_labels, val_context_lens, val_ws, val_feas, val_context_ch, val_context_ch_len],
        model_suffix=model_name)

    print('模型内集成')
    preds = get_model_predict(model, score_detail,
                              [val_context, val_context_lens, val_ws, val_feas, val_context_ch, val_context_ch_len])
    df_ensemble, df_ensemble_val = ensemble_a_model(preds, score_detail, subjects_eng, factor=150, max_save_num=5,
                                                    ensemble_aspects=True)
    local_f1 = cal_local_f1(df_val_data, df_ensemble, subjects_eng)


import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('run GCAE model ')
        train_GCAE()
    elif sys.argv[1] == 'GCAE':
        train_GCAE()
    elif sys.argv[1] == 'SYN_ATT':
        train_syn_att()
    else:
        print('not found model, model must be GCAE or SYN_ATT')
