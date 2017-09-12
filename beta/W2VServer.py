#!/usr/bin/env python
# -*- coding:utf-8 -*-
import ConfigParser
import os
import re
import sys

import tornado.ioloop
import tornado.web

import word2vec
from gensim.models import KeyedVectors

re_msg = re.compile(r'\+|\-')

#dir_gensim_model_xingshi = os.path.expanduser('~/fastText/w2vdata/xingshi/w2vgensim/')
#dir_gensim_model_minshi = os.path.expanduser('~/fastText/w2vdata/minshi/w2vgensim/')
#dir_google_model_xingshi = os.path.expanduser('~/fastText/w2vdata/xingshi/w2vgoogle/')
#dir_google_model_minshi = os.path.expanduser('~/fastText/w2vdata/minshi/w2vgoogle/')
#dir_glove_model_xingshi = os.path.expanduser('~/fastText/w2vdata/xingshi/glove/')
#dir_glove_model_minshi = os.path.expanduser('~/fastText/w2vdata/minshi/glove/')

dir_gensim_model_xingshi = os.path.expanduser('~/work/data/w2vdata/xingshi/w2vgensim/')
dir_gensim_model_minshi = os.path.expanduser('~/work/data/w2vdata/minshi/w2vgensim/')
dir_google_model_xingshi = os.path.expanduser('~/work/data/w2vdata/xingshi/w2vgoogle/')
dir_google_model_minshi = os.path.expanduser('~/work/data/w2vdata/minshi/w2vgoogle/')
dir_glove_model_xingshi = os.path.expanduser('~/work/data/w2vdata/xingshi/glove/')
dir_glove_model_minshi = os.path.expanduser('~/work/data/w2vdata/minshi/glove/')

static_duanluo = None
static_anyou = None

gensim_model = None
gensim_model_wv = None
glove_model = None
glove_model_wv = None
google_model = None

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


def split_pos_neg(msg):
    """
    :param msg:
    :return: pos, neg
    """
    pos = []
    neg = []
    split_msg = re_msg.split(msg)
    if len(split_msg) < 1:
        return pos, neg
    elif len(split_msg) == 1:
        pos.append(msg.strip(' \r\t\f\v').decode('utf-8'))
        return pos, neg

    pos.append(split_msg[0].strip(' \r\t\f\v').decode('utf-8'))
    msg_index = 0
    for i in range(len(split_msg) - 1):
        msg_index = msg.index(split_msg[i], msg_index) + len(split_msg[i])
        if msg[msg_index:msg.index(split_msg[i + 1], msg_index)] == '+':
            pos.append(split_msg[i + 1].strip(' \r\t\f\v').decode('utf-8'))
        # elif msg[(msg.index(split_msg[i]) + len(split_msg[i])):msg.index(split_msg[i + 1])] == '-':
        elif msg[msg_index:msg.index(split_msg[i + 1], msg_index)] == '-':
            neg.append(split_msg[i + 1].strip(' \r\t\f\v').decode('utf-8'))
        else:
            pos.append(split_msg[i + 1].strip(' \r\t\f\v').decode('utf-8'))

    return pos, neg


def load_w2v_gensim(duanluo, anyou):
    """
    :param duanluo:
    :param anyou_type:
    :return:
    """
    print 'load model...'
    file_w2v_model_xingshi = dir_gensim_model_xingshi + duanluo + '/W2VSz100SgNeg' + anyou + '.model.bin'
    if os.path.isfile(file_w2v_model_xingshi):
        return KeyedVectors.load_word2vec_format(file_w2v_model_xingshi, binary=True)

    file_w2v_model_minshi = dir_gensim_model_minshi + duanluo + '/W2VSz100SgNeg' + anyou + '.model.bin'
    if os.path.isfile(file_w2v_model_minshi):
        return KeyedVectors.load_word2vec_format(file_w2v_model_minshi, binary=True)

    return None


def load_glove(duanluo, anyou):
    """
    :param duanluo:
    :param anyou_type:
    :return:
    """
    print 'load model...'
    file_glove_model_xingshi = dir_glove_model_xingshi + duanluo + '/W2VSz100SgNeg' + anyou + '.txt'
    if os.path.isfile(file_glove_model_xingshi):
        return KeyedVectors.load_word2vec_format(file_glove_model_xingshi, binary=False)

    file_glove_model_minshi = dir_glove_model_minshi + duanluo + '/W2VSz100SgNeg' + anyou + '.txt'
    if os.path.isfile(file_glove_model_minshi):
        return KeyedVectors.load_word2vec_format(file_glove_model_minshi, binary=False)

    return None


def load_w2v_google(duanluo, anyou):
    """
    :param duanluo:
    :param anyou_type:
    :return:
    """
    print 'load model...'
    file_w2v_model_xingshi = dir_google_model_xingshi + duanluo + '/W2VSz100SgNeg' + anyou + '.bin'
    if os.path.isfile(file_w2v_model_xingshi):
        return word2vec.load(file_w2v_model_xingshi)

    file_w2v_model_minshi = dir_google_model_minshi + duanluo + '/W2VSz100SgNeg' + anyou + '.bin'
    if os.path.isfile(file_w2v_model_minshi):
        return word2vec.load(file_w2v_model_minshi)

    return None


def get_relate_words(message, method, num=10, duanluo_type='', anyou_type=''):
    """
    :param message:
    :param method:
    :param num:
    :param duanluo_tyoe:
    :param anyou_type:
    :return:
    """
    global gensim_model
    global glove_model
    global google_model

    global static_duanluo
    global static_anyou

    res_out = ''
    res_simi = []

    print message, method

    msg_pos, msg_neg = split_pos_neg(message)

    print 'pos:'
    for one in msg_pos:
        print one
    print 'neg:'
    for one in msg_neg:
        print one
    print '.'

    if method == 'google-':
        if static_duanluo != duanluo_type or static_anyou != anyou_type or google_model is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            google_model = load_w2v_google(duanluo_type, anyou_type)
        if google_model is None:
            return ''

        if len(msg_pos) == 1 and len(msg_neg) == 0 and msg_pos[0] in google_model.vocab:
            indexes, metrics = google_model.cosine(msg_pos[0].decode('utf-8'), n=num)
            res_simi = list(google_model.generate_response(indexes, metrics))
        else:
            print 'Delete:'
            for para in msg_pos:
                if para not in google_model.vocab:
                    msg_pos.remove(para)
                    print para
            for para in msg_neg:
                if para not in google_model.vocab:
                    msg_neg.remove(para)
                    print para
            print '.'

            if len(msg_pos) + len(msg_neg) >= 1:
                indexes, metrics = google_model.analogy(pos=msg_pos, neg=msg_neg, n=num)
                res_simi = list(google_model.generate_response(indexes, metrics))

        for i in range(len(res_simi)):
            if i == len(res_simi) - 1:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1])
            else:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1]) + ';'

    elif method == 'fasttext-':
        pass
    elif method == 'glove-':
        if static_duanluo != duanluo_type or static_anyou != anyou_type or glove_model is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            glove_model = load_glove(duanluo_type, anyou_type)
        if glove_model is None:
            return ''

        if len(msg_pos) == 1 and len(msg_neg) == 0 and msg_pos[0] in glove_model.vocab:
            res_simi = glove_model.similar_by_word(msg_pos[0].decode('utf-8'), topn=num)
        else:
            print 'Delete:'
            for para in msg_pos:
                if para not in glove_model.vocab:
                    msg_pos.remove(para)
                    print para
            for para in msg_neg:
                if para not in glove_model.vocab:
                    msg_neg.remove(para)
                    print para
            print '.'

            if len(msg_pos) + len(msg_neg) >= 1:
                res_simi = glove_model.most_similar(positive=msg_pos, negative=msg_neg, topn=num)

        for i in range(len(res_simi)):
            if i == len(res_simi) - 1:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1])
            else:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1]) + ';'
    else:# default-method:'gensim'
        if static_duanluo != duanluo_type or static_anyou != anyou_type or gensim_model is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            gensim_model = load_w2v_gensim(duanluo_type, anyou_type)
        if gensim_model is None:
            return ''

        if len(msg_pos) == 1 and len(msg_neg) == 0 and msg_pos[0] in gensim_model.vocab:
            # res_simi = gensim_model.most_similar([msg_pos[0].decode('utf-8')], topn=num)
            res_simi = gensim_model.similar_by_word(msg_pos[0].decode('utf-8'), topn=num)
        else:
            print 'Delete:'
            for para in msg_pos:
                if para not in gensim_model.vocab:
                    msg_pos.remove(para)
                    print para
            for para in msg_neg:
                if para not in gensim_model.vocab:
                    msg_neg.remove(para)
                    print para
            print '.'

            if len(msg_pos) + len(msg_neg) >= 1:
                res_simi = gensim_model.most_similar(positive=msg_pos, negative=msg_neg, topn=num)

        for i in range(len(res_simi)):
            if i == len(res_simi) - 1:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1])
            else:
                res_out += str(res_simi[i][0].encode('utf-8')) + ':' + str(res_simi[i][1]) + ';'

    return res_out


def get_word_vec(message, method='', duanluo_type='', anyou_type=''):
    """
    :param message:
    :param method:
    :param duanluo_type:
    :param anyou_type:
    :return:
    """
    global gensim_model_wv
    global glove_model_wv
    global google_model_wv

    global static_duanluo
    global static_anyou

    res_out = ''

    if method == 'google-':
        if static_duanluo != duanluo_type or static_anyou != anyou_type or google_model_wv is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            google_model_wv = load_w2v_google(duanluo_type, anyou_type)
        if google_model_wv is None:
            return ''

        if message in google_model_wv.vocab:
            res_out = str(google_model_wv.get_vector(message))

    elif method == 'fasttext-':
        pass
    elif method == 'glove-':
        if static_duanluo != duanluo_type or static_anyou != anyou_type or glove_model_wv is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            glove_model_wv = load_glove(duanluo_type, anyou_type)
        if glove_model_wv is None:
            return ''

        if message in glove_model_wv.vocab:
            res_out = str(glove_model_wv.word_vec(message))
    else:  # default-method: 'gensim'
        if static_duanluo != duanluo_type or static_anyou != anyou_type or gensim_model_wv is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            gensim_model_wv = load_w2v_gensim(duanluo_type, anyou_type)
        if gensim_model_wv is None:
            return ''

        if message in gensim_model_wv.vocab:
            res_out = str(gensim_model_wv.word_vec(message))

    return res_out


class AnalogyWordsHandler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        message = self.get_argument('message')
        method = self.get_argument('method')
        simi_num = int(self.get_argument('siminum'))
        duanluo_type = self.get_argument('duanluo')
        anyou_type = self.get_argument('anyou')
        print '---link server---'
        result = get_relate_words(message, method, simi_num, duanluo_type, anyou_type)
        print 'finish compute'
        self.write(str(result))


class WordVector(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/wordVector" method="post">'
                   '<p>请输入单词</p>'
                   '<textarea name="message" style="width:100px;height50px;">'
                   """"""
                   '</textarea>'
                   '<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>'
                   '<span for="select_model" class="select_model_label">w2vModel</span>'
                   '<select name="model_type" id="select_model">'
                   '<option value="gensim">gensim</option>'
                   '<option value="google">google</option>'
                   '<option disabled="disabled" value="fasttext">fasttext</option>'
                   '<option disabled="disabled" value="glove">glove</option>'
                   '</select>'
                   '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                   '<span for="select_duanluo" class="select_duanluo_label">段落</span>'
                   '<select name="duanluo_type" id="select_duanluo">'
                   '<option value="ssjl">诉讼记录</option>'
                   '<option value="ajjbqk">案件基本情况</option>'
                   '<option value="cpfxgc">裁判分析过程</option>'
                   '<option value="pjjg">判决结果</option>'
                   '</select>'
                   '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                   '<span>案由</span>'
                   '<input id="anyou_type" type="text" name="anyou_type" class="inputbox"'
                   'placeholder="请输入案由类型" style="width:120px;"/>'
                   '<br><br> <input type="submit" value="Submit">'
                   '</form>'
                   '</body></html>')

    def post(self):
        self.set_header("Content-Type", "text/html")
        message = self.get_argument('message')
        method = self.get_argument('model_type')
        duanluo_type = self.get_argument('duanluo_type')
        anyou_type = self.get_argument('anyou_type')
        print '---link server---'
        result = get_word_vec(message, method, duanluo_type, anyou_type)

        if not result:
            result = 'No exist the word'

        self.write('<!DOCTYPE html>'
                   '<html><head>'
                   '<meta http-equiv="content-type" content="text/html;charset=utf-8">')
        self.write("</head>")
        self.write("<body>")
        self.write("<div>")
        self.write(str(result))
        self.write("</div>")
        self.write("</body></html>")


if __name__ == "__main__":
    print '---Start ComputSimiWord Service---'
    cp = ConfigParser.SafeConfigParser()
    cp.read('w2vservice.conf')
    server_port = cp.get('service', 'port')

    application = tornado.web.Application([
        (r'/analogyWordsDemo', AnalogyWordsHandler),
        (r'/wordVector', WordVector)
    ])

    application.listen(server_port)
    tornado.ioloop.IOLoop.instance().start()
