#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
import time
import sys
import threading
import subprocess
import tornado.ioloop
import tornado.web

import config
import word2vec
from gensim.models import KeyedVectors

re_msg = re.compile(r'\+|\-')

dir_w2v_model_xingshi = os.path.expanduser('~/work/data/w2vdata/xingshi/w2vgensim/')
dir_w2v_model_minshi = os.path.expanduser('~/work/data/w2vdata/minshi/w2vgensim/')
dir_w2v_model_google_xingshi = os.path.expanduser('~/work/data/w2vdata/xingshi/w2vgoogle/')
dir_w2v_model_google_minshi = os.path.expanduser('~/work/data/w2vdata/minshi/w2vgoogle/')

static_duanluo = ''
static_anyou = ''
gensim_model = None
google_model = None

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


def checkIP(host):
    """
    :param host:
    :return:
    """
    global host_alive
    while True:
        p = subprocess.Popen(['ping -c 1 ' + host],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True)
        out = p.stdout.read()
        regex = re.compile('time=\d*', re.IGNORECASE | re.MULTILINE)
        if len(regex.findall(out)) > 0:
            host_alive[host] = True
            print time.strftime('%Y/%m/%d %H:%M:%S  ') + host + ': Host Up!'
        else:
            host_alive[host] = False
            print time.strftime('%Y/%m/%d %H:%M:%S  ') + host + ': Host Down!'
        time.sleep(2)


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
    print '-----: load model.'
    file_w2v_model_xingshi = dir_w2v_model_xingshi + duanluo + '/W2VSz100SgNeg' + anyou + '.model.bin'
    if os.path.isfile(file_w2v_model_xingshi):
        return KeyedVectors.load_word2vec_format(file_w2v_model_xingshi, binary=True)

    file_w2v_model_minshi = dir_w2v_model_minshi + duanluo + '/W2VSz100SgNeg' + anyou + '.model.bin'
    if os.path.isfile(file_w2v_model_minshi):
        return KeyedVectors.load_word2vec_format(file_w2v_model_minshi, binary=True)

    print 'No anyou'
    return None


def load_w2v_google(duanluo, anyou):
    """
    :param duanluo:
    :param anyou_type:
    :return:
    """
    file_w2v_model_xingshi = dir_w2v_model_google_xingshi + duanluo + '/W2VSz100SgNeg' + anyou + '.bin'
    if os.path.isfile(file_w2v_model_xingshi):
        return word2vec.load(file_w2v_model_xingshi)

    file_w2v_model_minshi = dir_w2v_model_google_minshi + duanluo + '/W2VSz100SgNeg' + anyou + '.bin'
    if os.path.isfile(file_w2v_model_minshi):
        return word2vec.load(file_w2v_model_minshi)
    
    print 'No anyou'
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
    # global gensim_model_global
    # global google_model_global
    global gensim_model
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
        # msg_split = message.split(' ')
        # if len(msg_split) == 3 and msg_split[0] in google_model.vocab and \
        #                 msg_split[1] in google_model.vocab and msg_split[2] in google_model.vocab:
        #     indexes, metrics = google_model.analogy(
        #         pos=[msg_split[0].decode('utf-8'), msg_split[2].decode('utf-8')], neg=[msg_split[1].decode('utf-8')])
        #     res_simi = list(google_model.generate_response(indexes, metrics))
        # elif message in google_model.vocab:
        #     indexes, metrics = google_model.cosine(message.decode('utf-8'))
        #     res_simi = list(google_model.generate_response(indexes, metrics))

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
    # elif method == 'gensim':
    else:
        if static_duanluo != duanluo_type or static_anyou != anyou_type or gensim_model is None:
            static_duanluo = duanluo_type
            static_anyou = anyou_type
            gensim_model = load_w2v_gensim(duanluo_type, anyou_type)
        if gensim_model is None:
            return ''
        # msg_split = message.split(' ')
        # if len(msg_split) == 3 and msg_split[0] in gensim_model.vocab and \
        #                 msg_split[1] in gensim_model.vocab and msg_split[2] in gensim_model.vocab:
        #     res_simi = gensim_model.most_similar(positive=[msg_split[0].decode('utf-8'), msg_split[2].decode('utf-8')],
        #                                          negative=[msg_split[1].decode('utf-8')])
        # elif message in gensim_model.vocab:
        #     res_simi = gensim_model.most_similar([message.decode('utf-8')])

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


class AnalogyWordsHandler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        message = self.get_argument('message')
        method = self.get_argument('method')
        simi_num = int(self.get_argument('siminum'))
        duanluo_type = self.get_argument('duanluo')
        anyou_type = self.get_argument('anyou')
        print '---link---'
        result = get_relate_words(message, method, simi_num, duanluo_type, anyou_type)
        self.write(str(result))


application = tornado.web.Application([
    (r'/analogyWordsDemo', AnalogyWordsHandler)
])

threads = []
t1 = threading.Thread(target=checkIP, args=(config.remoteHost,))
threads.append(t1)
host_alive = {}

if __name__ == "__main__":

    # gensim_model_global = KeyedVectors.load_word2vec_format(w2v_model_gensim, binary=True)
    # google_model_global = word2vec.load(w2v_model_google)

    for t in threads:
        t.setDaemon(True)
        t.start()

    application.listen(config.servicePort)
    tornado.ioloop.IOLoop.instance().start()
