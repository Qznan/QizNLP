#!/usr/bin/env python
# coding=utf-8
"""
@Author : yonas
@Time   : 2020/3/15 下午10:55
@File   : web_API.py
"""
import os, time, json, sys
from run.run_cls import Run_Model_Cls
import tornado.ioloop
import tornado.web
from urllib import parse  # parse.quote 文本->url编码 ' '->'%20'  parse.unquote url编码->文本 '%20'->' '

curr_dir = os.path.dirname(os.path.realpath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

rm_cls = Run_Model_Cls('trans_mhattnpool')
rm_cls.restore('trans_ckpt_10')  # restore to infer


def web_predict(sent, need_cut=True):
    time0 = time.time()
    pred = rm_cls.predict([sent], need_cut=need_cut)[0]
    print('elapsed:', time.time() - time0)
    return pred


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        sent = self.get_argument('sent')
        sent = parse.unquote(sent)  # 你好%20嗯嗯 -> 你好 嗯嗯

        print('sent:', sent)
        ret = web_predict(sent)
        ret = {'result': ret}
        ret_str = json.dumps(ret, ensure_ascii=False)
        print(ret_str)
        self.write(ret_str)
        sys.stdout.flush()

    def post(self):
        body_data = self.request.body
        if isinstance(body_data, bytes):
            body_data = body_data.decode('utf-8')
        args_data = json.loads(body_data)
        sent = args_data.get('sent', None)
        print('sent:', sent)
        if sent is None:
            return
        ret = web_predict(sent)
        ret = {'result': ret}
        ret_str = json.dumps(ret, ensure_ascii=False)
        print(ret_str)
        self.write(ret_str)
        sys.stdout.flush()


def make_app():
    return tornado.web.Application([
        (r"/QZ/predict", MainHandler),
    ])


if __name__ == '__main__':
    DEFAULT_PORT = 8090
    app = make_app()
    app.listen(DEFAULT_PORT)
    tornado.ioloop.IOLoop.current().start()

    # curl localhost:8090/QZ/predict?sent=你好
