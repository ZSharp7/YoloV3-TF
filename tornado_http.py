import tornado.ioloop
import tornado.web
import tornado.options
import client_tfserving
import json
import requests
import os
tornado.options.define("port",default=8888,help='tornado run on port',type=int)
tornado.options.define("docker",default='127.0.0.1:8500',help='ip&port of run docker(tf-serving)',type=str)


# Handler
def download_img(url):
    name = url.split('/')[-1] # 文件名
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
    }
    response = requests.get(url,headers=headers)
    img = response.content
    with open('./temporary/images/%s'%name,'wb') as f:
        f.write(img)
    return './temporary/images/%s'%name

class CoorHandler(tornado.web.RequestHandler):
    service = client_tfserving.Client()
    def post(self):
        json_content = json.loads(self.request.body)
        adress = json_content['adress']
        is_url = json_content['is_url']
        if is_url:
            print('开始下载图片,下载地址:%s'%adress)
            try:
                adress = download_img(adress)
            except:
                print('下载失败.')
                raise ConnectionError
        self.finish(json.dumps(self.service.get_image(adress,True,ip=tornado.options.options.docker)))

class ArrayHandler(tornado.web.RequestHandler):
    service = client_tfserving.Client()
    def post(self):
        json_content = json.loads(self.request.body)
        adress = json_content['adress']
        is_url = json_content['is_url']
        if is_url:
            print('开始下载图片,下载地址:%s' % adress)
            try:
                adress = download_img(adress)
            except:
                print('%s下载失败.'%adress)
                raise ConnectionError
        self.finish(json.dumps(self.service.get_image(adress, False,ip=tornado.options.options.docker)))
# route
def make_app():
    return tornado.web.Application([
        (r"/model/coor_cls", CoorHandler),
        (r"/model/array_cls",ArrayHandler)
    ])

if __name__ == "__main__":

    app = make_app()
    app.listen(tornado.options.options.port)
    tornado.ioloop.IOLoop.current().start()



'''
输入图片(H,w,3)
return json:
{
    'cls_name1': # cls名
    {
        'len': 2, # 数量
        '1':[h,w,3], # 图片array
        '2':[h,w,3]  # 图片array
    },
    'cls_name2': # cls名
    {
        'len': 1, # 数量
        '1':[h,w,3]  # 图片array
    }
}
'''