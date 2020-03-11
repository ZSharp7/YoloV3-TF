import json
import requests


def post(image_path,is_url,server_path):
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
    }
    request_body = {'adress':image_path,'is_url':is_url}
    body = json.dumps(request_body)
    content = requests.post(server_path,body,headers=headers)
    return content.json()
if __name__ == '__main__':
    print(post(image_path ='https://dss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=1208538952,1443328523&fm=26&gp=0.jpg',\
               server_path='http://127.0.0.1:8888/model/array_cls',\
               is_url=True))

