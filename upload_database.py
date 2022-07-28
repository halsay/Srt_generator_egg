import requests
import os
import subprocess
import ast
import re


def upload_to_database_dev(info):
     # upload cover
     file_path = info['cover']
     if os.path.exists(file_path):
          command = f'curl -H "Authorization: Bearer 123" -F file=@{file_path} https://api-dev.zimu.live/files/image'
          p = subprocess.Popen(command,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT 
                              )
          res = p.communicate()[0].decode('utf-8')
          print(res)
          fn = res.split('\n')[-1].split('"')[-2]
          print(fn)

          # upload information
          url = 'https://api-dev.zimu.live/clips'
          headers = {
               'Authorization': 'Bearer 123',
               'Content-Type': 'application/json'
          }
          if '"' in info["title"]:
               a = re.findall('\"\w+\"', info["title"])
               if len(a) == 1:
                    a = '“' + a[0][1:-1] + '”'
                    info["title"] = re.sub('\"\w+\"', a, info["title"])
          if len(info["title"]) > 30:
               info["title"] = info["title"][:30]
               # print(a)
          data_str = {
               "filename": fn,
               "authorId": info["authorId"],
               "datetime": info["datetime"],
               "title": info["title"],
               "bv": info["bv"]
          }
          print(str(data_str).replace("'",'"'), len(info["title"]))
          resp = requests.post(url=url, verify=False, data=str(data_str).replace("'",'"').encode('utf-8'), headers=headers)
          print(resp, resp.text)
          res_dict = ast.literal_eval(resp.text)
          if str(resp.status_code) == '400':
               clip_id = res_dict['message'].split(':')[-1]
          else:
               clip_id = res_dict['id']
          url = f'https://api-dev.zimu.live/clips/{clip_id}/subtitles'
          headers = {
               'Authorization': 'Bearer 123',
               'Content-Type': 'text/plain'
          }
          srt_path = info['srt']
          data = open(srt_path, 'r', encoding='utf-8').read().rstrip()
          resp = requests.post(url=url, verify=False, data=data.encode('utf-8'), headers=headers)
          print(resp, resp.text)
     else:
          print('video cover does not exist!')


def upload_to_database_main(info):
     # upload cover
     file_path = info['cover']
     if os.path.exists(file_path):
          command = f'curl -H "Authorization: Bearer Y2vD0P7vOJ" -F file=@{file_path} https://api.zimu.live/files/image'
          p = subprocess.Popen(command,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT 
                              )
          res = p.communicate()[0].decode('utf-8')
          print(res)
          fn = res.split('\n')[-1].split('"')[-2]
          print(fn)

          # upload information
          url = 'https://api.zimu.live/clips'
          headers = {
               'Authorization': 'Bearer Y2vD0P7vOJ',
               'Content-Type': 'application/json'
          }
          if '"' in info["title"]:
               a = re.findall('\"\w+\"', info["title"])
               if len(a) == 1:
                    a = '“' + a[0][1:-1] + '”'
                    info["title"] = re.sub('\"\w+\"', a, info["title"])
          if len(info["title"]) > 30:
               info["title"] = info["title"][:30]
          data_str = {
               "filename": fn,
               "authorId": info["authorId"],
               "datetime": info["datetime"],
               "title": info["title"],
               "bv": info["bv"]
          }
          print(data_str)
          resp = requests.post(url=url, verify=False, data=str(data_str).replace("'",'"').encode('utf-8'), headers=headers)
          print(resp)
          res_dict = ast.literal_eval(resp.text)
          if str(resp.status_code) == '400':
               clip_id = res_dict['message'].split(':')[-1]
          else:
               clip_id = res_dict['id']
          url = f'https://api.zimu.live/clips/{clip_id}/subtitles'
          headers = {
               'Authorization': 'Bearer Y2vD0P7vOJ',
               'Content-Type': 'text/plain'
          }
          srt_path = info['srt']
          data = open(srt_path, 'r', encoding='utf-8').read().rstrip()
          resp = requests.post(url=url, verify=False, data=data.encode('utf-8'), headers=headers)
          print(resp, resp.text)

     else:
          print('video cover does not exist!')


def notify_database_for_download(info):
     # for server download
     url = 'https://disk.zimu.live:8443/disks'
     data_str = {
          "bv": info["bv"]
     }
     resp = requests.post(url=url, verify=False, data=str(data_str).replace("'",'"').encode('utf-8'))
     print(resp, resp.text)


if __name__ == "__main__":
     upload_to_database_dev({'cover': './download/cover/BV1uv411q7Mv.jpg'})