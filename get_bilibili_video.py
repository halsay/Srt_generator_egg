import asyncio
import requests
from bilibili_api import video, sync

async def main():
    # 实例化 Video 类
    bv = 'BV1uv411q7Mv'
    v = video.Video(bvid=bv)
    # 获取信息
    info = await v.get_info()
    # 打印信息
    print(info)
    r = requests.get(info['pic'])
    if r.status_code == 200:
        open(f'./download/cover/{bv}.jpg','wb').write(r.content)

def get_cover_title(bv):
    v = video.Video(bvid=bv)
    info = sync(v.get_info())
    r = requests.get(info['pic'])
    if r.status_code == 200:
        open(f'./download/cover/{bv}.jpg','wb').write(r.content)
        title = info['title']
        return title
    else:
        print('download cover failed')
        return ''

if __name__ == '__main__':
    # asyncio.get_event_loop().run_until_complete(main())
    t = get_cover_title('BV1rb4y1r7wi')
    print(t)