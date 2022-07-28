import os
import time
from tools.download_video import download_by_bv
from tools.fake_arg_parser import FakeArg
from tools.data_utils import parse_title_new
from get_bilibili_video import get_cover_title
from upload_database import upload_to_database_dev, upload_to_database_main
from srt_generator import SRT_Generator


def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def generate_srt_by_bv(bv_dict, authorId=8, mode='main'):
    # bv_dict: dict contains bv and upload time(if not given in video title)
    # eg. bv_dict = {
    #     'BV1XU4y1e74N': '',
    #     'BV1FT411E7r7': '',
    #     'BV1Sa41157o4': '2022-06-05 12:00:00',
    # }
    args = FakeArg(
            model_folder='./ASR/20211025_conformer_exp',
            vad_model_folder='./VAD/vad_models',
            token_dict='./ASR/20211025_conformer_exp/words.txt'
        ) # change to your model path
    sg = SRT_Generator(args)
    cover_folder = './download/cover'
    video_path = './download/live_video'
    make_dir(cover_folder)
    make_dir(video_path)
    for bv in bv_dict:
        cover_path = os.path.join(cover_folder, bv+'.jpg')
        if not os.path.exists(cover_path):
            title = get_cover_title(bv) # download video cover
            db_title, upload_time = parse_title_new(title) # get video title and try to get time
            if '/' in title:
                title = title.replace('/','-') # fix some linux path problems
            if bv_dict[bv] != '':
                upload_time = bv_dict[bv]
            print(upload_time)
            video = os.path.join(video_path, title+'.mp4')
            if not os.path.exists(video):
                download_by_bv(bv, video_path) # download video by you-get
            srt = os.path.join(video_path, title+'.srt')
            if not os.path.exists(srt):
                sg.generate_srt(video) # recognize and generate srt file

            upload_info = {
                'cover': cover_path,
                'authorId': str(authorId),
                'datetime': upload_time,
                'title': db_title,
                'bv': bv,
                'srt': os.path.join(video_path, title+'.srt')
            }
            if mode == 'main':
                upload_to_database_main(upload_info) # upload to master branch
            else:
                upload_to_database_dev(upload_info) # upload to dev branch
            time.sleep(30)
        else:
            print(f'Cover file already exists for {bv}, skipping.')


if __name__ == "__main__":
     generate_srt_by_bv({'BV1XU4y1e74N': ''}, mode='dev')