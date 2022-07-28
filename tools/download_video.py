import os
import sys
import csv
from you_get import common


def download_by_csv(data_path, download_path):
    data_dict = {}
    with open(data_path, 'r', encoding='gbk') as f:
        csv_f = csv.reader(f)
        header = next(csv_f)
        for line in csv_f:
            if len(line) > 1:
                line = [' '.join(line)]
            if len(line) == 1:
                # print(line)
                info = line[0].split('\t')
                if len(info) == 8:
                    video_name = info[7]
                    if '【直播回放】' in video_name:
                        data_dict[video_name] = info[2]
                        print(info)

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    bili_url = 'https://www.bilibili.com/video/'
    for video_name in data_dict:
        # print(bili_url+data_dict[video_name])
        common.any_download(url=bili_url+data_dict[video_name],
                            info_only=False,
                            output_dir=download_path,
                            merge=True)


def download_by_bv(bv, download_path):
    bili_url = 'https://www.bilibili.com/video/'
    common.any_download(url=bili_url+bv,
                    info_only=False,
                    output_dir=download_path,
                    merge=True)