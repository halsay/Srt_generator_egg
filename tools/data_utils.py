import os
import math
import re
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi_torch

torchaudio.set_audio_backend("sox_io")


def load_feature(data_path, mel_bins=80,
                 frame_length=25, frame_shift=10):
    waveform, sample_rate = torchaudio.load(data_path)
    waveform = waveform * (1 << 15)
    feat = kaldi_torch.fbank(
        waveform,
        num_mel_bins=mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=1.0,
        energy_floor=0.0,
        sample_frequency=sample_rate)
    feat_len = len(feat)
    feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
    return feat, feat_len


def get_data_list(movie_folder):
    test_data = []
    for mp4 in os.listdir(movie_folder):
        file_name, suffix = os.path.splitext(mp4)
        if suffix == '.mp4':
            if not os.path.exists(os.path.join(movie_folder, file_name+'.srt')):
                in_movie = os.path.join(movie_folder, mp4)
                test_data.append(in_movie)
    return test_data


def write_res_file(res_file_path, res, format='srt'):
    with open(res_file_path, 'w', encoding='utf-8') as f:
        if format == 'res':
            for st, ed, asr_res in res:
                f.write(f'{st*1.0/100} {ed*1.0/100} {asr_res}\n')
        elif format == 'srt':
            idx = 1
            for st, ed, asr_res in res:
                f.write(f'{idx}\n')
                f.write(f'{convert_sec_to_time(st*1.0/100)} --> {convert_sec_to_time(ed*1.0/100)}\n')
                f.write(f'{asr_res}\n\n')
                idx += 1


def convert_sec_to_time(seconds):
    ms, s = math.modf(seconds)
    ms = int(ms * 1000)
    s = int(s)
    h = s // 3600
    m = s // 60 % 60
    s = s % 60
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def extract_wav_from_video(movie_path, wav_path):
    os.system(f'ffmpeg -i {movie_path} -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav {wav_path}')


def parse_title(title):
    time = ''
    new_title = title
    if '【直播回放】' in new_title:
        new_title = new_title.replace('【直播回放】', '')
    elif '直播回放' in new_title:
        new_title = new_title.replace('直播回放', '')
    info = new_title.split(' ')
    if len(info) == 2:
        new_title = info[0]
    elif len(info) > 2:
        new_title = ' '.join(info[:-1])
    if len(info) >= 2:
        time = get_time(info[-1])
    return new_title, time


def parse_title_new(title):
    format_time = ''
    new_title = title
    if '【直播回放】' in new_title:
        new_title = new_title.replace('【直播回放】', '')
    elif '直播回放' in new_title:
        new_title = new_title.replace('直播回放', '')
    info = re.findall('\d+年\d+月\d+日\d+点场',new_title)
    if len(info) == 1:
        new_title = new_title.replace(info[0], '')
        new_title = new_title.rstrip()
        time_info = re.findall('\d+',info[0])
        if len(time_info) == 4:
            format_time = f'{int(time_info[0])}-{int(time_info[1]):02d}-{int(time_info[2]):02d} {int(time_info[3]):02d}:00:00'
    print(info)
    return new_title, format_time


def get_time(str_time):
    format_time = ''
    info = re.findall('\d+',str_time)
    if len(info) == 4:
        format_time = f'{int(info[0])}-{int(info[1]):02d}-{int(info[2]):02d} {int(info[3]):02d}:00:00'
    return format_time
 

if __name__ == "__main__":
    # print(convert_sec_to_time(3723.67))
    print(parse_title_new('【直播回放】和水友一起玩picopark！2021年8月4日18点场'))