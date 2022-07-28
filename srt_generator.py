import os
import shlex
import logging
from ASR.asr_recognize import ASR_Recognizer
from VAD.vad_inference import VAD_detector
from tools.data_utils import load_feature, extract_wav_from_video, write_res_file

class SRT_Generator():
    def __init__(self, args):
        self.detector = VAD_detector(args)
        self.recognizer = ASR_Recognizer(args)

    def generate_srt(self, movie_path):
        res_list = []
        print(movie_path)
        out_wav_path = movie_path.replace('.mp4', '.wav')
        out_res_path = movie_path.replace('.mp4', '.srt')
        if not os.path.exists(out_wav_path):
            movie_path = shlex.quote(movie_path)
            print(movie_path)
            out_wav_path_sh = shlex.quote(out_wav_path)
            extract_wav_from_video(movie_path, out_wav_path_sh)
        logging.info(f'Extracting file {movie_path}')
        feat, feat_len = load_feature(out_wav_path)
        logging.info(f'Cutting file {movie_path}')
        res = self.detector.vad_inference(feat)
        logging.info(f'Recognizing file {movie_path}')
        for st, ed in res:
            asr_res = self.recognizer.recognize((feat[:,st:ed], ed-st))
            if asr_res != '':
                res_list.append([st,ed,asr_res])
        write_res_file(out_res_path, res_list)