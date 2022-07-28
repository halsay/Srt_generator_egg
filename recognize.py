import os
import argparse
import shlex
from ASR.asr_recognize import ASR_Recognizer
from VAD.vad_inference import VAD_detector
from tools.data_utils import load_feature, get_data_list, extract_wav_from_video, write_res_file

parser = argparse.ArgumentParser(description='export your script model')
parser.add_argument('--model_folder', required=True, help='model folder')
parser.add_argument('--vad_model_folder', required=True, help='model folder')
parser.add_argument('--token_dict', required=True, help='token dict')
parser.add_argument('--decode_method', default='rescore')
parser.add_argument('--data_dir', required=True,
                    help='test data dir contains wav type data')
parser.add_argument('--ctc_weight', type=float, default=0.5)
parser.add_argument('--reverse_weight', type=float, default=0.5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--model_type', type=str, default='onnx')
parser.add_argument('--precision', type=str, default='fp32')


args = parser.parse_args()

detector = VAD_detector(args)
recognizer = ASR_Recognizer(args)
data_list = get_data_list(args.data_dir)
for movie_path in data_list:
    res_list = []
    out_wav_path = movie_path.replace('.mp4', '.wav')
    out_res_path = movie_path.replace('.mp4', '.srt')
    if not os.path.exists(out_wav_path):
        movie_path = shlex.quote(movie_path)
        out_wav_path_sh = shlex.quote(out_wav_path)
        extract_wav_from_video(movie_path, out_wav_path_sh)
    print(f'Extracting file {movie_path}')
    feat, feat_len = load_feature(out_wav_path)
    print(f'Cutting file {movie_path}')
    res = detector.vad_inference(feat)
    print(f'Recognizing file {movie_path}')
    for st, ed in res:
        asr_res = recognizer.recognize((feat[:,st:ed], ed-st))
        if asr_res != '':
            res_list.append([st,ed,asr_res])
        # print(f'st:{st*1.0/100} ed:{ed*1.0/100} res:{asr_res}')
    write_res_file(out_res_path, res_list)