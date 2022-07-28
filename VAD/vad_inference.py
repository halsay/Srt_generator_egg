import torch
import torch.nn as nn
import onnxruntime
import os
from VAD.vad_utils import get_vadtime

class VAD_detector():
    def __init__(self, args):
        if args.model_type == 'libtorch':
            self.sess = torch.jit.load(os.path.join(
                args.vad_model_folder, "vad_model.script"), map_location=args.device)
        elif args.model_type == 'onnx':
            self.sess = onnxruntime.InferenceSession(os.path.join(
                    args.vad_model_folder, "vad_model.onnx"))
        self.args = args

    def vad_inference(self, feat):
        args = self.args
        if 'onnx' in args.model_type:
            input_name = self.sess.get_inputs()[0].name
            output_name = self.sess.get_outputs()[0].name
            input_ = feat
            input_ = input_.numpy()
            pred = self.sess.run([output_name], {input_name: input_})[0]
            pred[pred == 2] = 1
            pred = 1 - pred
            pred_labels = pred.flatten()
            seg = get_vadtime(pred_labels)
        return seg