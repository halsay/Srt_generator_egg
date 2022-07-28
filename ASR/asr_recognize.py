from __future__ import print_function
from ASR.asr_utils import *

from collections import defaultdict
import time
import argparse
import os
import numpy as np
import onnxruntime

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class ASR_Recognizer():
    def __init__(self, args):
        if args.model_type == 'libtorch':
            self.enc_model = torch.jit.load(os.path.join(
                args.model_folder, "libtorch_enc.pt"), map_location=args.device)
            self.ctc_model = torch.jit.load(os.path.join(
                args.model_folder, "libtorch_ctc.pt"), map_location=args.device)
            if args.decode_method == 'rescore':
                self.dec_model = torch.jit.load(os.path.join(
                    args.model_folder, "libtorch_dec.pt"), map_location=args.device)
            else:
                self.dec_model = None
        elif args.model_type == 'onnx':
            self.enc_model = onnxruntime.InferenceSession(os.path.join(
                args.model_folder, "encoder.onnx"))
            self.ctc_model = onnxruntime.InferenceSession(os.path.join(
                args.model_folder, "ctc.onnx"))
            if args.decode_method == 'rescore':
                self.dec_model = onnxruntime.InferenceSession(os.path.join(
                    args.model_folder, "decoder.onnx"))
            else:
                self.dec_model = None
        self.token_dict = []
        with open(args.token_dict, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line != "":
                    line = line.rstrip()
                    token = line.split(' ')[0]
                    self.token_dict.append(token)
        self.cls_num = len(self.token_dict)
        self.args = args
        

    def recognize(self, feats):
        args = self.args
        BOS = self.cls_num - 1
        EOS = self.cls_num - 1
        feat, feat_len = feats
        # print(feat.shape, feat_len)
        beam_size = args.beam
        ctc_weight = args.ctc_weight

        if args.model_type == 'libtorch':
            enc_out, enc_mask = self.enc_model(
                feat.to(args.device), torch.tensor([feat_len]).to(args.device))
            ctc_out = self.ctc_model(enc_out)
            ctc_probs = F.log_softmax(ctc_out, dim=2)
            maxlen = ctc_probs.size(1)

        elif args.model_type == 'onnx':
            feat_npy = to_numpy(feat)
            in_len = np.array([feat_len]).astype(np.int32)
            if args.precision == 'fp16':
                feat_npy = feat_npy.astype(np.float16)

            enc_inputs = {self.enc_model.get_inputs()[0].name: feat_npy,
                            self.enc_model.get_inputs()[1].name: in_len,
                            }
            enc_out, enc_out_len = self.enc_model.run(None, enc_inputs)
            ctc_inputs = {self.ctc_model.get_inputs()[0].name: enc_out}
            ctc_out = self.ctc_model.run(None, ctc_inputs)[0]
            ctc_probs = torch.tensor(ctc_out.astype(np.float32))
            enc_out = torch.tensor(enc_out.astype(np.float32))
            maxlen = ctc_probs.size(1)

        # greedy decode
        # print(ctc_probs)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        # print(topk_index.shape)
        topk_index = topk_index.squeeze(2)
        greedy_hyps = topk_index.tolist()
        # print(greedy_hyps)

        greedy_res = remove_duplicates_and_blank(greedy_hyps[0])
        out_res = greedy_res
        # print(out_res)

        # rescore decode to achieve better result (costs higher mem and computation)
        if args.decode_method == 'rescore':
            # ctc_prefix beam search
            ctc_probs = ctc_probs.squeeze(0)
            cur_hyps = [(tuple(), (0.0, -float('inf')))]

            # CTC beam search step by step
            for t in range(0, maxlen):
                logp = ctc_probs[t]  # (vocab_size,)
                # key: prefix, value (pb, pnb), default value(-inf, -inf)
                next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                # First beam prune: select topk best
                top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
                for s in top_k_index:
                    s = s.item()
                    ps = logp[s].item()
                    for prefix, (pb, pnb) in cur_hyps:
                        last = prefix[-1] if len(prefix) > 0 else None
                        if s == 0:  # blank
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pb = log_add([n_pb, pb + ps, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                        elif s == last:
                            #  Update *ss -> *s;
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pnb = log_add([n_pnb, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                            # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s,)
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                        else:
                            n_prefix = prefix + (s,)
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)

                # Second beam prune
                next_hyps = sorted(next_hyps.items(),
                                key=lambda x: log_add(list(x[1])),
                                reverse=True)
                cur_hyps = next_hyps[:beam_size]
            hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
            prefix_res = hyps[0]

            # ctc_prefix + rescore decode
            hyps_pad = pad_sequence([
                torch.tensor(hyp[0], dtype=torch.long, device=args.device)
                for hyp in hyps
            ], True, -1)  # (beam_size, max_hyps_len)
            hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                    dtype=torch.long, device=args.device)  # (beam_size,)
            ori_hyps_pad = hyps_pad
            # print(hyps_pad)
            hyps_pad, _ = add_sos_eos(hyps_pad, BOS, EOS, -1)
            # print(hyps_pad)
            hyps_lens = hyps_lens + 1  # Add <sos> at begining
            encoder_out = enc_out.repeat(beam_size, 1, 1)
            encoder_mask = torch.ones(beam_size,
                                    1,
                                    encoder_out.size(1),
                                    dtype=torch.bool,
                                    device=args.device)
            if args.model_type == 'libtorch':
                if args.reverse_weight == 0:
                    decoder_out = self.dec_model(
                        encoder_out, encoder_mask, hyps_pad,
                        hyps_lens)  # (beam_size, max_hyps_len, vocab_size)
                else:
                    r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, -1)
                    r_hyps_pad, _ = add_sos_eos(r_hyps_pad, BOS, EOS, -1)
                    hyps_tmp = hyps_pad.cpu().detach().numpy()
                    r_hyps_tmp = r_hyps_pad.cpu().detach().numpy()
                    decoder_out, r_decoder_out = self.dec_model(
                        encoder_out, encoder_mask, hyps_pad,
                        hyps_lens, r_hyps_pad)  # (beam_size, max_hyps_len, vocab_size)
                decoder_out = decoder_out.cpu().detach().numpy()
                # print(decoder_out.shape)
            elif args.model_type == 'onnx':
                hyps_pad = hyps_pad.unsqueeze(0)
                hyps_lens = hyps_lens.unsqueeze(0).to(torch.int32)

                if args.precision == 'fp16':
                    enc_out = enc_out.to(torch.half)

                if args.reverse_weight == 0:
                    ort_inputs = {self.dec_model.get_inputs()[0].name: to_numpy(enc_out),
                                self.dec_model.get_inputs()[1].name: enc_out_len,
                                self.dec_model.get_inputs()[2].name: to_numpy(hyps_pad),
                                self.dec_model.get_inputs()[3].name: to_numpy(hyps_lens)
                                }
                    decoder_out = self.dec_model.run(None, ort_inputs)
                    decoder_out = decoder_out[0][0]
                else:
                    r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, -1)
                    r_hyps_pad, _ = add_sos_eos(r_hyps_pad, BOS, EOS, -1)
                    r_hyps_pad = r_hyps_pad.unsqueeze(0)
                    # print(hyps_lens.shape, r_hyps_pad.shape, enc_out.shape)
                    ort_inputs = {self.dec_model.get_inputs()[0].name: to_numpy(enc_out),
                                self.dec_model.get_inputs()[1].name: enc_out_len,
                                self.dec_model.get_inputs()[2].name: to_numpy(hyps_pad),
                                self.dec_model.get_inputs()[3].name: to_numpy(hyps_lens),
                                self.dec_model.get_inputs()[4].name: to_numpy(r_hyps_pad)
                                }
                    decoder_out, r_decoder_out = self.dec_model.run(None, ort_inputs)
                    decoder_out = decoder_out[0]
                    r_decoder_out = r_decoder_out[0]

            # Only use decoder score for rescoring
            # print(decoder_out.shape)
            best_score = -float('inf')
            best_index = 0
            for i, hyp in enumerate(hyps):
                score = 0.0
                for j, w in enumerate(hyp[0]):
                    score += decoder_out[i][j][w]
                score += decoder_out[i][len(hyp[0])][EOS]
                # add right to left decoder score
                if args.reverse_weight > 0:
                    r_score = 0.0
                    for j, w in enumerate(hyp[0]):
                        r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                    r_score += r_decoder_out[i][len(hyp[0])][EOS]
                    score = score * (1 - args.reverse_weight) + \
                        r_score * args.reverse_weight
                # add ctc score
                score += hyp[1] * ctc_weight
                if score > best_score:
                    best_score = score
                    best_index = i
            prefix_rescore_res = hyps[best_index][0]
            out_res = prefix_rescore_res

        res = ''.join([self.token_dict[x] for x in out_res])
        return res
