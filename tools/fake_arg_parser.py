class FakeArg():
    def __init__(
        self, model_folder, vad_model_folder, token_dict,
        decode_method='rescore', ctc_weight=0.5, beam=10,
        reverse_weight=0.0, device='cpu', model_type='onnx',
        precision='fp32'
    ):
        self.model_folder = model_folder
        self.vad_model_folder = vad_model_folder
        self.token_dict = token_dict
        self.decode_method = decode_method
        self.ctc_weight = ctc_weight
        self.beam = beam
        self.reverse_weight = reverse_weight
        self.device = device
        self.model_type = model_type
        self.precision = precision

