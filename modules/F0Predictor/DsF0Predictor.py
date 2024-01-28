import numpy as np

from modules.F0Predictor.F0Predictor import F0Predictor

from .dspitch import DiffSinger,SOME

class DSF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "dspitch"

    def interpolate_f0(self,f0):
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector

    def resize_f0(self,x, target_len):
        source = np.array(x)
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        return res
        
    def compute_f0(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        base="pretrain/ds_enhance"
        some=SOME(f"{base}/dsconfig.yaml",f"{base}/SOME.onnx")
        ds=DiffSinger(
            f"{base}/dsconfig.yaml",
            f"{base}/phonemes.txt",
            f"{base}/ds_linguistic.onnx",
            f"{base}/ds_pitch.onnx"
        )
        note_durs,note_rests,note_midis,ph_dur_phrase,phrases=some.inference(wav,self.sampling_rate)
        linguistic_res,n_frames,linguistic_params=ds.infer_linguistic(ph_dur_phrase,phrases)
        '''note_params={}
        note_durs = note_params["note_dur"]
        note_midis = note_params["note_midi"]
        note_rests = note_params["note_rest"]'''
        note_params={
            "note_dur":note_durs,
            "note_rest":note_rests,
            "note_midi":note_midis
        }
        f0=ds.infer_pitch(note_params,linguistic_params,n_frames,linguistic_res)
        
        
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        base="pretrain/ds_enhance"
        some=SOME(f"{base}/dsconfig.yaml",f"{base}/SOME.onnx")
        ds=DiffSinger(
            f"{base}/dsconfig.yaml",
            f"{base}/phonemes.txt",
            f"{base}/ds_linguistic.onnx",
            f"{base}/ds_pitch.onnx"
        )
        note_durs,note_rests,note_midis,ph_dur_phrase,phrases=some.inference(wav,self.sampling_rate)
        linguistic_res,n_frames,linguistic_params=ds.infer_linguistic(ph_dur_phrase,phrases)
        '''note_params={}
        note_durs = note_params["note_dur"]
        note_midis = note_params["note_midi"]
        note_rests = note_params["note_rest"]'''
        note_params={
            "note_dur":note_durs,
            "note_rest":note_rests,
            "note_midi":note_midis
        }
        f0=ds.infer_pitch(note_params,linguistic_params,n_frames,linguistic_res)
        return self.interpolate_f0(self.resize_f0(f0, p_len))
