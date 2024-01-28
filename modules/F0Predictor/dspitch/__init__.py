import soundfile as sf
from .utils.onnx import infer
from .utils.slicer import Slicer
import yaml
import numpy as np
import math

class DiffSinger:
    def __init__(self,dsconfig_path,phoneme_path,linguistic_path,pitch_base_path):
        with open (dsconfig_path,"r",encoding="utf-8") as f:
            dsConfig=yaml.safe_load(f)
        with open(phoneme_path, "r") as f:
            phonemes = f.read().splitlines()
        self.dsconfig=dsConfig
        self.frame_ms=1000 * dsConfig["hop_size"] / dsConfig["sample_rate"]
        self.head_ms=100
        self.tail_ms=100
        self.phonemes=phonemes
        self.linguistic_path=linguistic_path
        self.pitch_base_path=pitch_base_path


    def infer_linguistic(self,ph_dur_phrase,phrases):
        def calculate_ph_dur(phrase, frame_ms, head_ms, tail_ms):
            ph_dur = []
            head_frames=head_ms/frame_ms
            tail_frames=tail_ms/frame_ms
            for p in phrase:
                start_frame = (int(p['endMs'] / frame_ms) - int(p['positionMs'] / frame_ms))
                ph_dur.append(start_frame)
            return ph_dur
        
        ph_dur=calculate_ph_dur(ph_dur_phrase,self.frame_ms,self.head_ms,self.frame_ms)
        phrase_phones = phrases

        # 获取音素列表的长度
        phonemes=self.phonemes
        # 获取音素列表的长度
        len_phonemes = len(phonemes)

        # 初始化结果列表
        tokens = []

        # 遍历字符串中的每个音素
        for p in phrase_phones:
            # 获取音素在音素列表中的索引
            index = phonemes.index(p)

            # 添加特殊标记
            if index == 0:
                tokens.append(0)  # 添加左边界标记
            elif index == len_phonemes - 1:
                tokens.append(len_phonemes - 1)  # 添加右边界标记
            else:
                tokens.append(index)  




        # 输出结果
        print(tokens)
        print(ph_dur)
        n_frames=0
        for ph_d in ph_dur:
            n_frames+=ph_d
        print(n_frames)
        print(range(n_frames))
        linguistic_params={}
        linguistic_params["tokens"] = tokens

        n_tokens = len(tokens)

        linguistic_params["tokens"]=np.array(linguistic_params["tokens"]).reshape(1,n_tokens).astype(np.int64)

        linguistic_params["ph_dur"]=np.array(ph_dur).reshape(1,n_tokens).astype(np.int64)

        linguistic_res=infer(f"{self.linguistic_path}",linguistic_params)
        return linguistic_res,n_frames,linguistic_params
    def infer_pitch(self,note_params,linguistic_params,n_frames,linguistic_res):
        retake=[]
        for i in range(n_frames):
            retake.append(False)
        # Check if required linguistic and note parameters are present
        assert 'ph_dur' in linguistic_params, "Missing 'ph_dur' must in linguistic_params"
        assert 'encoder_out' in linguistic_res, "Missing 'encoder_out' must in linguistic_res"
        assert 'note_dur' in note_params, "Missing 'note_dur' must in note_params"
        assert 'note_midi' in note_params, "Missing 'note_midi' must in note_params"
        assert 'note_rest' in note_params, "Missing 'note_rest' must in note_params"
        note_durs = note_params["note_dur"]
        note_midis = note_params["note_midi"]
        note_rests = note_params["note_rest"]

        print(retake)
        print(note_durs)
        print(np.array([1]).astype(np.int64))
        pitch_params={
            "encoder_out":linguistic_res["encoder_out"],
            "ph_dur":linguistic_params["ph_dur"],
            "note_dur":note_durs,
            "note_midi":note_midis,
            "note_rest":note_rests,
            "pitch":np.zeros((1,n_frames)).astype(np.float32),
            "retake":np.array(retake).reshape(1,n_frames),
            "speedup":np.array(50).astype(np.int64)

        }
        pitch_res=infer(f"{self.pitch_base_path}",pitch_params)
        pitch_preds = pitch_res["pitch_pred"][0]
        print(pitch_res)
        print(pitch_preds )
        pred_f0s=[]
        for item in pitch_preds:
            m=item
            fm=440*(math.pow(2,((m-69)/12)))
            pred_f0s.append(fm)
        return pred_f0s

class SOME:
    def __init__(self,dsconfig_path,some_path):
        self.some_path=str(some_path)
        with open (dsconfig_path,"r",encoding="utf-8") as f:
            dsConfig=yaml.safe_load(f)
        self.frame_ms=1000 * dsConfig["hop_size"] / dsConfig["sample_rate"]
    def inference(self,wf,osr=44100):
        #exit()
        owfl=wf.shape[0]
        #wf=np.mean(wf,axis=1)
        #wf=wf.reshape(1,owfl)
        #wf=wf.astype(np.float32)
        slicer = Slicer(sr=osr, max_sil_kept=1000)
        slices=[{"offset":0,"r":wf}]
        print(slices)
        #exit()
        t=0
        SOME_responses=[]
        phrases=[]
        world_datas=[]
        linguistic_params={"ph_dur":[]}
        for segment in slices:
            segment_wf=segment["waveform"]
            res=infer(f"{self.some_path}",{
                "waveform":segment_wf.reshape([1,segment_wf.shape[0]])
            })
            SOME_responses.append({"r":res,"offset":segment["offset"]})
            #res like {"note_midi":,"note_rest":,"note_dur":}
        note_midis=[]
        note_durs=[]
        note_rests=[]
        ph_dur_phrase=[]
        n_notes=0
        last_note_dur=0
        for rd in SOME_responses:
            res=rd["r"]
            last_note_dur=rd["offset"]
            note_rest=res["note_rest"]
            print(res)
            for flags in note_rest:
                for flag in flags:
                    note_rests.append(flag)
                    if flag:
                        phrases.append("a")
                    else:
                        phrases.append("SP")
            for durs in res["note_dur"]:
                for dur in durs:
                    now_dur=last_note_dur+dur
                    frame_ms=self.frame_ms
                    note_durs.append(int(now_dur*1000/frame_ms)-int(last_note_dur*1000/frame_ms))
                    ph_dur_phrase.append({"positionMs":last_note_dur*1000,"endMs":now_dur*1000})
                    last_note_dur=now_dur
            for mids in res["note_midi"]:
                for mid in mids:
                    note_midis.append(mid)
                    n_notes+=1

        note_durs=np.array(note_durs).reshape(1,n_notes).astype(np.int64)
        note_rests=np.array(note_rests).reshape(1,n_notes)
        note_midis=np.array(note_midis).reshape(1,n_notes).astype(np.float32)
        return note_durs,note_rests,note_midis,ph_dur_phrase,phrases
        
