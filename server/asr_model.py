import sys
import os
from abc import ABC, abstractmethod
import torch


Firered_asr_root_dir = Firered_asr_root_dir = os.path.dirname(os.getcwd())

sys.path.append(f'{Firered_asr_root_dir}')
sys.path.append(f'{Firered_asr_root_dir}/fireredasr')
sys.path.append(f'{Firered_asr_root_dir}/fireredasr/utils')

from fireredasr.models.fireredasr import FireRedAsr





# 基类定义
class ASR_engine_base(ABC):
    def __init__(self, params=None, device='cpu',):
        self.model_name = "base_model"
        self.model_space = "FireRedASR"
        self.model_vision = "base_vision"
        self.engine_name = '{}@{}@{}'.format(self.model_space, self.model_name, self.model_vision)
        
        
    

    @abstractmethod
    def asr_inference(self, speech_data, segs = [],  fs=16000):
        pass
    
    

class FireRedASR_AED_engine(ASR_engine_base):
    def __init__(self, params=None, device='cpu',):
        super().__init__(params, device)
        self.model_name = "FireRedASR_AED"
        
        gpu_id = torch.cuda.current_device()
        device = 'cpu'
        if torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            
        
        self.model_dir = f'{Firered_asr_root_dir}/pretrained_models/FireRedASR-AED-L/'
        
        self.model = FireRedAsr.from_pretrained('aed', self.model_dir, device=device)
        
        self.args = {
                        'use_gpu': 1,
                        'batch_size': 4,
                        'beam_size': 1,
                        'decode_max_len': 0,
                        'nbest': 1,
                        'softmax_smoothing': 1.0,
                        'aed_length_penalty': 0.0,
                        'eos_penalty': 1.0,
                        'decode_min_len': 0,
                        'repetition_penalty': 1.0,
                        'llm_length_penalty': 0.0,
                        'temperature': 1.0
                    }

    def prepare_segs(self, speech_data, segs=[], fs=16000):
        chunks = []
        utts = []
        if len(segs) == 0:
            return ['utt'], [[speech_data, fs]]
        
        for i, seg in enumerate(segs):
            st = int(seg[0]*fs/1000.0)
            ed = int(seg[1]*fs/1000.0)
            chunk = speech_data[st:ed]  
            utt = f'seg_{i}'
            utts.append(utt)
            chunks.append([chunk, fs])
        
        return utts, chunks
        
    
    def asr_inference(self, speech_data, segs=[], fs=16000):
        utts, chunks = self.prepare_segs(speech_data, segs, fs)
        results = self.model.transcribe(utts, chunks, self.args)
        
        res = {}
        res['model_space'] = self.model_space
        res['model_name'] = self.model_name
        res['model_vision'] = self.model_vision
        
        res['segs'] = []
        
        full_text = ''
        for result, seg in zip(results, segs):
            text = result['text']
            res['segs'].append({"seg": seg, "text": text})
            full_text += text
        
        res['full_text'] = full_text
        
        return res
    
    

if __name__ == '__main__':
    import soundfile as sf
    
    model = FireRedASR_AED_engine()
    
    
    wav_path = '/data/nas/dataset/asr/kefu/shidian/wavs/date1211/cc-1866736084303028224.wav'
    
    segs = [[1380, 4110], [14500, 16070], [16390, 18640], [20510, 23060], [23550, 27920], [30000, 32530], [34920, 36480]]
    
    speech_data, fs = sf.read(wav_path, dtype='int16')
    
    res = model.asr_inference(speech_data, segs=segs, fs=fs)
    
    print(res)