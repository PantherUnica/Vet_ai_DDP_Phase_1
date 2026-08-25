from transformers import AutoModel
import torch
import torchaudio

AUDIO_PATH = r"C:\Users\Aditya\OneDrive\Desktop\VetAI_Phase1\input_audio_examples\soap_testing.wav"
LANG = "hi"   # change if needed: hi, te, ta, ml, bn, mr, gu, kn, etc.
DECODE = "ctc"

model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
)

wav, sr = torchaudio.load(AUDIO_PATH)
wav = torch.mean(wav, dim=0, keepdim=True)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wav = resampler(wav)

text = model(wav, LANG, DECODE)
print(text)