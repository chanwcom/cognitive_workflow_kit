import torchaudio
import channel_weighting_resynthesis

# Loads a waveform from file.
file_path = "./testdata/r1.wav"
waveform, sample_rate = torchaudio.load(file_path)

# waveform: Tensor of shape (channel, time)
# sample_rate: integer

print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}")

import pdb; pdb.set_trace()

output = channel_weighting_resynthesis.generate_adversarial_speech(waveform)
