from pathlib import Path
from piano_transcription_inference import PianoTranscription
import torch
import librosa

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trans = PianoTranscription(
    device=device,
    checkpoint_path='models/piano_transcription/note_F1=0.9677_pedal_F1=0.9186.pth'
)

audio_path = 'output/MIMI - サイエンス (feat.重音テトSV)/other.wav'
output_dir = Path("output") / "MIDI_exports"
output_dir.mkdir(parents=True, exist_ok=True)
midi_output_path = output_dir / "other.mid"

# Load audio manually
audio, sr = librosa.load(audio_path, sr=16000)
print(audio.shape)  # should be (num_samples,)

# Transcribe and save MIDI
transcription = trans.transcribe(audio=audio, midi_path=str(midi_output_path))

print(f"MIDI saved to {midi_output_path}")
