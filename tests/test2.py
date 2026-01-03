# test_madmom.py
import madmom
import librosa
import soundfile as sf
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)

stem_path = "output/MIMI - サイエンス (feat.重音テトSV)/drums.wav"
try:
    # Clean up any existing temp file
    temp_wav = Path(stem_path).parent / "temp_drums.wav"
    if temp_wav.exists():
        try:
            temp_wav.unlink()
            logging.info(f"Cleaned up existing {temp_wav}")
        except PermissionError:
            logging.warning(f"Could not clean up {temp_wav}, may be in use")
    
    audio, sr = librosa.load(stem_path, sr=44100, mono=True)
    audio = librosa.util.normalize(audio)
    with sf.SoundFile(str(temp_wav), 'w', samplerate=sr, channels=1) as f:
        f.write(audio)
    proc = madmom.features.onsets.RNNOnsetProcessor()
    onsets = proc(str(temp_wav))
    logging.info(f"Madmom onsets: {onsets}")
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if temp_wav.exists():
                temp_wav.unlink()
            logging.info(f"Successfully deleted {temp_wav}")
            break
        except PermissionError:
            logging.warning(f"Attempt {attempt + 1}: Cannot delete {temp_wav}, file in use")
            time.sleep(0.5)
    else:
        logging.warning(f"Failed to delete {temp_wav} after {max_attempts} attempts, skipping")
except Exception as e:
    logging.error(f"Madmom test failed: {str(e)}")