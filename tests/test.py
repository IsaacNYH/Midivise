# test_basic_pitch.py
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from pathlib import Path
import librosa
import logging
import sys
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    audio_path = Path("output/MIMI - サイエンス (feat.重音テトSV)/bass.wav")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, sr = librosa.load(str(audio_path), sr=None)
    if len(audio) == 0:
        raise ValueError(f"Audio file {audio_path} is empty or corrupted")
    logging.info(f"Audio file loaded: {audio_path}, sample rate: {sr}, duration: {len(audio)/sr:.2f}s")
    
    logging.info(f"Using default Basic Pitch model at {ICASSP_2022_MODEL_PATH}")
    
    try:
        # Verify SavedModel loading with TensorFlow
        logging.info(f"Checking SavedModel at {ICASSP_2022_MODEL_PATH}")
        model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
        logging.info("SavedModel loaded successfully with TensorFlow")
        
        # Run predict
        model_output, midi_data, note_events = predict(
            audio_path=str(audio_path)
        )
        logging.info("Predicting MIDI completed")
        
        # Save MIDI manually
        midi_path = Path("output/test") / f"{audio_path.stem}_basic_pitch.mid"
        midi_data.write(str(midi_path))
        logging.info(f"MIDI file saved: {midi_path}")
    except Exception as e:
        logging.error(f"predict or save failed: {str(e)}", exc_info=True)
        raise
    
    if midi_path.exists():
        logging.info(f"MIDI file generated: {midi_path}")
    else:
        logging.warning(f"No MIDI file generated at {midi_path}")
    logging.info("Basic Pitch test completed")
except Exception as e:
    logging.error(f"Basic Pitch test failed: {str(e)}", exc_info=True)
    sys.exit(1)