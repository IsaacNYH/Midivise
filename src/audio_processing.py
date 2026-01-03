import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage
from demucs.pretrained import get_model
from demucs.apply import apply_model
import logging
import tensorflow as tf
from magenta.models.onsets_frames_transcription import infer  # Updated import to use infer.py
from magenta.models.onsets_frames_transcription import configs  # Keep for CONFIG_MAP
import pretty_midi
import subprocess
import re
from scipy.signal import butter, lfilter
from midi2audio import FluidSynth
import sys
from utils import get_resource_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Memory growth set to True for GPU(s)")
    except RuntimeError as e:
        logging.error(f"Failed to set memory growth: {str(e)}")

try:
    from basic_pitch.inference import predict, ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    logging.error("Basic Pitch not available. Install with: pip install basic-pitch")
    ICASSP_2022_MODEL_PATH = None

try:
    from piano_transcription_inference import PianoTranscription
    PIANO_TRANSCRIPTION_AVAILABLE = True
except ImportError:
    PIANO_TRANSCRIPTION_AVAILABLE = False
    logging.error("PianoTranscription not available. Install with: pip install piano_transcription_inference")

try:
    from midi2audio import FluidSynth
    MID2AUDIO_AVAILABLE = True
except ImportError:
    MID2AUDIO_AVAILABLE = False
    logging.error("midi2audio not available. Install with: pip install midi2audio and ensure FluidSynth is installed.")

def run_demucs(audio_path):
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    local_model_path = get_resource_path("models/demucs/htdemucs.pth")
    if not local_model_path.exists():
        raise FileNotFoundError(f"Demucs model not found at {local_model_path}")

    try:
        model = get_model("htdemucs")
        state_dict = torch.load(str(local_model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        waveform, sr = librosa.load(audio_path, sr=None, mono=False)
        waveform = torch.tensor(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        waveform = waveform.unsqueeze(0).to(device)  # (1, 2, length)

        with torch.no_grad():
            sources = apply_model(model, waveform, progress=True)[0]

        out_dir = Path("output") / Path(audio_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = {}
        stem_names = ['drums', 'bass', 'other', 'vocals']
        for i, stem in enumerate(stem_names):
            path = out_dir / f"{stem}.wav"
            data = sources[i].cpu().numpy().T  # (length, channels)
            sf.write(str(path), data, sr)
            stems[stem] = path
        logging.info(f"Stems saved to {out_dir}")
        return stems
    except Exception as e:
        logging.error(f"Demucs processing failed: {str(e)}")
        raise RuntimeError(f"Demucs processing failed: {str(e)}")

def _validate_checkpoint(checkpoint_dir: Path):
    """Validate checkpoint files based on step in 'checkpoint' file."""
    ckpt_file = checkpoint_dir / "checkpoint"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Missing 'checkpoint' in {checkpoint_dir}")

    with open(ckpt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Empty checkpoint file: {ckpt_file}")

    match = re.search(r'model\.ckpt-(\d+)', content)
    if not match:
        raise ValueError(f"Cannot parse step from checkpoint: {content}")
    step = match.group(1)

    required = [
        f"model.ckpt-{step}.data-00000-of-00001",
        f"model.ckpt-{step}.index",
        f"model.ckpt-{step}.meta"
    ]
    missing = [f for f in required if not (checkpoint_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for step {step}: {missing}")

    logging.info(f"E-GMD checkpoint validated at step {step}")
    return step


def transcribe_drums(stem_path, model, output_path):
    if model != 'OaF Drums (recommended)':
        raise ValueError("Only OaF Drums is supported for drum transcription")

    checkpoint_dir = get_resource_path("models/drum_transcription/e-gmd")
    if not (checkpoint_dir / "checkpoint").exists():
        raise FileNotFoundError(f"E-GMD model not found at {checkpoint_dir}")

    # Validate checkpoint
    _validate_checkpoint(checkpoint_dir)

    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resample to 16kHz mono
        logging.info(f"Resampling {stem_path} to 16000 Hz...")
        audio, sr = librosa.load(str(stem_path), sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError(f"Empty audio: {stem_path}")

        temp_wav = output_dir / f"{Path(stem_path).stem}_16k.wav"
        sf.write(str(temp_wav), audio, 16000)

        if not temp_wav.exists():
            raise RuntimeError(f"Failed to create resampled file: {temp_wav}")
        logging.info(f"Resampled WAV ready: {temp_wav} ({temp_wav.stat().st_size} bytes)")

        # Use absolute path
        input_file = str(temp_wav.resolve())

        cmd = [
            sys.executable,
            '-m', 'magenta.models.onsets_frames_transcription.onsets_frames_transcription_transcribe',
            '--config=drums',
            '--model_dir', str(checkpoint_dir),
            input_file
        ]

        logging.info(f"Running Magenta CLI...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        logging.info(f"CLI stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"CLI stderr: {result.stderr}")

        # Magenta outputs: <input>.wav.midi
        input_path = Path(input_file)
        generated_midi = input_path.with_suffix('.wav.midi')

        if not generated_midi.exists():
            # Fallback: maybe .mid
            generated_midi = input_path.with_suffix('.mid')
            if not generated_midi.exists():
                raise RuntimeError(f"MIDI not found. Expected .wav.midi or .mid")

        logging.info(f"Magenta generated: {generated_midi}")

        # Rename to final drums.mid
        final_path = Path(output_path)
        if generated_midi != final_path:
            if final_path.exists():
                final_path.unlink()
            generated_midi.rename(final_path)
            logging.info(f"Final MIDI saved: {final_path}")

        # Cleanup temp WAV
        try:
            temp_wav.unlink()
            logging.info(f"Cleaned up: {temp_wav}")
        except:
            pass

        # Verify drum notes
        pm = pretty_midi.PrettyMIDI(str(final_path))
        drum_notes = sum(len(i.notes) for i in pm.instruments if i.is_drum)
        logging.info(f"SUCCESS: {drum_notes} drum hits transcribed!")
        if drum_notes == 0:
            logging.warning("No drum hits detected. Stem may be silent.")

        return str(final_path)

    except subprocess.CalledProcessError as e:
        error_msg = f"Magenta CLI failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        logging.error(f"Drum transcription failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Drum transcription failed: {e}")
    
def transcribe_bass(stem_path, model, program, output_path):
    if not BASIC_PITCH_AVAILABLE:
        raise RuntimeError("Basic Pitch not available. Install with: pip install basic-pitch")
    if model != 'Basic Pitch (recommended)':
        raise ValueError("Only Basic Pitch is supported for bass transcription")
    
    try:
        # Load audio and get total duration
        audio, sr = librosa.load(str(stem_path), sr=None)
        if len(audio) == 0:
            raise ValueError(f"Audio file {stem_path} is empty or corrupted")
        total_duration = len(audio) / sr  # Total duration in seconds
        logging.info(f"Audio file loaded: {stem_path}, sample rate: {sr}, duration: {total_duration:.2f}s")

        # Initialize MIDI file
        mid = MidiFile()
        mid.ticks_per_beat = 480
        tempo_track = MidiTrack()
        tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
        mid.tracks.append(tempo_track)
        note_track = MidiTrack()
        note_track.append(Message('program_change', channel=1, program=program, time=0))
        mid.tracks.append(note_track)

        # Calculate ticks per second (120 BPM = 2 beats per second)
        ticks_per_second = mid.ticks_per_beat * 2  # 120 BPM = 2 beats/second

        # Process audio with Basic Pitch
        logging.info(f"Using default Basic Pitch model at {ICASSP_2022_MODEL_PATH}")
        with tf.device('/CPU:0'):
            model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
            logging.info("SavedModel loaded successfully with TensorFlow on CPU")
            model_output, midi_data, note_events = predict(audio_path=str(stem_path))
            logging.info(f"midi_data type: {type(midi_data)}, notes: {sum(len(i.notes) for i in midi_data.instruments)}")

        # Collect and filter notes
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                # Filter invalid notes
                if start_time < 0 or end_time <= start_time or end_time > total_duration:
                    logging.warning(f"Skipping invalid note: pitch={note.pitch}, start={start_time:.2f}s, end={end_time:.2f}s")
                    continue
                all_notes.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })

        # Sort notes by start time
        all_notes.sort(key=lambda x: x['start_time'])

        # Add notes to track
        prev_tick = 0
        for note in all_notes:
            start_tick = int(note['start_time'] * ticks_per_second)
            duration_tick = int((note['end_time'] - note['start_time']) * ticks_per_second)
            if duration_tick <= 0:
                logging.warning(f"Skipping note with invalid duration: pitch={note['pitch']}, start={note['start_time']:.2f}s")
                continue
            # Ensure non-negative delta time
            delta_time = max(0, start_tick - prev_tick)
            note_track.append(Message('note_on', note=note['pitch'], velocity=note['velocity'], 
                                     time=delta_time, channel=1))
            note_track.append(Message('note_off', note=note['pitch'], velocity=0, 
                                     time=duration_tick, channel=1))
            prev_tick = start_tick + duration_tick
            logging.debug(f"Added note: pitch={note['pitch']}, start={note['start_time']:.2f}s, end={note['end_time']:.2f}s, delta_time={delta_time}")

        # Save MIDI file
        mid.save(str(output_path))
        logging.info(f"Bass MIDI saved to {output_path}")

        debug_midi_programs(output_path)

        # Verify MIDI duration
        midi_duration_ticks = sum(msg.time for msg in note_track)
        midi_duration_seconds = midi_duration_ticks / ticks_per_second
        logging.info(f"MIDI duration: {midi_duration_seconds:.2f}s")
        if midi_duration_seconds > total_duration * 1.1:  # Allow 10% margin
            logging.warning(f"MIDI duration ({midi_duration_seconds:.2f}s) exceeds audio duration ({total_duration:.2f}s)")

        return output_path
    except Exception as e:
        logging.error(f"Bass transcription failed for {stem_path}: {str(e)}")
        raise RuntimeError(f"Bass transcription failed: {str(e)}")
    
def transcribe_piano(stem_path, model, program, output_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Starting piano transcription for {stem_path} with model {model} on device {device}")
    if model == 'Onsets & Frames (recommended)':
        if not PIANO_TRANSCRIPTION_AVAILABLE:
            raise RuntimeError("PianoTranscription not available. Install with: pip install piano_transcription_inference")
        local_checkpoint = get_resource_path("models/piano_transcription/note_F1=0.9677_pedal_F1=0.9186.pth")
        if not local_checkpoint.exists():
            raise FileNotFoundError(f"Piano transcription model not found at {local_checkpoint}")
        try:
            logging.info(f"Loading Onsets & Frames model from {local_checkpoint}")
            trans = PianoTranscription(
                device=device,
                checkpoint_path=str(local_checkpoint)
            )
            audio, sr = librosa.load(str(stem_path), sr=16000)
            logging.info(f"Loaded audio with length {len(audio)} samples at {sr} Hz")
            trans.transcribe(audio, str(output_path))
            logging.info(f"Piano MIDI saved to {output_path}")

            debug_midi_programs(output_path)

            import pretty_midi

            pm = pretty_midi.PrettyMIDI(str(output_path))

            for instrument in pm.instruments:
                if not instrument.is_drum:  # Shouldn't be drums anyway
                    instrument.program = program  # Use the spinbox value from GUI

                    # Add explicit bank select for maximum compatibility (same as synthesize_midi)
                    instrument.control_changes.append(
                        pretty_midi.ControlChange(number=0, value=0, time=0)  # Bank MSB = 0
                    )
                    instrument.control_changes.append(
                        pretty_midi.ControlChange(number=32, value=0, time=0)  # Bank LSB = 0
                    )

            pm.write(str(output_path))
            logging.info(f"Applied program {program} to Onsets & Frames piano MIDI")

            if Path(output_path).exists():
                return output_path
            else:
                logging.error(f"MIDI file not found after writing to {output_path}")
                raise RuntimeError(f"Failed to verify MIDI file creation at {output_path}")
        except Exception as e:
            logging.error(f"Piano transcription (Onsets & Frames) failed for {stem_path}: {str(e)}")
            raise RuntimeError(f"Piano transcription (Onsets & Frames) failed: {str(e)}")
    elif model == 'Basic Pitch':
        if not BASIC_PITCH_AVAILABLE:
            raise RuntimeError("Basic Pitch not available. Install with: pip install basic-pitch")
        try:
            audio, sr = librosa.load(str(stem_path), sr=None)
            logging.info(f"Loaded audio from {stem_path} with length {len(audio)} samples at {sr} Hz")
            if len(audio) == 0:
                raise ValueError(f"Audio file {stem_path} is empty or corrupted")
            
            logging.info(f"Using default Basic Pitch model at {ICASSP_2022_MODEL_PATH}")
            with tf.device('/CPU:0'):
                model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
                logging.info("SavedModel loaded successfully with TensorFlow on CPU")
                model_output, midi_data, note_events = predict(audio_path=str(stem_path))
                logging.info(f"midi_data type: {type(midi_data)}, notes detected: {sum(len(i.notes) for i in midi_data.instruments)}")
                logging.info("Predicting MIDI completed on CPU")
            
            midi_path = Path(output_path)
            midi_data.write(str(midi_path))
            logging.info(f"MIDI file saved: {midi_path}")
            
            # Post-process: set channel 0, program change
            mid = MidiFile(str(midi_path))
            for track in mid.tracks:
                try:
                    track.insert(0, Message('program_change', channel=0, program=program, time=0))
                    for msg in track:
                        if hasattr(msg, 'channel'):
                            msg.channel = 0
                except Exception as e:
                    logging.error(f"Post-processing failed for track in {midi_path}: {str(e)}")
                    raise
            mid.save(str(midi_path))
            logging.info(f"Piano MIDI saved to {midi_path}")
            if Path(midi_path).exists():
                return str(midi_path)
            else:
                logging.error(f"MIDI file not found after post-processing at {midi_path}")
                raise RuntimeError(f"Failed to verify MIDI file after post-processing at {midi_path}")
        except Exception as e:
            logging.error(f"Piano transcription (Basic Pitch) failed for {stem_path}: {str(e)}")
            raise RuntimeError(f"Piano transcription (Basic Pitch) failed: {str(e)}")
    else:
        raise ValueError("Invalid model for piano transcription")

def generate_midi(notes, instrument_program: int, output_path: str, is_drum: bool = False):
    try:
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        tempo = MetaMessage('set_tempo', tempo=500000, time=0)  # 120 BPM
        track.append(tempo)

        channel = 9 if is_drum else 0
        if not is_drum:
            track.append(Message('program_change', program=instrument_program, time=0, channel=channel))

        ticks_per_beat = mid.ticks_per_beat
        ticks_per_second = ticks_per_beat * (120 / 60)  # For 120 BPM

        prev_tick = 0
        for start, end, pitch in sorted(notes, key=lambda x: x[0]):
            start_tick = int(start * ticks_per_second)
            duration_tick = int((end - start) * ticks_per_second)
            if duration_tick <= 0:
                logging.warning(f"Skipping note with invalid duration at {start} for {output_path}")
                continue
            track.append(Message('note_on', note=int(pitch), velocity=100, time=(start_tick - prev_tick), channel=channel))
            track.append(Message('note_off', note=int(pitch), velocity=0, time=duration_tick, channel=channel))
            prev_tick = start_tick + duration_tick

        mid.save(output_path)
        logging.info(f"MIDI file generated: {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate MIDI {output_path}: {str(e)}")
        raise RuntimeError(f"MIDI generation failed: {str(e)}")

def merge_midis(midi_paths, selected_instruments, output_path):
    try:
        target_ticks_per_beat = 480
        mid = MidiFile()
        mid.ticks_per_beat = target_ticks_per_beat
        tempo_track = MidiTrack()
        tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        mid.tracks.append(tempo_track)
        
        for p in midi_paths:
            temp_mid = MidiFile(str(p))
            source_ticks_per_beat = temp_mid.ticks_per_beat
            scale = target_ticks_per_beat / source_ticks_per_beat
            for track in temp_mid.tracks[1:]:  # Skip meta if present
                new_track = MidiTrack()
                for msg in track:
                    new_msg = msg.copy()
                    new_msg.time = int(new_msg.time * scale)
                    new_track.append(new_msg)
                mid.tracks.append(new_track)
        mid.save(str(output_path))
        logging.info(f"Merged MIDI saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"MIDI merge failed for {output_path}: {str(e)}")
        raise RuntimeError(f"MIDI merge failed: {str(e)}")

def synthesize_midi(midi_path, audio_path, soundfont=None):
    """
    Synthesize MIDI → WAV using direct FluidSynth CLI.
    Handles Japanese/Unicode paths safely.
    Ensures correct instrument by explicitly inserting bank select CC0=0, CC32=0
    before any program change (prevents fallback to piano in some cases).
    """
    if soundfont is None:
        project_root = Path(__file__).parent.parent
        sf_path = get_resource_path("models/soundfont/FluidR3_GM.sf2")
        if not sf_path.exists():
            raise FileNotFoundError(f"SoundFont missing: {sf_path}")
        soundfont = str(sf_path.resolve())

    # Load MIDI and add explicit bank select (0,0) for all non-drum instruments
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for instrument in pm.instruments:
        if not instrument.is_drum:
            # Insert Bank Select MSB (CC0) = 0 and LSB (CC32) = 0 at time 0
            # This forces standard GM bank and prevents fallback to default piano
            instrument.control_changes.insert(
                0, pretty_midi.ControlChange(number=0, value=0, time=0)
            )
            instrument.control_changes.insert(
                1, pretty_midi.ControlChange(number=32, value=0, time=0)
            )

    # Save temporary MIDI with bank selects
    temp_midi = Path(audio_path).with_suffix('.tmp.mid')
    pm.write(str(temp_midi))

    cmd = [
        'fluidsynth',
        '-ni',                  # No interactive shell
        '-F', str(audio_path),  # Output to WAV/RAW
        '-r', '44100',          # Sample rate
        '-g', '1.0',            # Gain (adjust if clipping occurs)
        soundfont,
        str(temp_midi)
    ]

    logging.info(f"Running FluidSynth: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
        )
        # Safely decode stderr for warnings
        try:
            stderr = result.stderr.decode('utf-8', errors='replace')
            if stderr.strip():
                logging.warning(f"FluidSynth: {stderr.strip()}")
        except Exception:
            pass

        logging.info("MIDI synthesis complete!")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        logging.error(f"FluidSynth failed: {err}")
        raise RuntimeError(f"Synthesis failed: {err}")
    except FileNotFoundError:
        raise RuntimeError("fluidsynth.exe not found. Add to PATH.")
    finally:
        # Cleanup temporary MIDI file
        try:
            temp_midi.unlink()
            logging.info(f"Cleaned up temporary MIDI: {temp_midi}")
        except Exception:
            pass

def merge_audio(audio_paths):
    try:
        audios = []
        sr = None
        max_len = 0
        for p in audio_paths:
            a, s = librosa.load(str(p), sr=None)
            audios.append(a)
            sr = s
            max_len = max(max_len, len(a))
        for i in range(len(audios)):
            if len(audios[i]) < max_len:
                audios[i] = np.pad(audios[i], (0, max_len - len(audios[i])))
        merged = np.sum(audios, axis=0) / len(audios)  # ← ADD THIS: Divide by number of tracks to normalize
        merged = np.clip(merged, -1.0, 1.0)  # ← ADD THIS: Clip any remaining overflow
        logging.info(f"Merged {len(audio_paths)} audio tracks")
        return merged, sr
    except Exception as e:
        logging.error(f"Audio merge failed: {str(e)}")
        raise RuntimeError(f"Audio merge failed: {str(e)}")

def save_audio(audio, sr, path):
    try:
        sf.write(str(path), audio, sr)
        logging.info(f"Audio saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save audio {path}: {str(e)}")
        raise RuntimeError(f"Audio save failed: {str(e)}")
    
def debug_midi_programs(midi_path):
    """
    Quick debug tool to print what program numbers and bank selects are actually in a MIDI file.
    """
    import pretty_midi
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        print(f"\n=== Debugging MIDI: {midi_path} ===")
        if not pm.instruments:
            print("No instruments found in the MIDI file!")
            return

        for i, inst in enumerate(pm.instruments):
            print(f"Instrument {i}:")
            print(f"  is_drum: {inst.is_drum}")
            print(f"  program: {inst.program}  ← This is the GM instrument number (0=Piano, 32=Acoustic Bass, 33=Electric Bass finger, etc.)")
            print(f"  name:    {inst.name if inst.name else '(no name)'}")

            # Check for bank select (CC0 and CC32)
            bank_msb = [cc.value for cc in inst.control_changes if cc.number == 0]
            bank_lsb = [cc.value for cc in inst.control_changes if cc.number == 32]
            if bank_msb or bank_lsb:
                print(f"  Bank Select MSB (CC0):  {bank_msb}")
                print(f"  Bank Select LSB (CC32): {bank_lsb}")
            else:
                print("  No bank select messages (normal for GM)")

            # Note: pretty_midi does NOT expose separate program_changes list
            # The program is set via inst.program
            print("  Note: Program changes are embedded in instrument.program (above)")
        
        print("=== End Debug ===\n")
    except Exception as e:
        print(f"Failed to read MIDI file: {e}")
        import traceback
        traceback.print_exc()