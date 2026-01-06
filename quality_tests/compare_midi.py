import pretty_midi
import numpy as np
from scipy.stats import pearsonr
from Levenshtein import distance as lev_distance
import pathlib

def pitch_class_histogram(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    histogram = np.zeros(12)
    total_duration = 0.0
    for instrument in pm.instruments:
        for note in instrument.notes:
            pc = note.pitch % 12
            duration = note.end - note.start
            histogram[pc] += duration
            total_duration += duration
    if total_duration > 0:
        histogram /= total_duration
    return histogram

def note_sequence(midi_path, quantize=0.25):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    seq = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            start = round(note.start / quantize)
            dur = round((note.end - note.start) / quantize)
            seq.append(f"P{note.pitch}D{dur}")
    return sorted(seq)  # Sort by start time implicitly via string

def onset_offset_f1(midi1_path, midi2_path, onset_tol=0.05, offset_tol=0.1):
    pm1 = pretty_midi.PrettyMIDI(str(midi1_path))
    pm2 = pretty_midi.PrettyMIDI(str(midi2_path))
    notes1 = [n for inst in pm1.instruments for n in inst.notes]
    notes2 = [n for inst in pm2.instruments for n in inst.notes]

    matches = 0
    for n1 in notes1:
        for n2 in notes2:
            if (abs(n1.start - n2.start) <= onset_tol and
                abs(n1.end - n2.end) <= offset_tol and
                n1.pitch == n2.pitch):
                matches += 1
                break

    precision = matches / len(notes2) if notes2 else 0
    recall = matches / len(notes1) if notes1 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# ————————————————————————
# Main comparison script
# ————————————————————————

print("MIDI File Similarity Tester")
print("Enter the full paths to two MIDI files for comparison.\n")

file1 = input("Path to first MIDI file (e.g. ground_truth.mid): ").strip().strip('"\'')
file2 = input("Path to second MIDI file (e.g. transcribed.mid): ").strip().strip('"\'')
file1 = pathlib.Path(file1)
file2 = pathlib.Path(file2)

if not file1.exists():
    print(f"Error: File not found: {file1}")
    exit(1)
if not file2.exists():
    print(f"Error: File not found: {file2}")
    exit(1)

print("\nCalculating similarity metrics...\n")

# 1. Pitch class histogram correlation
hist1 = pitch_class_histogram(file1)
hist2 = pitch_class_histogram(file2)
corr, _ = pearsonr(hist1, hist2)
print(f"1. Pitch Class Histogram Correlation: {corr:.3f} (closer to 1 = more similar)")

# 2. Note sequence edit distance
seq1 = note_sequence(file1)
seq2 = note_sequence(file2)
lev = lev_distance(seq1, seq2)
max_len = max(len(seq1), len(seq2))
seq_similarity = 1 - (lev / max_len) if max_len > 0 else 0
print(f"2. Note Sequence Similarity (Levenshtein): {seq_similarity:.3f}")

# 3. Onset/Offset F1 score
f1 = onset_offset_f1(file1, file2)
print(f"3. Note Onset/Offset F1 Score (tol=50ms onset, 100ms offset): {f1:.3f}")

print("\nDone! Higher values indicate greater similarity.")