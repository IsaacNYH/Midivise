# src/visualization.py
from mido import MidiFile, MetaMessage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import logging

def generate_midi_vis(midi_path):
    if not midi_path or not Path(midi_path).exists():
        logging.warning(f"Invalid or missing MIDI file: {midi_path}")
        return None
    try:
        mid = MidiFile(str(midi_path))
        notes = []
        time = 0.0
        tempo = 500000  # Default microseconds per beat
        ticks_per_beat = mid.ticks_per_beat
        active = {}

        for track in mid.tracks:
            for msg in track:
                time += msg.time * (tempo / 1e6) / ticks_per_beat
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    active[(msg.channel, msg.note)] = time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in active:
                        start = active.pop(key)
                        dur = time - start
                        notes.append((start, dur, msg.note, msg.channel))

        if not notes:
            logging.warning(f"No notes found in MIDI file: {midi_path}")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {0: 'blue', 1: 'green', 9: 'red', 10: 'red'}  # Add channel 10 for standard drum channel
        for start, dur, note, channel in notes:
            color = colors.get(channel, 'black')
            ax.add_patch(patches.Rectangle((start, note - 0.5), dur, 1, facecolor=color, edgecolor=color))
        ax.set_xlim(0, max(start + dur for start, dur, _, _ in notes) + 1)
        ax.set_ylim(0, 128)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('MIDI Note')
        vis_path = Path(midi_path).with_suffix('.png')
        plt.savefig(str(vis_path), bbox_inches='tight', dpi=200)
        plt.close(fig)
        return vis_path
    except Exception as e:
        logging.error(f"Failed to generate visualization for {midi_path}: {str(e)}")
        return None