# main.py
import sys
import os
import time
import subprocess
import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QLabel, QVBoxLayout, QTableWidgetItem, QDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi

import src.audio_processing as audio_processing
import src.visualization as visualization
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import get_resource_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi(str(get_resource_path("interface.ui")), self)

        # Make "MIDI Program Change" label clickable
        self.label_4.setStyleSheet("QLabel { color: blue; text-decoration: underline; }")
        self.label_4.setCursor(Qt.PointingHandCursor)

        def open_gm_dialog():
            dialog = GMInstrumentDialog(self)
            dialog.exec_()

        self.label_4.mousePressEvent = lambda event: open_gm_dialog()

        self.pushButton_import.clicked.connect(self.import_file)
        self.pushButton_transcribe.clicked.connect(self.transcribe)

        self.checkBox_drums.toggled.connect(self.toggle_drums)
        self.checkBox_bass.toggled.connect(self.toggle_bass)
        self.checkBox_piano.toggled.connect(self.toggle_piano)

        self.toggle_drums(self.checkBox_drums.isChecked())
        self.toggle_bass(self.checkBox_bass.isChecked())
        self.toggle_piano(self.checkBox_piano.isChecked())

        self.audio_file = None

    def toggle_drums(self, checked):
        self.comboBox_drums.setEnabled(checked)

    def toggle_bass(self, checked):
        self.comboBox_bass.setEnabled(checked)
        self.spinBox_bass.setEnabled(checked)

    def toggle_piano(self, checked):
        self.comboBox_piano.setEnabled(checked)
        self.spinBox_piano.setEnabled(checked)

    def import_file(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select Audio File', '', 'Audio files (*.wav *.mp3)')
        if file:
            self.audio_file = file
            self.label_fileSelect.setText(f"File selected: {Path(file).name}")

    def transcribe(self):
        if not self.audio_file:
            QMessageBox.warning(self, "No File", "Please select an audio file first.")
            return

        checked = self.checkBox_drums.isChecked() or self.checkBox_bass.isChecked() or self.checkBox_piano.isChecked()
        if not checked:
            QMessageBox.warning(self, "No Instruments", "At least one instrument must be selected.")
            return

        song_name = Path(self.audio_file).stem
        self.progressTooltip.setText("Running stem separation with Demucs...")
        self.progressBar.setValue(10)
        QApplication.processEvents()

        try:
            stems = audio_processing.run_demucs(self.audio_file)
        except Exception as e:
            logging.error(f"Stem separation failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to separate stems: {str(e)}")
            self.progressTooltip.setText("Stem separation failed.")
            self.progressBar.setValue(0)
            return

        self.progressTooltip.setText("Stem separation complete. Starting transcription...")
        self.progressBar.setValue(30)
        QApplication.processEvents()

        midis = {}
        midi_dir = Path("output") / song_name / "MIDI_exports"
        midi_dir.mkdir(parents=True, exist_ok=True)

        if self.checkBox_drums.isChecked():
            self.progressTooltip.setText("Transcribing drums...")
            QApplication.processEvents()
            model = self.comboBox_drums.currentText()
            try:
                midis['drums'] = audio_processing.transcribe_drums(stems['drums'], model, midi_dir / "drums.mid")
            except Exception as e:
                logging.error(f"Drum transcription failed: {str(e)}")
                QMessageBox.warning(self, "Error", f"Drum transcription failed: {str(e)}")
                self.progressBar.setValue(50)

        if self.checkBox_bass.isChecked():
            self.progressTooltip.setText("Transcribing bass...")
            QApplication.processEvents()
            model = self.comboBox_bass.currentText()
            program = self.spinBox_bass.value()
            try:
                midis['bass'] = audio_processing.transcribe_bass(stems['bass'], model, program, midi_dir / "bass.mid")
            except Exception as e:
                logging.error(f"Bass transcription failed: {str(e)}")
                QMessageBox.warning(self, "Error", f"Bass transcription failed: {str(e)}")
                self.progressBar.setValue(70)

        if self.checkBox_piano.isChecked():
            self.progressTooltip.setText("Transcribing piano/others...")
            QApplication.processEvents()
            model = self.comboBox_piano.currentText()
            program = self.spinBox_piano.value()
            try:
                midis['piano'] = audio_processing.transcribe_piano(stems['other'], model, program, midi_dir / "piano.mid")
                logging.info(f"Piano MIDI generated: {midis['piano']}")
            except Exception as e:
                logging.error(f"Piano transcription failed for {stems['other']}: {str(e)}")
                QMessageBox.warning(self, "Error", f"Piano transcription failed: {str(e)}")
                self.progressBar.setValue(90)

        self.progressBar.setValue(100)
        self.progressTooltip.setText("Transcription complete." if any(midis.values()) else "Transcription failed for all instruments.")
        QApplication.processEvents()

        if any(midis.values()):
            trans_win = TranscriptionWindow(song_name, midis, stems)
            trans_win.show()
        else:
            QMessageBox.critical(self, "Error", "No MIDI files were generated successfully.")


class TranscriptionWindow(QMainWindow):
    def __init__(self, song_name, midis, stems):
        super(TranscriptionWindow, self).__init__()
        loadUi(str(get_resource_path("transcription.ui")), self)
        self.setWindowTitle(song_name)

        self.midis = midis
        self.stems = stems
        self.output_dir = Path("output") / song_name
        self.midi_dir = self.output_dir / "MIDI_exports"
        self.merged_wav_path = None

        # ------------------------------------------------------------------
        # Helper: detect audible content
        # ------------------------------------------------------------------
        def has_content(path, threshold=0.01):
            try:
                y, _ = librosa.load(str(path), sr=None)
                rms = librosa.feature.rms(y=y)[0]
                return np.mean(rms) > threshold
            except Exception as e:
                logging.error(f"Failed to analyze content for {path}: {str(e)}")
                return False

        # ------------------------------------------------------------------
        # 1. Ensure waveform widgets have layouts
        # ------------------------------------------------------------------
        def _ensure_layout(widget):
            if not widget.layout():
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
                widget.setLayout(layout)

        for w in (self.widget_vocals, self.widget_vocals_merged):
            _ensure_layout(w)

        # ------------------------------------------------------------------
        # 2. Instrument Tabs (Drums, Bass, Piano)
        # ------------------------------------------------------------------
        show_drums = 'drums' in self.midis and self.midis['drums'] and Path(self.midis['drums']).exists()
        if show_drums:
            vis_path = visualization.generate_midi_vis(self.midis['drums'])
            if vis_path:
                self.display_vis(self.widget_drums, vis_path)
            self.label_pathToDrums.setText(str(Path(self.midis['drums']).absolute()))
            self.pushButton_openDrumsWav.clicked.connect(lambda: os.startfile(str(self.stems['drums'])))
            self.pushButton_openDrumsMidi.clicked.connect(lambda: os.startfile(str(self.midis['drums'])))
            self.pushButton_toDrums.clicked.connect(lambda: os.startfile(str(self.output_dir)))
        self.tabWidget.setTabVisible(0, show_drums)

        show_bass = 'bass' in self.midis and self.midis['bass'] and Path(self.midis['bass']).exists()
        if show_bass:
            vis_path = visualization.generate_midi_vis(self.midis['bass'])
            if vis_path:
                self.display_vis(self.widget_bass, vis_path)
            self.label_pathToBass.setText(str(Path(self.midis['bass']).absolute()))
            self.pushButton_openBassWav.clicked.connect(lambda: os.startfile(str(self.stems['bass'])))
            self.pushButton_openBassMidi.clicked.connect(lambda: os.startfile(str(self.midis['bass'])))
            self.pushButton_toBass.clicked.connect(lambda: os.startfile(str(self.output_dir)))
        self.tabWidget.setTabVisible(1, show_bass)

        show_piano = 'piano' in self.midis and self.midis['piano'] and Path(self.midis['piano']).exists()
        if show_piano:
            vis_path = visualization.generate_midi_vis(self.midis['piano'])
            if vis_path:
                self.display_vis(self.widget_piano, vis_path)
            self.label_pathToPiano.setText(str(Path(self.midis['piano']).absolute()))
            self.pushButton_openPianoWav.clicked.connect(lambda: os.startfile(str(self.stems['other'])))
            self.pushButton_openPianoMidi.clicked.connect(lambda: os.startfile(str(self.midis['piano'])))
            self.pushButton_toPiano.clicked.connect(lambda: os.startfile(str(self.output_dir)))
        self.tabWidget.setTabVisible(2, show_piano)

        # ------------------------------------------------------------------
        # 3. Vocals Tab
        # ------------------------------------------------------------------
        show_vocals = has_content(self.stems['vocals'])
        if show_vocals:
            self.display_waveform(self.widget_vocals, self.stems['vocals'])
            self.label_pathToVocals.setText(str(Path(self.stems['vocals']).absolute()))
            self.pushButton_openVocalsWav.clicked.connect(lambda: os.startfile(str(self.stems['vocals'])))
            self.pushButton_toVocals.clicked.connect(lambda: os.startfile(str(self.output_dir)))
        self.tabWidget.setTabVisible(3, show_vocals)

        # ------------------------------------------------------------------
        # 4. Populate ComboBox
        # ------------------------------------------------------------------
        self.comboBox.clear()
        self.instrument_map = {}
        for key, display in [('drums', 'drums.mid'), ('bass', 'bass.mid'), ('piano', 'piano.mid')]:
            if key in self.midis and Path(self.midis[key]).exists():
                self.comboBox.addItem(display)
                self.instrument_map[display] = self.midis[key]

        # ------------------------------------------------------------------
        # 5. Disable merge controls
        # ------------------------------------------------------------------
        self.pushButton_openMergedVocalsWav.setEnabled(False)
        self.pushButton_toMergedVocals.setEnabled(False)
        self.label_pathToMergedVocals.setText("No merged file yet")

        # ------------------------------------------------------------------
        # 6. Connect Merge Button
        # ------------------------------------------------------------------
        self.pushButton_merge.clicked.connect(self.merge_vocals)

    # ----------------------------------------------------------------------
    # Display MIDI visualization
    # ----------------------------------------------------------------------
    def display_vis(self, widget, vis_path):
        label = QLabel(widget)
        pix = QPixmap(str(vis_path))
        label.setPixmap(pix.scaled(widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setScaledContents(True)
        label.resize(widget.size())

    # ----------------------------------------------------------------------
    # Display waveform with file polling
    # ----------------------------------------------------------------------
    def display_waveform(self, widget, wav_path):
        # Clear previous content
        layout = widget.layout()
        while layout.count():
            child = layout.takeAt(0).widget()
            if child:
                child.deleteLater()

        # Generate waveform PNG
        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            fig, ax = plt.subplots(figsize=(8.5, 1.8), dpi=100)
            times = np.linspace(0, len(y)/sr, num=len(y))
            ax.plot(times, y, color='#333333', linewidth=0.5)
            ax.axis('off')
            fig.tight_layout(pad=0)

            png_path = Path(wav_path).with_suffix('.waveform.png')
            fig.savefig(str(png_path), bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
        except Exception as e:
            logging.error(f"Waveform generation failed: {e}")
            return

        # Poll for file to be fully written
        max_wait = 3.0
        waited = 0
        while waited < max_wait:
            if png_path.exists() and png_path.stat().st_size > 1000:
                break
            time.sleep(0.1)
            waited += 0.1
        else:
            logging.warning(f"PNG took too long to write: {png_path}")

        # Load and display
        label = QLabel()
        pix = QPixmap(str(png_path))
        if pix.isNull():
            logging.error(f"Failed to load PNG: {png_path}")
            return

        label.setPixmap(pix)
        label.setScaledContents(True)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        widget.update()

    # ----------------------------------------------------------------------
    # Merge selected MIDI with vocals
    # ----------------------------------------------------------------------
    def merge_vocals(self):
        if self.comboBox.count() == 0:
            QMessageBox.warning(self, "No MIDI", "No instrument MIDI available to merge.")
            return

        selected = self.comboBox.currentText()
        if not selected:
            QMessageBox.warning(self, "Select", "Please select an instrument to merge.")
            return

        midi_path = self.instrument_map[selected]
        vocals_path = self.stems['vocals']

        # 1. Synthesize MIDI to WAV
        synth_wav = self.output_dir / f"{Path(midi_path).stem}_synth.wav"
        try:
            audio_processing.synthesize_midi(str(midi_path), str(synth_wav))
        except Exception as e:
            QMessageBox.critical(self, "Synthesis Failed", f"Could not synthesize MIDI:\n{e}")
            return

        # 2. Mix with vocals
        try:
            merged, sr = audio_processing.merge_audio([str(synth_wav), str(vocals_path)])
            self.merged_wav_path = self.output_dir / f"{Path(midi_path).stem}_with_vocals.wav"
            audio_processing.save_audio(merged, sr, str(self.merged_wav_path))
        except Exception as e:
            QMessageBox.critical(self, "Mix Failed", f"Could not mix audio:\n{e}")
            return
        finally:
            try: synth_wav.unlink()
            except: pass

        # 3. Update UI
        self.label_pathToMergedVocals.setText(str(self.merged_wav_path.absolute()))
        self.display_waveform(self.widget_vocals_merged, self.merged_wav_path)

        self.pushButton_openMergedVocalsWav.setEnabled(True)
        self.pushButton_toMergedVocals.setEnabled(True)

        self.pushButton_openMergedVocalsWav.clicked.connect(lambda: os.startfile(str(self.merged_wav_path)))
        self.pushButton_toMergedVocals.clicked.connect(lambda: os.startfile(str(self.output_dir)))

        QMessageBox.information(self, "Merge Complete", f"Merged file saved:\n{self.merged_wav_path.name}")


class GMInstrumentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        loadUi(str(get_resource_path("gm_instrument_list.ui")), self)

        self.setWindowTitle("General MIDI (GM) Instrument List")
        self.tableWidget.setHorizontalHeaderLabels(["MIDI No.", "Instrument Sounds"])

        gm_instruments = [
            "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
            "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
            "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
            "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
            "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
            "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
            "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
            "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
            "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
            "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
            "Violin", "Viola", "Cello", "Contrabass",
            "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
            "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
            "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
            "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
            "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
            "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
            "Oboe", "English Horn", "Bassoon", "Clarinet",
            "Piccolo", "Flute", "Recorder", "Pan Flute",
            "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
            "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
            "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
            "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
            "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
            "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
            "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
            "Sitar", "Banjo", "Shamisen", "Koto",
            "Kalimba", "Bagpipe", "Fiddle", "Shanai",
            "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
            "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
            "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
            "Telephone Ring", "Helicopter", "Applause", "Gunshot"
        ]
        
        for i, name in enumerate(gm_instruments):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(name))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setStretchLastSection(True)

        from PyQt5.QtWidgets import QPushButton
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl

        help_button = QPushButton("?", self)
        button_size = 20  # Square size in pixels
        help_button.setFixedSize(button_size, button_size)

        help_button.setStyleSheet(f"""
            QPushButton {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
                background-color: #f8f8f8;
                border: 2px solid #999;
                border-radius: 0px;   /* Small radius = soft square corners */
                /* Use border-radius: 0px; for completely sharp corners */
            }}
            QPushButton:hover {{
                background-color: #e8e8e8;
                border-color: #777;
            }}
            QPushButton:pressed {{
                background-color: #d8d8d8;
            }}
        """)

        margin = 0
        help_button.move(self.width() - button_size - margin, margin)

        def open_gm_wiki():
            QDesktopServices.openUrl(QUrl("https://en.wikipedia.org/wiki/General_MIDI"))

        help_button.clicked.connect(open_gm_wiki)
        help_button.setToolTip("Open General MIDI information on Wikipedia")

        # Keep button pinned to top-right when window is resized
        def resizeEvent(event):
            help_button.move(self.width() - button_size - margin, margin)
            super(GMInstrumentDialog, self).resizeEvent(event)

        self.resizeEvent = resizeEvent

        # Hide the confusing title bar "?" button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())