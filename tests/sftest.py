from pathlib import Path
print("Looking for SoundFont...")
sf = Path("models/soundfont/FluidR3_GM.sf2")
print("Exists?" , sf.exists())
print("Full path:", sf.resolve())