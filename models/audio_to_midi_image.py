import os
import shutil
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
from basic_pitch_torch.inference import predict


class AudioToMidiImage:
    def __init__(self, output_dir: str = "data/temp_stems"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Check if GPU is available for Demucs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"VERITAS: AudioToMidiImage initialized on {self.device}")

    def separate_stems(self, audio_path: str) -> dict:
        """
        Uses Demucs to separate audio into all stems.
        Returns dict of stem_name -> stem_audio_path
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Input file not found: {audio_path}")

        print(f"--> [Demucs] Separating all stems for: {os.path.basename(audio_path)}")

        cmd = [
            "demucs",
            "-n",
            "htdemucs",  # Use 4-stem model
            "-d",
            self.device,
            "-o",
            self.output_dir,
            audio_path,
        ]

        subprocess.run(cmd, check=True)

        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        stems_dir = os.path.join(self.output_dir, "htdemucs", song_name)

        # Demucs outputs: vocals.wav, drums.wav, bass.wav, other.wav (piano/guitar)
        stem_names = ["vocals", "drums", "bass", "other"]
        stems = {}

        for stem_name in stem_names:
            stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
            else:
                print(f"Warning: Stem {stem_name} not found at {stem_path}")

        if not stems:
            raise RuntimeError(f"Demucs failed to generate any stems in {stems_dir}")

        print(f"--> [Demucs] Generated {len(stems)} stems: {list(stems.keys())}")
        return stems

    def stems_to_midi(self, stems: dict, output_dir: str = None) -> dict:
        """
        Process each stem to generate MIDI files and piano roll images.
        Returns dict of stem_name -> {'midi': path, 'image': path}
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "midi")
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        for stem_name, stem_path in stems.items():
            print(f"--> [MIDI] Processing {stem_name} stem...")

            try:
                # Use Basic Pitch PyTorch to get MIDI data
                _, midi_data, _ = predict(
                    stem_path,
                    onset_threshold=0.5,
                    frame_threshold=0.3
                )

                # Create MIDI file
                midi = pretty_midi.PrettyMIDI()
                piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
                piano = pretty_midi.Instrument(program=piano_program)

                # Convert Basic Pitch output to MIDI notes
                for note in midi_data:
                    start_time = note[0]  # onset time
                    end_time = note[1]    # offset time
                    pitch = int(note[2])  # MIDI pitch
                    velocity = int(note[3] * 127)  # velocity 0-127

                    midi_note = pretty_midi.Note(
                        velocity=max(1, velocity),  # ensure velocity > 0
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    piano.notes.append(midi_note)

                midi.instruments.append(piano)

                # Save MIDI file
                midi_filename = f"{stem_name}.mid"
                midi_path = os.path.join(output_dir, midi_filename)
                midi.write(midi_path)

                # Generate piano roll image
                image_path = self.midi_to_image(midi_path)

                results[stem_name] = {
                    'midi': midi_path,
                    'image': image_path
                }
                print(f"--> [MIDI] Saved {stem_name} MIDI and image")

            except Exception as e:
                print(f"--> [MIDI] Failed to process {stem_name}: {e}")
                continue

        return results

    def midi_to_image(self, midi_path: str, output_path: str = None) -> str:
        """
        Convert MIDI file to piano roll image.
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(midi_path))[0]
            output_path = os.path.join(os.path.dirname(midi_path), f"{base_name}.png")

        # Load MIDI
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Get piano roll (88 keys, time in seconds)
        piano_roll = midi_data.get_piano_roll(fs=10)  # 10 Hz resolution

        # Create visualization
        plt.figure(figsize=(12, 4))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest')
        plt.title(f"MIDI Piano Roll: {os.path.basename(midi_path)}")
        plt.xlabel("Time (frames)")
        plt.ylabel("MIDI Pitch")
        plt.colorbar(label="Velocity")
        plt.tight_layout()

        # Save image
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"--> [Image] Saved piano roll to {output_path}")
        return output_path

    def cleanup(self) -> None:
        """Optional: Clear temp files to save space."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print("--> Cleanup complete.")