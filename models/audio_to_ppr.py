import os
import shutil
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict


class AudioToPPR:
    def __init__(self, output_dir: str = "data/temp_stems"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Check if GPU is available for Demucs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"VERITAS: AudioToPPR initialized on {self.device}")

    def separate_stems(self, audio_path: str) -> str:
        """
        Uses Demucs to separate audio.
        We strictly want to remove drums to clean up the harmonic signal.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Input file not found: {audio_path}")

        print(f"--> [Demucs] Separating stems for: {os.path.basename(audio_path)}")

        cmd = [
            "demucs",
            "--two-stems",
            "drums",
            "-n",
            "htdemucs",
            "-d",
            self.device,
            "-o",
            self.output_dir,
            audio_path,
        ]

        subprocess.run(cmd, check=True)

        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        clean_path = os.path.join(self.output_dir, "htdemucs", song_name, "no_drums.wav")

        if not os.path.exists(clean_path):
            raise RuntimeError(f"Demucs failed to generate: {clean_path}")

        return clean_path

    def extract_soft_matrix(self, audio_path: str) -> np.ndarray:
        """
        Runs Basic Pitch to get the frame-level probability matrix.
        Returns: numpy array of shape (Time, 88)
        """
        print(f"--> [Basic Pitch] Extracting PPR from: {os.path.basename(audio_path)}")

        model_output, _, _ = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
        )

        if len(model_output.shape) == 3:
            matrix = model_output[0]
        else:
            matrix = model_output

        return matrix

    def visualize_ppr(self, ppr_matrix: np.ndarray, title: str = "Probabilistic Piano Roll") -> None:
        """
        Visualizes the heatmap for inspection.
        """
        plt.figure(figsize=(12, 4))
        plt.imshow(ppr_matrix.T, aspect="auto", origin="lower", cmap="inferno")
        plt.title(title)
        plt.xlabel("Time (Frames)")
        plt.ylabel("Pitch (MIDI Bins)")
        plt.colorbar(label="Activation Probability")
        plt.tight_layout()
        plt.show()

    def cleanup(self) -> None:
        """Optional: Clear temp files to save space."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print("--> Cleanup complete.")
"""
Audio to Piano Performance Representation (PPR) Conversion Module

This module handles the conversion of audio files to structured piano performance
representations suitable for music analysis and generation tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np


class AudioToPPR:
    """
    Converts audio files to Piano Performance Representation (PPR).
    
    This class orchestrates the full pipeline:
    1. Audio preprocessing and source separation
    2. Pitch detection and note extraction
    3. Conversion to structured PPR format
    
    Attributes:
        device (torch.device): Computing device (CPU/GPU)
        sample_rate (int): Audio sampling rate in Hz
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 44100,
        **kwargs
    ):
        """
        Initialize the AudioToPPR converter.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto)
            sample_rate: Target sample rate for audio processing
            **kwargs: Additional configuration parameters
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.sample_rate = sample_rate
        
        print(f"AudioToPPR initialized on {self.device}")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio as numpy array
        """
        # TODO: Implement audio loading with librosa
        raise NotImplementedError("Audio loading not yet implemented")
    
    def separate_sources(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform source separation on the audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of separated sources (vocals, drums, bass, other)
        """
        # TODO: Implement demucs source separation
        raise NotImplementedError("Source separation not yet implemented")
    
    def extract_notes(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract note events from audio using pitch detection.
        
        Args:
            audio: Input audio array (ideally isolated piano/melody)
            
        Returns:
            Array of note events (onset, offset, pitch, velocity)
        """
        # TODO: Implement basic-pitch note extraction
        raise NotImplementedError("Note extraction not yet implemented")
    
    def to_ppr(self, notes: np.ndarray) -> Dict[str, Any]:
        """
        Convert note events to Piano Performance Representation.
        
        Args:
            notes: Array of note events
            
        Returns:
            Structured PPR dictionary
        """
        # TODO: Implement PPR conversion
        raise NotImplementedError("PPR conversion not yet implemented")
    
    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Full pipeline: audio file -> PPR.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Piano Performance Representation
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Separate sources
        sources = self.separate_sources(audio)
        
        # Extract notes from relevant source(s)
        notes = self.extract_notes(sources.get('other', audio))
        
        # Convert to PPR
        ppr = self.to_ppr(notes)
        