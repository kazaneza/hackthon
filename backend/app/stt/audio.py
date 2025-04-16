import numpy as np
from dataclasses import dataclass
from ..config import RATE, SILENCE_THRESHOLD, SILENCE_DURATION, CHUNK_DURATION

@dataclass
class AudioBuffer:
    buffer: list
    last_chunk_time: float = 0
    silence_start: float = None
    
    def add_samples(self, samples: np.ndarray) -> bool:
        # Normalize audio to [-1, 1] range
        if len(samples) > 0:
            samples = samples / np.max(np.abs(samples))
        
        self.buffer.extend(samples)
        
        # Check for silence using RMS with normalized audio
        rms = np.sqrt(np.mean(np.square(samples)))
        is_silent = rms < SILENCE_THRESHOLD
        
        if is_silent:
            if self.silence_start is None:
                self.silence_start = len(self.buffer) / RATE
        else:
            self.silence_start = None
            
        # Check if we should process (either due to duration or silence)
        buffer_duration = len(self.buffer) / RATE
        should_process = (
            buffer_duration >= CHUNK_DURATION or
            (self.silence_start is not None and 
             buffer_duration - self.silence_start >= SILENCE_DURATION)
        )
        
        return should_process
        
    def get_audio_data(self):
        audio_data = np.array(self.buffer, dtype=np.float32)
        
        # Apply pre-processing
        if len(audio_data) > 0:
            # Normalize
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Apply simple noise reduction
            noise_floor = np.mean(np.abs(audio_data)) * 2
            audio_data[np.abs(audio_data) < noise_floor] = 0
            
        self.buffer = []
        self.silence_start = None
        return audio_data