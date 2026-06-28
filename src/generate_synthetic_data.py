import os
import wave
import math
import struct
import random

SPEAKERS = {
    'ABA': 'Arabic', 'YBAA': 'Arabic',
    'SVBI': 'Hindi', 'TNI': 'Hindi',
    'HJK': 'Korean', 'YDCK': 'Korean',
    'BWC': 'Mandarin', 'LXC': 'Mandarin',
    'EBVS': 'Spanish', 'ERMS': 'Spanish',
    'HQTV': 'Vietnamese', 'PNV': 'Vietnamese',
    'NJS': 'Native', 'TXHC': 'Native'
}

def generate_wav(filepath, duration, frequency, noise_level=0.1, sr=16000):
    """
    Generates a synthetic PCM 16-bit mono wav file with sine wave and noise components.
    """
    num_samples = int(duration * sr)
    data = []
    
    # Generate waveform: basic fundamental sine + 2nd harmonic + random noise
    for i in range(num_samples):
        t = i / sr
        val_sig = math.sin(2 * math.pi * frequency * t) + 0.3 * math.sin(2 * math.pi * 2 * frequency * t)
        val_noise = random.uniform(-1.0, 1.0) * noise_level
        val = int(32767 * 0.4 * (val_sig + val_noise))
        val = max(-32768, min(32767, val)) # Clip limits
        data.append(struct.pack('<h', val))
        
    with wave.open(filepath, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b''.join(data))

def create_textgrid(filepath, duration, phonemes_seq):
    """
    Writes a Praat-compliant TextGrid alignment file.
    """
    n_intervals = len(phonemes_seq)
    interval_dur = duration / n_intervals
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write('xmin = 0.0\n')
        f.write(f'xmax = {duration:.3f}\n')
        f.write('tiers? <exists>\n')
        f.write('size = 1\n')
        f.write('item []:\n')
        f.write('    item [1]:\n')
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "phones"\n')
        f.write('        xmin = 0.0\n')
        f.write(f'xmax = {duration:.3f}\n')
        f.write(f'        intervals: size = {n_intervals}\n')
        
        for i, ph in enumerate(phonemes_seq):
            xmin = i * interval_dur
            xmax = (i + 1) * interval_dur
            f.write(f'        intervals [{i+1}]:\n')
            f.write(f'            xmin = {xmin:.3f}\n')
            f.write(f'            xmax = {xmax:.3f}\n')
            f.write(f'            mark = "{ph}"\n')

def generate_all_synthetic_data(data_dir, num_utterances=15):
    """
    Generates a full mock dataset structured identical to L2-ARCTIC.
    """
    print(f"Generating synthetic dataset inside {data_dir}...")
    
    # 10 target phonemes we guarantee to exist with >= 5 occurrences
    target_phonemes = ['AA', 'IY', 'UW', 'EH', 'AH', 'M', 'N', 'T', 'S', 'K']
    
    # Acoustic configuration variations based on L1 language to create clustering
    l1_configs = {
        'Arabic': {'base_freq': 220, 'noise': 0.15},
        'Hindi': {'base_freq': 320, 'noise': 0.05},
        'Korean': {'base_freq': 420, 'noise': 0.22},
        'Mandarin': {'base_freq': 520, 'noise': 0.12},
        'Spanish': {'base_freq': 620, 'noise': 0.08},
        'Vietnamese': {'base_freq': 720, 'noise': 0.26},
        'Native': {'base_freq': 820, 'noise': 0.02}
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    for spk, l1 in SPEAKERS.items():
        spk_dir = os.path.join(data_dir, spk)
        wav_dir = os.path.join(spk_dir, 'wav')
        ann_dir = os.path.join(spk_dir, 'annotation')
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        config = l1_configs[l1]
        
        # Add a speaker-specific frequency offset
        spk_idx = list(SPEAKERS.keys()).index(spk)
        spk_freq = config['base_freq'] + (spk_idx * 5)
        
        print(f"  Generating {num_utterances} utterances for speaker '{spk}' ({l1})...")
        for u in range(num_utterances):
            file_id = f"b{u+1:04d}"
            wav_path = os.path.join(wav_dir, f"{file_id}.wav")
            tg_path = os.path.join(ann_dir, f"{file_id}.TextGrid")
            
            # Deterministic round-robin selection to guarantee uniform phoneme occurrences
            mid_phonemes = [
                target_phonemes[(u * 3) % 10],
                target_phonemes[(u * 3 + 1) % 10],
                target_phonemes[(u * 3 + 2) % 10],
                target_phonemes[(u * 3 + 3) % 10]
            ]
            
            phoneme_seq = ['sil'] + mid_phonemes + ['sp']
            duration = random.uniform(1.8, 2.4)
            
            generate_wav(wav_path, duration, spk_freq, config['noise'])
            create_textgrid(tg_path, duration, phoneme_seq)
            
    print("Synthetic dataset generation complete!")

if __name__ == '__main__':
    # Default path setting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data')
    generate_all_synthetic_data(data_path)
