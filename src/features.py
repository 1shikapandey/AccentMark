import librosa
import numpy as np
from collections import defaultdict
from src.utils import parse_textgrid

def extract_phoneme_features(audio_path, start_time, end_time, sr=16000):
    """
    Loads a specific segment of audio and extracts phoneme features.
    
    Parameters:
        audio_path (str): Path to WAV file.
        start_time (float): Start of phoneme interval.
        end_time (float): End of phoneme interval.
        sr (int): Sampling rate.
        
    Returns:
        np.ndarray: Vector of length 17, or None if segment is too short.
    """
    duration = end_time - start_time
    if duration <= 0:
        return None
        
    try:
        # Load only the portion of the audio that contains the phoneme
        # Using offset and duration to load only the segment to save memory/time
        y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
    except Exception as e:
        # Silently catch loading issues
        return None
        
    if len(y) < 100:
        return None
        
    # 1. Duration (1)
    # 2. RMS Energy (1)
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    
    # 3. Spectral Centroid (1)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent_mean = float(np.mean(spec_cent))
    
    # 4. Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))
    
    # 5. MFCCs (13)
    # Using 13 coefficients as requested
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1) # Shape: (13,)
    
    # Construct feature array of length 17
    feature_vector = np.zeros(17, dtype=np.float32)
    feature_vector[0] = duration
    feature_vector[1] = rms_mean
    feature_vector[2] = spec_cent_mean
    feature_vector[3] = zcr_mean
    feature_vector[4:17] = mfccs_mean
    
    return feature_vector

def extract_speaker_phoneme_means(speaker_files, max_utterances=150):
    """
    Extracts and averages feature vectors per phoneme for a single speaker.
    
    Parameters:
        speaker_files (list of tuple): List of (wav_path, textgrid_path)
        max_utterances (int): Limit processing to this many utterances.
        
    Returns:
        dict: {phoneme_string: np.ndarray} for phonemes with >= 5 occurrences.
    """
    phoneme_data = defaultdict(list)
    files_to_process = speaker_files[:max_utterances]
    total_files = len(files_to_process)
    
    for idx, (wav_path, tg_path) in enumerate(files_to_process):
        intervals = parse_textgrid(tg_path)
        for interval in intervals:
            start = interval['start']
            end = interval['end']
            phoneme = interval['phoneme']
            
            feat = extract_phoneme_features(wav_path, start, end)
            if feat is not None:
                phoneme_data[phoneme].append(feat)
                
        # Print progress every 50 utterances
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {total_files} utterances...")
            
    # Print final notification if not divisible by 50
    if total_files % 50 != 0:
        print(f"Processed {total_files} / {total_files} utterances...")
        
    # Calculate means
    phoneme_means = {}
    for phoneme, vectors in phoneme_data.items():
        if len(vectors) >= 5:
            # Average along the feature vector axis (axis=0)
            phoneme_means[phoneme] = np.mean(vectors, axis=0)
            
    return phoneme_means
