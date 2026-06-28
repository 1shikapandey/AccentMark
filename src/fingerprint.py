import numpy as np

def build_fingerprint_matrix(all_speaker_means, reference_means, speaker_ids):
    """
    Finds phonemes common across ALL speakers in speaker_ids and the reference_means,
    computes the deviation vector (speaker_phoneme_mean - reference_phoneme_mean) 
    for each common phoneme, and concatenates them into a single fingerprint vector per speaker.
    
    Parameters:
        all_speaker_means (dict): Dictionary mapping speaker_id to a dictionary of {phoneme: mean_vector}
        reference_means (dict): Dictionary mapping phoneme to mean_vector for native English reference
        speaker_ids (list): List of speaker IDs to construct fingerprints for
        
    Returns:
        tuple: (matrix of shape [n_speakers, fingerprint_dim], list of common phonemes)
    """
    # 1. Find common phonemes across all speakers and the reference
    common_phonemes = set(reference_means.keys())
    for spk in speaker_ids:
        if spk in all_speaker_means:
            common_phonemes = common_phonemes.intersection(all_speaker_means[spk].keys())
        else:
            print(f"Warning: Speaker '{spk}' not found in all_speaker_means dict.")
            
    common_phonemes = sorted(list(common_phonemes))
    
    if not common_phonemes:
        raise ValueError("Error: Zero common phonemes found across speakers and reference.")
        
    n_speakers = len(speaker_ids)
    # Each phoneme has a feature vector of length 17
    feature_len = 17
    fingerprint_dim = len(common_phonemes) * feature_len
    
    matrix = np.zeros((n_speakers, fingerprint_dim), dtype=np.float32)
    
    for idx, spk in enumerate(speaker_ids):
        deviations = []
        for phoneme in common_phonemes:
            spk_mean = all_speaker_means[spk][phoneme]
            ref_mean = reference_means[phoneme]
            deviation = spk_mean - ref_mean
            deviations.append(deviation)
            
        # Concatenate deviations into one flat vector for the speaker
        matrix[idx] = np.concatenate(deviations)
        
    return matrix, common_phonemes
