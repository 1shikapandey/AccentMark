import os
import textgrid

def parse_textgrid(textgrid_path):
    """
    Opens a .TextGrid file and extracts phoneme intervals from the 'phones' or 'phoneme' tier.
    
    Parameters:
        textgrid_path (str): Path to the .TextGrid file.
        
    Returns:
        list of dict: List of intervals formatted as {'start': float, 'end': float, 'phoneme': str}
    """
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except Exception as e:
        print(f"Error reading TextGrid file {textgrid_path}: {e}")
        return []
        
    # Find the tier representing phones or phonemes
    target_tier = None
    for tier in tg:
        if tier.name.lower() in ['phones', 'phoneme', 'phone']:
            target_tier = tier
            break
            
    # Fallback to the first interval tier if no matching name found
    if target_tier is None and len(tg) > 0:
        target_tier = tg[0]
        
    intervals = []
    if target_tier is not None:
        for interval in target_tier:
            phoneme = interval.mark.strip().upper()
            # Filter out silence and non-speech markers
            if phoneme in ['', 'SP', 'SIL', 'SPN', 'SILENCE']:
                continue
            intervals.append({
                'start': float(interval.minTime),
                'end': float(interval.maxTime),
                'phoneme': phoneme
            })
            
    return intervals

def get_speaker_files(data_dir, speaker_id):
    """
    Finds and returns matched pairs of (wav_path, textgrid_path) for all files 
    for the specified speaker_id in the wav/ and annotation/ directories.
    
    Parameters:
        data_dir (str): Root data directory.
        speaker_id (str): ID of the speaker.
        
    Returns:
        list of tuple: (wav_path, textgrid_path)
    """
    speaker_dir = os.path.join(data_dir, speaker_id)
    wav_dir = os.path.join(speaker_dir, 'wav')
    annotation_dir = os.path.join(speaker_dir, 'annotation')
    
    pairs = []
    if not os.path.exists(wav_dir) or not os.path.exists(annotation_dir):
        return pairs
        
    # Scan wav directory and pair with corresponding TextGrid
    for item in os.listdir(wav_dir):
        if item.lower().endswith('.wav'):
            base_name = os.path.splitext(item)[0]
            
            # The TextGrid files might have .TextGrid or .textgrid extensions
            tg_path = os.path.join(annotation_dir, base_name + '.TextGrid')
            if not os.path.exists(tg_path):
                tg_path = os.path.join(annotation_dir, base_name + '.textgrid')
                
            wav_path = os.path.join(wav_dir, item)
            
            if os.path.exists(tg_path):
                pairs.append((wav_path, tg_path))
                
    # Sort for deterministic processing
    pairs.sort()
    return pairs
