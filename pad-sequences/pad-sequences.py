import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    
    # If empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Determine max_len automatically
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    # Create padded array
    padded = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    # Fill sequences
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]  # truncate if longer
        padded[i, :len(trunc)] = trunc
    
    return padded