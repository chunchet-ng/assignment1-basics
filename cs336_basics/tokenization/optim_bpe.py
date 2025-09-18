import regex as re
from collections import Counter, defaultdict
from typing import Tuple
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_basic_vocab(special_tokens: list[str]):
    basic_vocab = {}
    # Add all 256 byte values
    for token in range(256):
        basic_vocab[token] = bytes([token])
    
    # Add special tokens
    for i, token in enumerate(special_tokens):
        token_id = 256 + i
        basic_vocab[token_id] = token.encode("utf-8")
    return basic_vocab

def word2bytes(word):
    """Convert word string to tuple of bytes"""
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def count_word(text: str):
    """Count word bytes frequency using GPT2 pattern"""
    word_cnt = defaultdict(int)
    regex_pattern = re.compile(PAT)
    for m in regex_pattern.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt

def count_pair(word_cnt):
    """Count byte pairs from word counts"""
    pair_cnt = defaultdict(int)
    for word_bytes, cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_cnt[pair] += cnt
    return pair_cnt

def get_max_pair(pair_cnt):
    """Get the most frequent pair with lexicographic tie-breaking"""
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))
    return max_pair

def apply_merge(word_bytes, merge):
    """Apply merge to a word"""
    merged = merge[0] + merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update_cnt(word_cnt, pair_cnt, merge_pair):
    """Efficiently update word and pair counts after a merge"""
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt)  # copy with defaultdict

    for word_bytes, cnt in word_cnt.items():
        # ----------for word cnt ---------------
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes] += cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes, merge_pair)
        new_word_cnt[new_word] += cnt

        # --------for pair cnt ----------------
        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt, new_pair_cnt

def remove_special_tokens_and_pretokenize(text: str, special_tokens: list[str]):
    if not special_tokens:
        chunks = [text]
    else:
        # Sort by descending length to prioritize longer tokens
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in sorted_tokens)
        pattern = re.compile(pattern)
        chunks = pattern.split(text)
        chunks = [c for c in chunks if c and c not in special_tokens]  # remove empty strings and special tokens
    return chunks

def _worker_count_slice(args):
    path, start, end, specials = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    # The starter suggests errors="ignore" — simplest & safe with boundary-aligned chunks
    text = data.decode("utf-8", errors="ignore")

    # Remove specials within the slice, then PAT-pretokenize and count
    word_cnt = defaultdict(int)
    for sub in remove_special_tokens_and_pretokenize(text, specials):
        d = count_word(sub)
        for k, v in d.items():
            word_cnt[k] += v
    return word_cnt

def train_bpe_optim(input_path: str, vocab_size: int, special_tokens: list[str]):
    procs = 4
    boundary_token = "<|endoftext|>"
    # Ensure the boundary token is removed inside slices as well
    specials = list(special_tokens)
    if boundary_token not in specials:
        specials.append(boundary_token)
        
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, procs, boundary_token.encode("utf-8"))
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    args = [(input_path, s, e, specials) for (s, e) in spans]
    with Pool(procs) as pool:
        parts = pool.map(_worker_count_slice, args)
    word_cnt = defaultdict(int)
    for d in parts:
        for k, v in d.items():
            word_cnt[k] += v
    
    # Count pairs
    pair_cnt = count_pair(word_cnt)
    
    # Initialize vocabulary
    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size
    
    if n_merges <= 0:
        return vocab, []
    
    # Perform BPE merges using the correct approach
    merges = []
    for i in range(n_merges):
        if not pair_cnt:
            break
            
        max_pair = get_max_pair(pair_cnt)
        vocab[base_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        word_cnt, pair_cnt = update_cnt(word_cnt, pair_cnt, max_pair)
    
    return vocab, merges
