#!/usr/bin/env python3
"""
Convert a Wikipedia Clickstream dump into the three pickle files
  clients_triehh.txt
  clients_sfp.txt
  word_frequencies.txt
that TrieHH / SFP simulations expect.

One synthetic “client” = one click (row) after optional down-sampling.
"""

import argparse, csv, gzip, os, pickle, random, collections, itertools

# ---------- helper -----------------------------------------------------------

def open_any(path, mode='rt', **kw):
    "Open .gz transparently."
    if path.endswith('.gz'):
        return gzip.open(path, mode, **kw)
    return open(path, mode, **kw)

def truncate_or_pad(s: str, L: int) -> str:
    return (s[:L]).ljust(L, '$')

def add_end_symbol(s: str) -> str:
    return s + '$'

def reservoir_extend(container, elem, k, rng):
    """Keep at most k copies of elem in container (reservoir sampling)."""
    while container[elem] >= k:
        # replace an old copy with prob k/(k+1) to keep expected count = k
        if rng.random() < k / (k + 1):
            return False
        container[elem] -= 1
    container[elem] += 1
    return True

# ---------- main -------------------------------------------------------------

def preprocess(path, max_len=16, cap=5000, keep_type='link', seed=0):
    rng = random.Random(seed)
    freq = collections.Counter()       # real counts (after cap)
    clients_triehh, clients_sfp = [], []

    with open_any(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader, 1):
            if i % 2_000_000 == 0:                 # every 2 M rows
                print(f"…parsed {i:,} lines", flush=True)
            if len(row) != 4:
                continue
            prev, curr, typ, cnt = row
            if keep_type and typ != keep_type:
                continue
            cnt = int(cnt)
            if cap and cnt > cap:
                cnt = cap

            # reservoir sampling to avoid extending huge lists at once
            if cap:
                for _ in range(cnt):
                    if reservoir_extend(freq, curr, cap, rng):
                        clients_triehh.append(add_end_symbol(curr))
                        clients_sfp.append(truncate_or_pad(curr, max_len))
            else:
                freq[curr] += cnt
                clients_triehh.extend([add_end_symbol(curr)] * cnt)
                clients_sfp.extend([truncate_or_pad(curr, max_len)] * cnt)

    total_clients = len(clients_triehh)
    rel_freq = {w: c / total_clients for w, c in freq.items()}

    print(f'Synthetic clients: {total_clients:,}')
    print(f'Unique tokens    : {len(freq):,}')

    with open('clients_triehh.txt', 'wb') as fp:
        pickle.dump(clients_triehh, fp)

    with open('clients_sfp.txt', 'wb') as fp:
        pickle.dump(clients_sfp, fp)

    with open('word_frequencies.txt', 'wb') as fp:
        pickle.dump(rel_freq, fp)

# ---------- CLI --------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Pre-process Wikipedia Clickstream for TrieHH/SFP.')
    ap.add_argument('--month', default='2025-03',
                    help='dump month folder, e.g. 2025-03')
    ap.add_argument('--lang', default='en', help='language code, e.g. en, de')
    ap.add_argument('--path', default=None,
                    help='explicit TSV(.gz) path; overrides --month/--lang')
    ap.add_argument('--max_len', type=int, default=16,
                    help='pad/cut length for SFP tokens')
    ap.add_argument('--cap', type=int, default=5000,
                    help='max duplicates per (prev,curr) row to keep')
    ap.add_argument('--seed', type=int, default=0, help='PRNG seed')

    args = ap.parse_args()

    if args.path:
        tsv_path = args.path
    else:
        fn = f'clickstream-{args.lang}wiki-{args.month}.tsv.gz'
        tsv_path = os.path.join(
            'clickstream', args.month, fn)  # change if your dumps dir differs
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    preprocess(tsv_path, max_len=args.max_len, cap=args.cap, seed=args.seed)
