import sys, time, json
sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from common import *
from retrieve2 import Phoneme, realisations, ngrams, squash

t0 = time.perf_counter()
words = set()
for terms, _, _ in (pool_big(), pool_small()):
    for t in terms:
        for r in realisations(t):
            words.update(r.split())
for p in (ASR_NEW, ASR_OLD_UNSEEN, ASR_OLD_SEEN):
    for r in read_jsonl(p):
        for g in ngrams(r["hyp"], 1):
            words.add(g)
for r in read_jsonl(REAL_HOLDOUT):
    for g in ngrams(r["input"], 1):
        words.add(g)
    for g in ngrams(r["reference"], 1):
        words.add(g)
print(f"distinct words={len(words)} [{time.perf_counter()-t0:.0f}s]", flush=True)
ph = Phoneme()
n = ph.warm(words)
ph.save()
print(f"g2p warmed {n} new words, cache={len(ph.cache)} [{time.perf_counter()-t0:.0f}s]", flush=True)
