"""Encode DNA strings and collate variable-length sequence pairs."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DNA_TABLE = np.full(256, 4, dtype=np.uint8)
for _index, _base in enumerate('ACGT'):
    DNA_TABLE[ord(_base)] = _index
COMPLEMENT = torch.tensor([3, 2, 1, 0, 4])


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode A,C,G,T,N as 0,1,2,3,4; accept either letter case."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError('Sequences must be nonempty strings')
    normalized = sequence.upper()
    if set(normalized) - set('ACGTN'):
        raise ValueError('Sequences must contain only A, C, G, T, and N')
    return DNA_TABLE[np.frombuffer(normalized.encode('ascii'), dtype=np.uint8)].copy()


class PairDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.ref = [encode_sequence(s) for s in self.frame.ref_seq]
        self.alt = [encode_sequence(s) for s in self.frame.alt_seq]
        if any(len(r) != len(a) for r, a in zip(self.ref, self.alt)):
            raise ValueError('Reference and alternate sequences must have equal lengths per pair')
        self.delta = (self.frame.delta.to_numpy(dtype=np.float32) if 'delta' in self.frame
                      else np.full(len(self.frame), np.nan, dtype=np.float32))
        self.activity = self.frame.reindex(columns=['y_ref','y_alt']).to_numpy(dtype=np.float32)
    def __len__(self): return len(self.frame)
    def __getitem__(self, i): return self.ref[i], self.alt[i], self.delta[i], self.activity[i], i



def collate_pairs(rows):
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    if any(len(row[0]) != len(row[1]) or len(row[0]) == 0 for row in rows):
        raise ValueError('Each pair requires equal, nonempty reference and alternate sequences')
    lengths = torch.tensor([len(r[0]) for r in rows], dtype=torch.long)
    ref = torch.full((len(rows),int(lengths.max())),4,dtype=torch.long); alt=ref.clone()
    for i,(r,a,_,_,_) in enumerate(rows):
        ref[i,:len(r)] = torch.from_numpy(r.astype(np.int64))
        alt[i,:len(a)] = torch.from_numpy(a.astype(np.int64))
    return {'ref':ref,'alt':alt,'lengths':lengths,
            'delta':torch.tensor(np.array([r[2] for r in rows])),
            'activity':torch.tensor(np.stack([r[3] for r in rows])),
            'index':torch.tensor([r[4] for r in rows])}



def loader(frame,batch_size,shuffle=False,seed=0):
    ds=frame if isinstance(frame,PairDataset) else PairDataset(frame)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_pairs,
                      num_workers=0,pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))



def device_batch(batch,device,augment=False):
    out={k:v.to(device,non_blocking=True) for k,v in batch.items() if k!='index'}
    if augment:
        selected=torch.rand(len(out['lengths']),device=device)<.5
        positions=torch.arange(out['ref'].shape[1],device=device)[None,:]
        reverse_index=(out['lengths'][:,None]-1-positions).clamp_min(0)
        valid=positions<out['lengths'][:,None]; complement=COMPLEMENT.to(device)
        for key in ('ref','alt'):
            reverse=complement[out[key].gather(1,reverse_index)].masked_fill(~valid,4)
            out[key]=torch.where(selected[:,None],reverse,out[key])
    return out

