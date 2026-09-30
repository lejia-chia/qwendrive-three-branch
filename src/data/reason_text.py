"""Current natural-language pseudo-labels; no behaviour suffix or GT input."""
from pathlib import Path
import hashlib
import json
import numpy as np
from src.utils.paths import DATA
from src.utils.hashing import sha256_file

def prompt_ids_sha256(prompt_ids):
    return hashlib.sha256(np.asarray(prompt_ids,dtype=np.int64).tobytes()).hexdigest()

class ReasonTextStore:
    def __init__(self, split):
        root=DATA / 'reason_text'
        self.manifest=json.loads((root/'manifest.json').read_text())
        if self.manifest.get('kind')!='reason_text' or not self.manifest.get('complete'):
            raise ValueError('incomplete natural-language pseudo-labels')
        info=self.manifest['splits'][split];path=root/info['file']
        if sha256_file(path)!=info['sha256']:raise ValueError('reason text digest differs')
        records=[json.loads(line) for line in path.open()]
        self.rows={r['token']:r for r in records}
        if len(self.rows)!=len(records) or len(records)!=info['rows']:
            raise ValueError('reason text duplicate token/count mismatch')
        if any(r['split']!=split for r in records):raise ValueError('reason text split mismatch')
    def __getitem__(self,token):return self.rows[token]
