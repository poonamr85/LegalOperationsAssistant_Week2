"""Local MiniLM ONNX inference, independent of any vector database SDK."""
from functools import lru_cache
from pathlib import Path
from threading import RLock
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request

MODEL_NAME = 'all-MiniLM-L6-v2'
DIMENSIONS = 384
MODEL_ID = 'all-MiniLM-L6-v2-onnx-mean-normalized-v1'
ASSET_URL = 'https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz'
ARCHIVE_SHA256 = '913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3'
HASHES = {'tokenizer.json': 'da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0',
          'model.onnx': '4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6'}
_lock = RLock()


def valid(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


@lru_cache(maxsize=1)
def model_directory():
    target = Path(os.getenv('MINILM_CACHE_DIR', str(Path(__file__).resolve().parents[1]/'storage/models/minilm')))
    target.mkdir(parents=True, exist_ok=True)
    with _lock:
        if all(valid(target/n, h) for n,h in HASHES.items()):
            return target
        # Reuse verified model assets from the original local installation, not its Python package.
        legacy = Path.home()/'.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx'
        if all(valid(legacy/n, h) for n,h in HASHES.items()):
            for n in HASHES: shutil.copyfile(legacy/n, target/n)
            return target
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'model.tar.gz'
            with urllib.request.urlopen(ASSET_URL, timeout=120) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
            if not valid(archive, ARCHIVE_SHA256):
                raise ValueError('MiniLM download checksum mismatch')
            with tarfile.open(archive) as tar:
                for n,h in HASHES.items():
                    member = tar.getmember('onnx/'+n)
                    if not member.isfile(): raise ValueError('Invalid model archive')
                    asset = Path(temp)/n
                    with tar.extractfile(member) as source, asset.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    if not valid(asset,h): raise ValueError('MiniLM asset checksum mismatch')
                    shutil.copyfile(asset,target/n)
        return target


@lru_cache(maxsize=1)
def counting_tokenizer():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_directory()/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


@lru_cache(maxsize=1)
def runtime():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_directory()/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(length=256,pad_id=0,pad_token='[PAD]')
    session = ort.InferenceSession(str(model_directory()/'model.onnx'), providers=['CPUExecutionProvider'])
    return tokenizer,session


def embed_texts(texts):
    import numpy as np
    if not texts: return []
    with _lock:
        tokenizer, session = runtime()
        encoded = tokenizer.encode_batch(texts)
        feeds = {'input_ids': np.array([e.ids for e in encoded],dtype=np.int64),
                 'attention_mask':np.array([e.attention_mask for e in encoded],dtype=np.int64),
                 'token_type_ids':np.array([e.type_ids for e in encoded],dtype=np.int64)}
        hidden = session.run(None,feeds)[0]
        mask=feeds['attention_mask'][...,None].astype(np.float32)
        pooled=(hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1e-9)
        vectors=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
        if vectors.shape != (len(texts),DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError('Invalid MiniLM embeddings')
        return vectors.tolist()
