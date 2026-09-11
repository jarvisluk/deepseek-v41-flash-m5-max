"""Compare native SSD reader/expert/Engram execution with the resident model."""
import json
import tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
from test_parity import _write_tiny_release, VOCAB, rel_max_diff
from deepseek_v41_mlx.convert import convert
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.native_stream import NativeStream


with tempfile.TemporaryDirectory(prefix='v41-native-test-') as td:
    src, dst = Path(td)/'src', Path(td)/'dst'
    _write_tiny_release(str(src), np.random.default_rng(19))
    (src/'engram_token_map.json').write_text(json.dumps(list(range(VOCAB))))
    convert(str(src), str(dst))
    resident, _ = load(str(dst))
    resident.set_token_map(list(range(VOCAB)))
    cache = resident.make_cache(bsz=1, max_seq_len=64)
    streamed = NativeStream(src, 64)
    cached = NativeStream(src, 64, dense_cache_bytes=8*1024**2, aggressive_cleanup=False, compile_dequant=True)
    rng = np.random.default_rng(4)
    for count in (20, 1, 1, 3):
        ids = rng.integers(3, VOCAB, size=(1,count), dtype=np.int32)
        expected = np.array(resident(mx.array(ids), cache))[:, -1, :]
        actual = np.array(streamed(ids))
        error = rel_max_diff(actual, expected)
        print('native vs resident', count, error, flush=True)
        assert error < 1e-5, error
        assert np.argmax(actual) == np.argmax(expected)
        cached_actual = np.array(cached(ids))
        assert rel_max_diff(cached_actual, expected) < 1e-5
        assert np.array_equal(cached_actual, actual)
        assert cached.reader.cached_bytes <= 8*1024**2
    assert cached.reader.cache_hits > 0
    assert cached.reader.bytes_read < streamed.reader.bytes_read
    print('NATIVE STREAM PASS', flush=True)
