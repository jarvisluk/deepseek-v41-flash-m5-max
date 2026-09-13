"""Bounded-memory text inference directly from the original FP4/FP8 checkpoint.

Uses pread, never materializes a whole expert bank or Engram table, and does
not requantize weights. The validated MLX port supplies the transformer math.
This is a correctness-first experimental runner, not an optimized server.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import struct
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from .config import ModelArgs
from .dequant import dequant_fp4, dequant_fp8, dequant_fp8_rows
from .layers import RMSNorm
from .load import load_token_map
from .model import Block
from .stream import StreamState, run_layer
from .hyper_connections import hc_pre
from .hardware import PROFILES, detect_hardware, resolve_settings


class NativeReader:
    def __init__(self, root, dense_cache_bytes=0, compile_dequant=False):
        self.root = Path(root)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}
        self.fds = {}
        self.bytes_read = 0
        self.dense_cache_bytes = dense_cache_bytes
        self.dense_cache = {}
        self.cached_bytes = 0
        self.cache_hits = 0
        self.read_seconds = 0.0
        self.dequant_fp4 = mx.compile(dequant_fp4) if compile_dequant else dequant_fp4
        self.dequant_fp8 = mx.compile(dequant_fp8) if compile_dequant else dequant_fp8

    def info(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            fd = os.open(self.root / shard, os.O_RDONLY)
            self.fds[shard] = fd
            n = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            if n > 100 * 1024**2:
                raise ValueError('Invalid safetensors header length')
            self.headers[shard] = (8+n, json.loads(os.pread(fd, n, 8)))
        base, header = self.headers[shard]
        return self.fds[shard], base, header[key]

    def read(self, key, start=None, stop=None):
        fd, base, meta = self.info(key)
        shape = list(meta['shape'])
        lo, hi = meta['data_offsets']
        if start is not None:
            if not (0 <= start <= stop <= shape[0]):
                raise IndexError((key, start, stop, shape))
            stride = (hi-lo) // shape[0]
            lo, hi = lo + start*stride, lo + stop*stride
            shape[0] = stop-start
        if hi-lo > 2*1024**3:
            raise MemoryError('Read must be sliced: ' + key)
        read_started = time.monotonic()
        raw = os.pread(fd, hi-lo, base+lo)
        self.read_seconds += time.monotonic() - read_started
        if len(raw) != hi-lo:
            raise IOError('Short tensor read: ' + key)
        self.bytes_read += len(raw)
        dt = meta['dtype']
        types = {'BF16':np.uint16, 'F16':np.float16, 'F32':np.float32,
                 'I32':np.int32, 'I64':np.int64, 'U32':np.uint32,
                 'U8':np.uint8, 'I8':np.uint8, 'F8_E4M3':np.uint8,
                 'F8_E8M0':np.uint8}
        if dt not in types:
            raise ValueError('Unsupported storage dtype: ' + dt)
        npdt = types[dt]
        a = mx.array(np.frombuffer(raw, dtype=npdt).reshape(shape))
        if dt == 'BF16': a = a.view(mx.bfloat16)
        return a

    def matrix(self, key, start=None, stop=None, expert=False):
        cache_key = (key, start, stop)
        if cache_key in self.dense_cache:
            self.cache_hits += 1
            return self.dense_cache[cache_key]
        w = self.read(key, start, stop)
        skey = key.removesuffix('.weight') + '.scale'
        if skey in self.index:
            if start is None:
                s = self.read(skey)
            elif expert:
                s = self.read(skey, start, stop)
            else:
                if start % 32 or stop % 32:
                    raise ValueError('FP8 matrix slices must align to 32 rows')
                s = self.read(skey, start//32, stop//32)
            w = self.dequant_fp4(w, s) if expert else self.dequant_fp8(w, s)
        mx.eval(w)
        # Stable admission avoids thrashing during the full layer scan each token.
        if not expert and self.cached_bytes + w.nbytes <= self.dense_cache_bytes:
            self.dense_cache[cache_key] = w
            self.cached_bytes += w.nbytes
        return w

    def rows(self, key, ids):
        shape = ids.shape
        flat = np.asarray(ids).reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = [self.read(key, int(i), int(i)+1) for i in unique]
        bank = mx.concatenate(rows, axis=0)
        return bank[mx.array(inverse.astype(np.int32))].reshape(*shape, bank.shape[-1])


class DiskEngram(nn.Module):
    def __init__(self, reader, prefix):
        super().__init__()
        self._reader, self._prefix = reader, prefix

    def __call__(self, ids):
        mx.eval(ids)
        ids = np.array(ids)
        w = self._reader.rows(self._prefix+'.weight', ids)
        s = self._reader.rows(self._prefix+'.scale', ids)
        result = dequant_fp8_rows(w, s)
        mx.eval(result)
        return result


class DiskExperts(nn.Module):
    def __init__(self, reader, prefix, limit):
        super().__init__()
        self._reader, self._prefix, self._limit = reader, prefix, limit

    def __call__(self, x, indices):
        mx.eval(x, indices)
        ids = np.array(indices)
        # Evaluate each selected expert once for all tokens that selected it.
        result = mx.zeros((*ids.shape, x.shape[-1]), dtype=mx.float32)
        for eid in np.unique(ids):
            ti, ki = np.where(ids == eid)
            tx = x[mx.array(ti.astype(np.int32))]
            def project(values, weight):
                return mx.gather_mm(values[:, None, None, :], weight[None].swapaxes(-1, -2),
                                    rhs_indices=mx.zeros((len(ti), 1), dtype=mx.int32)).reshape(len(ti), -1)
            p = self._prefix + f'.{eid}'
            w1 = self._reader.matrix(p+'.w1.weight', expert=True)
            gate = project(tx, w1).astype(mx.float32)
            mx.eval(gate)
            del w1
            w3 = self._reader.matrix(p+'.w3.weight', expert=True)
            up = project(tx, w3).astype(mx.float32)
            mx.eval(up)
            del w3
            if self._limit > 0:
                gate = mx.minimum(gate, self._limit)
                up = mx.clip(up, -self._limit, self._limit)
            # SwitchGLU carries the activation into down_proj in float32.
            h = nn.silu(gate)*up
            w2 = self._reader.matrix(p+'.w2.weight', expert=True)
            y = project(h, w2)
            result[mx.array(ti.astype(np.int32)), mx.array(ki.astype(np.int32))] = y
            mx.eval(result)
            del w2, y, h, up, gate
        return result


def build_disk_layer(args, i, reader):
    # MLX initialization is lazy. Replace the large placeholders before eval.
    layer = Block(i, args)
    prefix = f'layers.{i}.'
    layer.ffn.experts = DiskExperts(reader, prefix+'ffn.experts', args.swiglu_limit)
    if layer.engram is not None:
        layer.engram.embed = DiskEngram(reader, prefix+'engram.embed')
    weights = []
    for key in reader.index:
        if not key.startswith(prefix) or '.ffn.experts.' in key or '.engram.embed.' in key:
            continue
        if key.endswith('.scale'): continue
        w = reader.matrix(key) if key.endswith('.weight') else reader.read(key)
        mx.eval(w)
        weights.append((key[len(prefix):], w))
    layer.load_weights(weights, strict=True)
    mx.eval(layer.parameters())
    return layer


class NativeStream:
    def __init__(self, root, max_seq_len=128, dense_cache_bytes=0, aggressive_cleanup=True, compile_dequant=False):
        self.reader = NativeReader(root, dense_cache_bytes, compile_dequant)
        self.aggressive_cleanup = aggressive_cleanup
        self.args = ModelArgs.from_dict(json.loads((Path(root)/'config.json').read_text()))
        token_map = load_token_map(str(root), self.args)
        if token_map is None: raise ValueError('Engram tokenizer map missing')
        self.state = StreamState(self.args, 1, max_seq_len, token_map)
        self.norm = RMSNorm(self.args.dim, self.args.norm_eps)
        self.norm.weight = self.reader.read('norm.weight')

    def __call__(self, ids):
        ids = np.asarray(ids, dtype=np.int32).reshape(1,-1)
        embedded = self.reader.rows('embed.weight', ids)
        h = self.state.begin(ids, embedded, self.args.hc_mult)
        started = time.monotonic()
        for i in range(self.args.n_layers):
            layer = build_disk_layer(self.args, i, self.reader)
            h = run_layer(layer, h, self.state)
            mx.eval(h, self.state.pre_mix)
            if not bool(mx.all(mx.isfinite(h)).item()):
                raise FloatingPointError(f'Nonfinite layer {i}')
            del layer
            if self.aggressive_cleanup:
                mx.clear_cache()
                gc.collect()
            print(json.dumps(dict(layer=i, seconds=round(time.monotonic()-started,2),
                                  read_gib=round(self.reader.bytes_read/1024**3,3),
                                  peak_gib=round(mx.get_peak_memory()/1024**3,3))), flush=True)
        h = self.norm(hc_pre(h, self.state.pre_mix))[:, -1, :].astype(mx.float32)
        mx.eval(h)
        # Only the last position needs logits; stream the output vocabulary.
        logits = []
        for start in range(0, self.args.vocab_size, 2048):
            stop = min(start+2048, self.args.vocab_size)
            w = self.reader.matrix('head.weight', start, stop)
            scores = h @ w.astype(mx.float32).T
            mx.eval(scores)
            logits.append(scores)
            del w
        out = mx.concatenate(logits, axis=-1)
        mx.eval(out)
        self.state.cache.offset += ids.shape[1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--prompt', default='Hi')
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--profile', choices=['auto', *PROFILES], default='auto')
    ap.add_argument('--memory-limit-gib', type=float)
    ap.add_argument('--dense-cache-gib', type=float)
    ap.add_argument('--mlx-cache-mib', type=int)
    ap.add_argument('--reuse-allocator', action='store_true')
    ap.add_argument('--compile-dequant', action='store_true')
    opts = ap.parse_args()
    hardware = detect_hardware()
    if hardware.architecture != 'arm64':
        ap.error(f'Apple Silicon arm64 is required; detected {hardware.architecture}')
    try:
        runtime = resolve_settings(
            opts.profile,
            info=hardware,
            memory_limit_gib=opts.memory_limit_gib,
            dense_cache_gib=opts.dense_cache_gib,
            mlx_cache_mib=opts.mlx_cache_mib,
        )
    except ValueError as exc:
        ap.error(str(exc))
    mx.set_memory_limit(int(runtime.memory_limit_gib*1024**3))
    mx.set_cache_limit(runtime.mlx_cache_mib*1024**2)
    tok = Tokenizer.from_file(str(Path(opts.model)/'tokenizer.json'))
    prompt = opts.prompt if opts.raw else '<｜begin▁of▁sentence｜><｜User｜>'+opts.prompt+'<｜Assistant｜></think>'
    ids = tok.encode(prompt, add_special_tokens=False).ids
    print(json.dumps(dict(event='runtime_profile', profile=runtime.profile,
                          chip=hardware.chip, physical_memory_gib=hardware.memory_gib,
                          memory_limit_gib=runtime.memory_limit_gib,
                          dense_cache_gib=runtime.dense_cache_gib,
                          mlx_cache_mib=runtime.mlx_cache_mib)), flush=True)
    model = NativeStream(opts.model, max_seq_len=max(128,len(ids)+opts.max_tokens+16),
                         dense_cache_bytes=int(runtime.dense_cache_gib*1024**3),
                         aggressive_cleanup=not opts.reuse_allocator, compile_dequant=opts.compile_dequant)
    print(json.dumps(dict(prompt=prompt, input_ids=ids)), flush=True)
    started = time.monotonic()
    output = []
    timings = []
    for n in range(opts.max_tokens):
        step_start = time.monotonic()
        logits = model(ids)
        token = int(mx.argmax(logits, axis=-1).item())
        timings.append(time.monotonic()-step_start)
        output.append(token)
        print(json.dumps(dict(step=n, token=token, text=tok.decode(output),
                              seconds=round(time.monotonic()-started,2),
                              peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)), flush=True)
        if token == 1: break
        ids = [token]
    print(json.dumps(dict(event='complete', output=tok.decode(output), output_ids=output,
                          prompt_tokens=len(tok.encode(prompt, add_special_tokens=False).ids),
                          generated_tokens=len(output), first_token_seconds=timings[0],
                          decode_tokens_per_second=(len(timings)-1)/sum(timings[1:]) if len(timings)>1 else None,
                          total_seconds=time.monotonic()-started,
                          read_gib=model.reader.bytes_read/1024**3,
                          read_seconds=model.reader.read_seconds,
                          dense_cache_gib=model.reader.cached_bytes/1024**3,
                          cache_hits=model.reader.cache_hits,
                          peak_mlx_gib=mx.get_peak_memory()/1024**3,
                          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)), flush=True)


if __name__ == '__main__':
    main()
