# Fit-screen: bge-small-en-v1.5 — PyTorch eager vs ONNX Runtime fp32 / int8-dynamic on this CPU.
# Measures the "strong baseline" margin a hand Rust port would have to beat.
import time, sys, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"
N, BATCH, WARM, REPS = 512, 32, 2, 5

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).eval()

rng = np.random.default_rng(7)
vocabish = ("system latency cache kernel tensor ratio judgment prior elicitation "
            "market forecast rust memory bandwidth quantization embedding rerank "
            "pipeline oracle evidence measurement tolerance parity fusion schedule").split()
texts = [" ".join(rng.choice(vocabish, size=int(rng.integers(8, 40)))) for _ in range(N)]

enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
print("padded seq len:", enc["input_ids"].shape[1], flush=True)

def batches(d):
    for i in range(0, N, BATCH):
        yield {k: v[i:i+BATCH] for k, v in d.items()}

def bench(fn, label):
    for _ in range(WARM):
        for b in batches(enc): fn(b)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        outs = [fn(b) for b in batches(enc)]
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    print(f"{label:28s} {N/best:8.1f} sent/s   best {best*1000:7.1f} ms  (reps {[round(t*1000) for t in ts]})", flush=True)
    return np.concatenate(outs, axis=0), best

def pool(hidden, mask):
    m = mask[..., None].astype(np.float32) if isinstance(hidden, np.ndarray) else mask.unsqueeze(-1).float()
    s = (hidden * m).sum(1 if isinstance(hidden, np.ndarray) else 1) if isinstance(hidden, np.ndarray) else (hidden * m).sum(1)
    return s / m.sum(1)

@torch.inference_mode()
def run_torch(b):
    h = model(**b).last_hidden_state
    return pool(h, b["attention_mask"]).numpy()

ref, t_eager = bench(run_torch, "torch eager fp32")

# ---- ONNX export ----
import torch.onnx
class Wrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = model
    def forward(s, input_ids, attention_mask, token_type_ids):
        return s.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state

ex = {k: v[:2] for k, v in enc.items()}
dyn = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "token_type_ids": {0: "b", 1: "s"}}
torch.onnx.export(Wrap().eval(), (ex["input_ids"], ex["attention_mask"], ex["token_type_ids"]),
                  "/tmp/fitscreen/model.onnx", input_names=list(dyn), output_names=["h"],
                  dynamic_axes=dyn, opset_version=17, dynamo=False)
print("onnx exported", flush=True)

import onnxruntime as ort
def ort_fn(path):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def f(b):
        feed = {k: v.numpy() for k, v in b.items()}
        h = sess.run(["h"], feed)[0]
        return pool(h, feed["attention_mask"])
    return f

o32, t_fp32 = bench(ort_fn("/tmp/fitscreen/model.onnx"), "ort fp32")

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("/tmp/fitscreen/model.onnx", "/tmp/fitscreen/model.int8.onnx", weight_type=QuantType.QInt8)
o8, t_int8 = bench(ort_fn("/tmp/fitscreen/model.int8.onnx"), "ort int8-dynamic")

def cos(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)

print(f"\nparity: ort-fp32 vs eager  cos min/mean = {cos(ref,o32).min():.6f} / {cos(ref,o32).mean():.6f}")
print(f"parity: ort-int8 vs eager  cos min/mean = {cos(ref,o8).min():.6f} / {cos(ref,o8).mean():.6f}")
print(f"\nspeedups vs torch eager:  ort-fp32 {t_eager/t_fp32:.2f}x   ort-int8 {t_eager/t_int8:.2f}x")
