# Chapter 20 — Live Coding Bank

> **Why this chapter exists:** Avrioc is confirmed (Glassdoor) to use a 3rd-party timed test and live coding in technical rounds. One reviewer specifically mentioned "rotate an array." This chapter covers (A) DS&A patterns most likely on the timed test, and (B) ML/Python live coding most likely in the human-led round.

**How to use:** Read each problem, think through the approach for 30s, **then code it from scratch on paper or whiteboard**. Don't peek at the solution until you've stalled. Do at least 2 from each section.

---

## A. DS&A patterns (timed test, ~30–45 min, HackerRank/CodeSignal style)

### A.1 Rotate an array by k positions (CONFIRMED asked at Avrioc)

**Problem:** Given `arr = [1,2,3,4,5,6,7]`, `k = 3`, return `[5,6,7,1,2,3,4]`.

**Naive approach (O(n·k)):** rotate one position at a time. Too slow.

**Optimal — three reversals (O(n) time, O(1) space):**

```python
def rotate(arr: list[int], k: int) -> None:
    n = len(arr)
    k %= n  # handle k > n
    # Reverse the whole array, then first k, then the rest
    arr.reverse()
    arr[:k] = reversed(arr[:k])
    arr[k:] = reversed(arr[k:])

# Walkthrough for [1,2,3,4,5,6,7], k=3:
# Step 1 reverse all:   [7,6,5,4,3,2,1]
# Step 2 reverse [0:3]: [5,6,7,4,3,2,1]
# Step 3 reverse [3:]:  [5,6,7,1,2,3,4]  ✓
```

**Why interviewers love this:** the three-reversal trick is non-obvious; the alternative (slice copy `arr[-k:] + arr[:-k]`) is O(n) extra space, which they'll dock you for if they ask for in-place.

### A.2 Two-sum (any variant)

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        if (need := target - x) in seen:
            return [seen[need], i]
        seen[x] = i
    return []
```

**Watch for:** they'll ask "what if the array is sorted?" → switch to two pointers, O(1) space.

### A.3 Longest substring without repeating characters

**Sliding window template — memorize this shape:**

```python
def longest_unique_substring(s: str) -> int:
    last: dict[str, int] = {}  # char -> most recent index
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

The same template solves: longest with at most K distinct, min window covering target, etc. **Just modify the "expand" and "contract" conditions.**

### A.4 Merge intervals

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
```

### A.5 Top-K frequent elements (heap pattern)

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    return heapq.nlargest(k, counts, key=counts.get)
```

`heapq.nlargest` is O(n log k), faster than sorting at O(n log n).

### A.6 Binary search — first position where condition flips

This is the universal binary search. Memorize once, reuse forever.

```python
def first_true(lo: int, hi: int, ok) -> int:
    """Smallest x in [lo, hi] where ok(x) is True. Assumes monotonic."""
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Used for: first bad version, smallest divisor, capacity to ship, square root.

### A.7 Tree level-order traversal (BFS)

```python
from collections import deque

def level_order(root):
    if not root: return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

### A.8 LRU Cache (commonly asked at AI/infra companies)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d: OrderedDict = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.d: return -1
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)  # evict oldest
```

**Why relevant:** KV-cache eviction in LLM serving is a real-world LRU. They might ask the follow-up: "what if you needed it thread-safe?" → `threading.Lock` around get/put.

### A.9 Producer-consumer with bounded queue (common Python design test)

```python
import threading, queue, time

q: queue.Queue[int] = queue.Queue(maxsize=10)
stop = threading.Event()

def producer():
    i = 0
    while not stop.is_set():
        q.put(i)  # blocks if full
        i += 1

def consumer():
    while not stop.is_set() or not q.empty():
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        # process item
        q.task_done()
```

This pattern shows up in batched-inference servers — be ready to explain when you'd use a `queue.Queue` vs an `asyncio.Queue` (threads vs coroutines).

---

## B. ML / Python coding (human-led round)

### B.1 Scaled dot-product attention from scratch (CLASSIC, expect this)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,    # (B, H, T_q, d_k)
    K: torch.Tensor,    # (B, H, T_k, d_k)
    V: torch.Tensor,    # (B, H, T_k, d_v)
    mask: torch.Tensor | None = None,  # (B, 1, T_q, T_k), 1 = keep, 0 = mask
    dropout_p: float = 0.0,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # (B, H, T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ V                                    # (B, H, T_q, d_v)
```

**Be ready to answer these follow-ups while you write the code:**

- *"Why divide by √d_k?"* — Without scaling, dot products grow as O(d_k); softmax saturates → vanishing gradients. Scaling keeps the variance ≈ 1.
- *"Where does causal masking fit?"* — Build a triangular mask `torch.tril(torch.ones(T, T))`; broadcast it into the `mask` argument.
- *"What's FlashAttention doing differently?"* — Same math, but tiles Q/K/V into SRAM blocks, recomputes during backward instead of storing the full attention matrix → O(N) memory instead of O(N²), 2–4x faster end-to-end.

### B.2 Multi-head attention (often asked right after B.1)

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # each (B, H, T, d_k)
        out = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).reshape(B, T, -1)        # (B, T, d_model)
        return self.out(out)
```

**Common pitfalls graders watch for:** forgetting to scale; mishandling the head reshape (off-by-one in the permute); applying dropout in the wrong place; using `view` where you must `reshape` after a transpose.

### B.3 FastAPI streaming endpoint with cancellation (HIGHLY likely — JD names FastAPI + vLLM)

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import httpx

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

VLLM_URL = "http://vllm-service:8000/v1/completions"

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    async def token_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", VLLM_URL,
                json={"prompt": req.prompt, "max_tokens": req.max_tokens, "stream": True},
            ) as upstream:
                async for chunk in upstream.aiter_lines():
                    if await request.is_disconnected():
                        # client gone — stop pulling, vLLM cancels via httpx context exit
                        break
                    if chunk.startswith("data: "):
                        yield chunk + "\n\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

**The 3 things they're checking:**
1. **You return `StreamingResponse`, not a JSONResponse with a generator** (won't stream).
2. **You handle disconnects** with `request.is_disconnected()` to free the upstream call.
3. **You use `httpx.AsyncClient` with streaming** — `requests` would block the event loop.

### B.4 Ray Serve deployment with batching

```python
from ray import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.01)
    async def __call__(self, prompts: list[str]) -> list[str]:
        # batch_size 1 entries arrive individually; serve.batch coalesces them
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=128)
        return self.tok.batch_decode(out, skip_special_tokens=True)

deploy = LLMDeployment.bind()
```

**Follow-up they'll ask:** "why `serve.batch` vs vLLM's continuous batching?"
**Answer:** `serve.batch` does **request-level** batching with a static window — fine for embeddings/classification. vLLM does **iteration-level** continuous batching, where finished sequences free their slots immediately and new ones join mid-batch. **For an LLM, vLLM is the right tool;** `serve.batch` is great for upstream stages of a Ray Serve graph (re-rankers, classifiers).

### B.5 Sliding-window LRU TTL cache (production pattern)

```python
import time
from collections import OrderedDict

class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max = max_size
        self.ttl = ttl_seconds
        self.d: OrderedDict[str, tuple[float, object]] = OrderedDict()

    def get(self, key: str):
        if key not in self.d: return None
        ts, val = self.d[key]
        if time.time() - ts > self.ttl:
            del self.d[key]
            return None
        self.d.move_to_end(key)
        return val

    def put(self, key: str, value):
        self.d[key] = (time.time(), value)
        self.d.move_to_end(key)
        while len(self.d) > self.max:
            self.d.popitem(last=False)
```

**Why production-relevant:** caching LLM responses, embedding caches, feature caches — the kind of code reviewers nod at.

### B.6 Compute embeddings cosine similarity matrix (numpy-fluent test)

```python
import numpy as np

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (n, d), B: (m, d) -> (n, m)"""
    A_n = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_n = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_n @ B_n.T
```

**Common gotcha:** if any row is zero, you'll divide by zero. Add `+ 1e-12` to the denominator in real code.

### B.7 Implement a simple RAG retriever (composite, expect a 30-min variant)

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Doc:
    id: str
    text: str
    embedding: np.ndarray

class InMemoryRetriever:
    def __init__(self, embed_fn):
        self.embed = embed_fn
        self.docs: list[Doc] = []
        self.matrix: np.ndarray | None = None

    def add(self, doc_id: str, text: str) -> None:
        emb = self.embed(text)
        self.docs.append(Doc(doc_id, text, emb))
        self.matrix = np.stack([d.embedding for d in self.docs])

    def search(self, query: str, k: int = 5) -> list[Doc]:
        q = self.embed(query)
        q /= np.linalg.norm(q) + 1e-12
        M = self.matrix / (np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-12)
        scores = M @ q
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.docs[i] for i in idx]
```

**Trade-off they'll probe:** "why argpartition then argsort, not just argsort?" → argpartition is O(n), argsort is O(n log n). Only sort the top-k.

### B.8 Pydantic v2 model + FastAPI input validation (Python production)

```python
from pydantic import BaseModel, Field, field_validator

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    top_p: float = Field(gt=0.0, le=1.0, default=0.95)
    max_tokens: int = Field(ge=1, le=4096, default=512)
    user_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt cannot be only whitespace")
        return v
```

This is the kind of input hardening any production-aware reviewer will check.

---

## C. Quick warm-ups (do all of these in 60 sec each)

These are mental flexibility drills. Solve each in your head out loud.

1. **Reverse a linked list** — three pointers, in-place, O(n) time, O(1) space.
2. **Detect cycle in a linked list** — Floyd's tortoise & hare.
3. **Validate balanced parentheses** — stack, push opens, pop on close.
4. **Find kth largest** — quickselect or `heapq.nlargest(k, nums)[-1]`.
5. **Word frequency from a string** — `Counter(s.split())`.
6. **Flatten a nested list** — recursion or stack.
7. **Are two strings anagrams** — `Counter(a) == Counter(b)`.
8. **Maximum sum subarray** — Kadane's: `cur = max(x, cur + x); best = max(best, cur)`.
9. **Number of islands** (DFS on grid) — flood fill.
10. **Implement `defaultdict(list)` from scratch** — subclass `dict`, override `__missing__`.

---

## D. Live coding — meta strategy

Even when stuck, do these and you'll keep momentum:

1. **Read the problem aloud and restate it.** "So I want a function that takes X and returns Y. Edge cases I'm thinking about: empty input, very large input, …"
2. **Talk through the brute force first**, then say "but we can do better."
3. **Pick a data structure first**, then write the function signature, then the body.
4. **Test with one example by hand** before declaring done. Walk the input through the loop counter by counter — catches off-by-one errors live.
5. **Mention complexity unprompted**: "This is O(n) time, O(k) space because…"
6. **If you stall**: "Let me think out loud. The blocker is X. One option is A, but its problem is B. So maybe C, which gives me…" — interviewers grade your *reasoning*, not just the answer.

---

## E. The 4 things that will make you stand out

1. **Type-hint everything.** `def f(x: list[int]) -> int:` — instant signal.
2. **Name variables semantically.** `seen_indices` beats `d`; `left, right` beats `i, j`.
3. **Talk about complexity proactively.** They will write it down whether you say it or not.
4. **Test the edge cases yourself.** Empty input, one element, duplicates, very large input.

---

Continue to **[Chapter 21 — Slurm, DGX, HPC stack](21_slurm_dgx_hpc.md)**.
