# Chapter 01 — Foundations
## Neural Networks & Word Embeddings (the bedrock)

> You will almost certainly not be asked to derive backprop on a whiteboard for a Senior ML role. But a sharp interviewer *will* ask "why does your transformer need LayerNorm?" or "why softmax temperature matters" — and those answers live here.

---

## 1.1 Neural Network — The 60-Second Mental Model

A neural network is a parameterised function f_θ(x) = y, trained by **minimizing a loss** L(y, ŷ) using **gradient descent** on θ.

```
          input x
            │
    ┌───────▼──────┐
    │ Linear: Wx+b │   ← θ = {W, b}
    └───────┬──────┘
            │
    ┌───────▼──────┐
    │ Activation σ │   ← non-linearity (ReLU, GELU, SiLU)
    └───────┬──────┘
            │
       ... (N layers) ...
            │
    ┌───────▼──────┐
    │   Output y   │
    └──────────────┘
```

### Why non-linear activations?
A stack of linear layers collapses to a single linear layer (matrix product of matrices is still a matrix). Non-linearity is what gives neural nets the ability to approximate arbitrary functions (Universal Approximation Theorem, 1989 Cybenko, 1991 Hornik).

### Common activations (with gotchas)

| Activation | Formula | Pros | Cons |
|------------|---------|------|------|
| Sigmoid | 1/(1+e^-x) | Smooth, bounded | Vanishing gradient, not zero-centered |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | Zero-centered | Still saturates |
| **ReLU** | max(0, x) | Fast, no saturation for x>0 | "Dying ReLU" for x<0 (dead neurons) |
| Leaky ReLU | max(0.01x, x) | Fixes dying ReLU | Extra hyperparam |
| **GELU** | x · Φ(x) | Smooth ReLU, used in BERT/GPT-2 | More compute |
| **SiLU/Swish** | x · sigmoid(x) | Used in LLaMA/Mistral | More compute |
| **SwiGLU** | Swish(xW) · (xV) | Gated, SOTA in LLMs | 2× params in FFN |

**Modern LLM default:** SwiGLU in the feed-forward block. Why? A gating mechanism lets the model dynamically scale information flow, empirically beating plain ReLU/GELU by a few perplexity points.

---

## 1.2 Backpropagation — The One Formula You Must Know

For loss L, layer output a = σ(z), z = Wx + b:

```
∂L/∂W = δ · xᵀ    where δ = ∂L/∂z = (∂L/∂a) ⊙ σ'(z)
∂L/∂b = δ
∂L/∂x = Wᵀ · δ    ← the signal propagated to the previous layer
```

**Chain rule everywhere.** Backprop is just the chain rule + caching intermediate activations (the "forward pass stash"). This is why training uses ~3× the memory of inference.

### Why transformers have gradient problems at depth
- **Vanishing gradients**: gradients shrink through each layer (especially with sigmoid/tanh).
- **Exploding gradients**: gradients blow up (RNNs suffer here).

**Fixes baked into transformers:**
1. **Residual connections** (x + Layer(x)) — gradients skip past non-linearities
2. **LayerNorm / RMSNorm** — stabilizes activation variance
3. **Careful weight init** (Xavier, Kaiming, Fixup)
4. **Gradient clipping** (typically ||g|| ≤ 1.0) during training

---

## 1.3 Optimizers — Just Know These Four

| Optimizer | Update rule (simplified) | When to use |
|-----------|--------------------------|-------------|
| **SGD + momentum** | v ← βv + g; θ ← θ - η·v | CV / simple tasks |
| **Adam** | Moments of g and g²; adaptive LR per param | Default for most ML |
| **AdamW** | Adam + decoupled weight decay | **Default for LLMs / transformers** |
| **Lion** | Sign-based update, less memory | Large-scale training, memory-efficient |

**Interview gotcha:** "Why AdamW and not Adam for transformers?" — In vanilla Adam, L2 weight-decay gets scaled by the adaptive LR, which is inconsistent. AdamW decouples weight-decay from the gradient update, restoring proper regularization. Mandatory for transformers.

---

## 1.4 Normalization — BatchNorm vs LayerNorm vs RMSNorm

```
BatchNorm: normalize across the BATCH for each feature (x - μ_batch) / σ_batch
LayerNorm: normalize across FEATURES for each sample  (x - μ_sample) / σ_sample
RMSNorm:   x / RMS(x), no mean-centering
```

| Aspect | BatchNorm | LayerNorm | RMSNorm |
|--------|-----------|-----------|---------|
| Works with batch=1? | ❌ | ✅ | ✅ |
| Used in CNNs? | ✅ | ❌ | ❌ |
| Used in Transformers? | ❌ | ✅ (GPT-2, BERT) | ✅ (LLaMA, Mistral) |
| Cost | Low | Medium | **Lowest** |

**Why RMSNorm in LLMs?** Drops the mean-centering step. ~7-10% speedup, no quality loss. Used in LLaMA 2/3, Mistral, Qwen.

---

## 1.5 Regularization — L1, L2, Dropout

- **L1 (Lasso)**: λ·Σ|w| → encourages sparsity (feature selection)
- **L2 (Ridge)**: λ·Σw² → shrinks all weights (smoother fits)
- **Dropout**: randomly zero activations with probability p at training time
- **Label smoothing**: softens hard 0/1 labels → model less overconfident
- **Early stopping**: monitor val loss, stop when it plateaus

**LLM-specific regularization:** Weight decay (0.1 typical), dropout (0.1 usually on attention and FFN, **zero for modern large LLMs** — they rely on data scale for regularization).

---

## 1.6 Word Embeddings — Before Transformers Ate the World

### 1.6.1 One-hot encoding (don't ever propose this)
```
"cat" → [0, 0, 1, 0, 0, ...]    # size = |V| = 50K+
```
- **Pro:** Trivial
- **Con:** No semantic similarity ("cat" and "dog" are just as far as "cat" and "pizza"), sparse, massive.

### 1.6.2 Word2Vec (Mikolov et al., 2013)
Two variants:

```
CBOW (Continuous Bag-of-Words): predict center word from context
[the, quick, brown, ___, jumps] → "fox"

Skip-gram: predict context from center word
"fox" → [the, quick, brown, jumps, over]
```

**Loss:** Softmax over vocabulary (with negative sampling / hierarchical softmax for speed):
```
L = -log σ(v_center · v_context) - Σ_neg log σ(-v_center · v_neg)
```

**Why it works:** The distributional hypothesis — "you shall know a word by the company it keeps" (Firth, 1957). Words appearing in similar contexts get similar embeddings.

**Classic property:** `king - man + woman ≈ queen`. Linear arithmetic in embedding space reveals semantic relationships.

### 1.6.3 GloVe (Stanford, 2014)
Factorizes the global word co-occurrence matrix:
```
L = Σ f(X_ij) · (wᵢᵀ · wⱼ + bᵢ + bⱼ - log X_ij)²
```
where X_ij = count of word j appearing in context of word i.

**Word2Vec is local (window-based); GloVe is global (full matrix).** GloVe often performs slightly better on analogy tasks; Word2Vec is faster to train.

### 1.6.4 FastText (Facebook, 2016)
Represents each word as sum of its **character n-grams**:
```
"apple" → <ap, app, appl, apple, pple, ple>, ple, le>, e>>
```
- **Pro:** Handles OOV words (new word = sum of its n-grams)
- **Pro:** Good for morphologically rich languages (Arabic, Finnish)
- **Con:** Larger model, slower

### 1.6.5 Contextual embeddings (ELMo → BERT → Modern LLMs)
Classic embeddings are **static**: "bank" has one vector regardless of meaning. Contextual embeddings compute a different vector per context:

```
"I sat by the river bank."         → bank₁
"I deposited money at the bank."   → bank₂
```

This is what every modern LLM does internally — every hidden state is a contextual embedding of the token at that position.

### 1.6.6 When would you still use Word2Vec/GloVe?
- **Latency-critical retrieval** (TF-IDF + Word2Vec averages are 100× faster than BERT)
- **Edge devices** without GPU
- **As features in classical ML** (XGBoost, logistic regression)
- **Baseline for ablation studies**

### 1.6.7 Embedding dimensionality — what to pick?
- Word2Vec/GloVe classic: 50-300
- Sentence-BERT: 384-768
- OpenAI text-embedding-3-large: 3072 (truncatable via Matryoshka to 256/1024)
- BGE-M3: 1024 dense + sparse + multi-vector

**Rule of thumb:** bigger is better but with diminishing returns; 768 is a sweet spot for retrieval. Use Matryoshka if storage matters (vector DB $$$).

---

## 1.7 Loss Functions — The Four That Matter

| Task | Loss | Formula |
|------|------|---------|
| Regression | MSE | (1/N)Σ(y-ŷ)² |
| Regression (robust) | MAE / Huber | \|y-ŷ\| / combination |
| Binary classification | BCE | -y·log(ŷ) - (1-y)·log(1-ŷ) |
| Multi-class | Cross-entropy | -Σ y_i·log(ŷ_i) |
| Contrastive (embeddings) | InfoNCE | see Chapter 04 |
| Ranking | Triplet / margin | max(0, d(a,p) - d(a,n) + α) |
| RL / RLHF | PPO with KL | see Chapter 06 |

**LLM training uses cross-entropy on next-token prediction:**
```
L = - Σ_t log P(x_t | x_<t)
```

---

## 1.8 Evaluation Metrics — Classification / Retrieval

### Classification
- **Accuracy** — (TP+TN)/All. Bad for imbalanced data.
- **Precision** — TP/(TP+FP). "Of what I said YES, how many were right?"
- **Recall** — TP/(TP+FN). "Of all actual YES, how many did I catch?"
- **F1** — harmonic mean of P, R. Balanced view.
- **AUC-ROC** — area under TPR-FPR curve. Threshold-independent.
- **AUC-PR** — better for very imbalanced data (e.g., fraud, your TrueBalance withdraw prediction).

### Retrieval / Ranking
- **Recall@k** — of all relevant docs, what fraction made top-k?
- **Precision@k** — of top-k, what fraction are relevant?
- **MRR** — mean reciprocal rank of first relevant. Good for "1 right answer" tasks.
- **nDCG@k** — discounted cumulative gain; weighs higher positions more. Industry standard.
- **Hit@k** — did any of top-k contain a relevant doc? Binary.

### LLM / Generation
- **Perplexity** = exp(avg cross-entropy). Lower = better LM.
- **BLEU / ROUGE** — n-gram overlap. Brittle for open-ended.
- **LLM-as-judge** — an LLM scores outputs. Dominant in 2025-2026.
- **Factuality / Faithfulness** — RAGAS, TruLens.
- **Human eval** — the ground truth, expensive.

---

## 1.9 Bias–Variance Tradeoff — The Diagram

```
   │                    Total Error
   │                   ╱‾‾╲
   │                  ╱    ╲
   │  Bias²          ╱      ╲─── Variance
E  │╲               ╱        ╲ ╱
r  │ ╲             ╱          ╳
r  │  ╲           ╱          ╱ ╲
o  │   ╲________╱          ╱   ╲
r  │                       ╱     ╲___
   │ underfit  sweet spot  overfit
   └─────────────────────────────────▶
                Model Complexity
```

- **High bias (underfit):** Train loss high, val loss high. Add capacity.
- **High variance (overfit):** Train loss low, val loss high. Regularize / more data.
- **LLMs are unusual** — grokking, double descent, emergent abilities muddle classical tradeoff at scale.

---

## 1.10 Interview Q&A — Foundations

**Q1. Why can't we use just a linear model with enough parameters?**
> A stack of linear layers is mathematically equivalent to a single linear layer (product of linear maps is linear). Non-linearity is what gives neural nets universal approximation.

**Q2. What's "dying ReLU" and how do you fix it?**
> If a ReLU neuron's weighted sum becomes negative for all inputs, its gradient is always zero and the neuron never updates again — effectively dead. Fixes: Leaky ReLU, ELU, GELU, proper initialization (He init), lower learning rate.

**Q3. Why do we need residual connections in deep networks?**
> Gradients of the form ∂L/∂x propagated through N layers suffer vanishing or exploding. A skip connection (x + f(x)) adds an identity path with gradient 1, so signal/gradient flows through arbitrarily deep stacks. ResNet and every modern transformer use them.

**Q4. LayerNorm vs BatchNorm in a single sentence each.**
> LayerNorm normalizes across features for a single sample (works with batch=1, sequence-friendly). BatchNorm normalizes across the batch for a single feature (fails with batch=1, bad for variable-length sequences).

**Q5. Why is Adam/AdamW the default for transformers?**
> Adam adapts learning rate per parameter using running means of gradients and their squares — robust to heterogeneous loss landscapes. AdamW decouples weight decay from the adaptive step, which matters for transformers where weights have very different scales across layers.

**Q6. What is Word2Vec's skip-gram loss at a conceptual level?**
> For each center word, predict its context words by maximizing the inner product between the center embedding and true-context embeddings, and minimizing it with random negative samples (negative sampling) — training via softmax approximations because the true softmax over 1M+ words is infeasible.

**Q7. Can Word2Vec handle out-of-vocabulary words?**
> No. Word2Vec assigns embeddings to whole words from a fixed vocabulary. FastText solved this by summing character-n-gram embeddings, so unseen words still get a reasonable vector.

**Q8. Why did contextual embeddings (BERT etc.) replace static embeddings?**
> Static embeddings give "bank" one vector regardless of meaning (river bank vs. financial bank). Contextual embeddings compute a vector per occurrence by running attention over the full sentence — capturing polysemy and syntactic role.

**Q9. Your model overfits — in order of priority, what would you try?**
> (1) Get more/better data (most impactful). (2) Regularize — weight decay, dropout, label smoothing. (3) Reduce model capacity. (4) Early stopping. (5) Data augmentation. For LLMs specifically, scale data before scaling model.

**Q10. How does softmax differ from argmax, and why does it matter for gradients?**
> Softmax is differentiable — output is a smooth probability distribution, gradients flow everywhere. Argmax is non-differentiable (zero gradient almost everywhere, undefined at ties). Softmax with temperature=0 approaches argmax; temperature→∞ approaches uniform.

**Q11. [Gotcha] When does accuracy mislead you as a metric?**
> Imbalanced datasets. If 99% of loans don't default, a model predicting "no default" every time scores 99% accuracy but catches zero defaults. Use precision/recall, PR-AUC, or balanced accuracy instead.

**Q12. [Gotcha] You train with cross-entropy but validate with accuracy — and they disagree. What happened?**
> Cross-entropy penalizes *confidence* on wrong answers (a confidently-wrong prediction has huge loss), while accuracy only counts top-1. A model can be less accurate but produce better-calibrated probabilities, or vice versa. For production, decide which matters — if a downstream system consumes probabilities (ranking, thresholding), cross-entropy / NLL is what you care about.

---

## 1.11 Resume tie-ins

- **"XGBoost Lambda, p99 < 500ms"** — uses tree-based, not neural, but all the bias/variance intuition, regularization (L1/L2 in XGBoost = alpha/lambda), and monitoring concepts carry over. Be ready to explain why XGBoost wins on tabular data.
- **"NER lender extractor"** — NER is a classic sequence-labeling task, now usually a BERT fine-tune. Be ready to talk about token-level classification, BIO tagging, and F1 at the entity level.
- **"Snowflake feature store"** — feature engineering quality dominates model quality in classical ML. Tie this to bias-variance: well-engineered features ≈ lower bias.

---

Continue to **[Chapter 02 — Transformers Deep Dive](02_transformers.md)**.
