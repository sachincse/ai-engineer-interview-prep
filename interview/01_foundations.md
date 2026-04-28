# Chapter 01 — Foundations
## Neural Networks & Word Embeddings (the bedrock)

> The interviewer at Avrioc may not ask you to derive backprop on a whiteboard, but they **will** test whether you understand *why* a transformer needs LayerNorm, *why* SwiGLU replaced ReLU in modern LLMs, and *why* Word2Vec arithmetic ever worked. This chapter is the warmup — but every later chapter assumes this is in your bones.

---

## 1.1 What a neural network actually is — the story before the math

Before 2012, most "AI" you encountered in industry was hand-engineered features fed into a logistic regression or a random forest. You'd spend three weeks crafting "is_capitalized," "previous_word_pos_tag," "tf-idf_of_unigram" features for an NLP model, then train a classifier on top. The model itself was dumb; the human was the intelligence.

A neural network flips this. Instead of you designing the features, you design a *parameterized function* — a stack of matrix multiplications and squashing functions — and let gradient descent find the features automatically. The function f_theta(x) maps an input x to an output y, and the parameters theta (the weights) are adjusted so that f makes fewer mistakes on training data.

The mental model I keep in my head is this: a neural network is a programmable circuit. Each layer is a soldering job — multiply by a weight matrix, add a bias, run through a nonlinearity. Stack enough of these and you can approximate any function (the Universal Approximation Theorem from Cybenko 1989 and Hornik 1991 made this formal). Training is the act of finding good solder.

### The block diagram of a generic feed-forward net

```
            ┌───────────────┐
   x ─────▶ │  Linear: Wx+b │  ← parameters: W, b
            └───────┬───────┘
                    │ (pre-activation z)
            ┌───────▼───────┐
            │ Activation σ  │  ← non-linearity (ReLU, GELU, SiLU)
            └───────┬───────┘
                    │ (post-activation a)
                  (repeat N times)
                    │
            ┌───────▼───────┐
            │ Output head   │  ← softmax for classification, identity for regression
            └───────┬───────┘
                    │
                    ▼
                  ŷ (prediction)
                    │
            ┌───────▼───────┐
            │  Loss L(ŷ, y) │  ← MSE, cross-entropy, etc.
            └───────────────┘
```

### Why nonlinearity matters

People often miss this in interviews: if every layer is just `Wx + b`, then stacking N layers gives `W_N(W_{N-1}(...W_1 x) + ...) = W' x + b'` — which is **still a linear function**. No matter how deep, a stack of linear layers collapses to one linear layer. You'd be doing logistic regression with extra steps.

Nonlinear activations break that collapse. Once you bend the output of each layer, composition no longer simplifies, and the network can model curves, kinks, decision boundaries — anything you want.

### How to say this in an interview

> "A neural network is a learnable function approximator built from alternating linear maps and nonlinearities. The linear maps move information around in vector spaces; the nonlinearities are what give the model expressive power, because without them the entire stack would collapse to a single matrix multiplication. Training is just gradient descent over a differentiable loss, and backprop is the algorithm that makes that gradient computation efficient."

---

## 1.2 The forward pass with shapes — a 2-layer MLP, walked through

Let's make this concrete with an actual worked example. Imagine we're building a digit classifier on MNIST. Inputs are 28x28 images, flattened to 784-dim vectors. We want 10 outputs (one per digit class).

### The architecture

```
   x ∈ ℝ^784                           (input image, flattened)
        │
        │  W1 ∈ ℝ^(784 × 256), b1 ∈ ℝ^256
        ▼
   z1 = W1·x + b1   ∈ ℝ^256             (pre-activation, hidden layer)
        │
        │  σ = ReLU
        ▼
   a1 = ReLU(z1)    ∈ ℝ^256             (post-activation)
        │
        │  W2 ∈ ℝ^(256 × 10), b2 ∈ ℝ^10
        ▼
   z2 = W2·a1 + b2  ∈ ℝ^10              (logits)
        │
        │  σ = softmax
        ▼
   ŷ  = softmax(z2) ∈ ℝ^10              (probabilities, summing to 1)
```

### Numbers flowing through

Imagine batch size B=32. Then:

- Input batch X has shape (32, 784).
- W1 has shape (784, 256), so XW1 has shape (32, 256). After adding b1 (broadcast across batch) and applying ReLU element-wise, we still have (32, 256).
- W2 has shape (256, 10), so the next matmul gives (32, 10) — 32 logit vectors of length 10.
- Softmax row-wise turns each row into a probability distribution.
- Total trainable parameters: `784*256 + 256 + 256*10 + 10 = 200,704 + 256 + 2,560 + 10 = 203,530`. About 200K params — tiny by 2026 standards.

### Common mistakes / gotchas

1. **Forgetting batch dim in shapes**. Engineers fresh out of theory often write `Wx` and forget that in PyTorch you do `x @ W` because PyTorch convention has W shape (in, out), not (out, in).
2. **Applying softmax inside the loss again**. PyTorch's `nn.CrossEntropyLoss` already includes log-softmax. If you softmax-then-CE manually, you get a numerically unstable double-softmax.
3. **Using a single sample dimension during inference but batch during training**. Always test with `model.eval()` and a single-sample tensor of shape `(1, 784)` to make sure your shape plumbing works.

### Interview Q&A

**Q1. Walk me through a forward pass of a 2-layer MLP.**
> "Sure. Say we're classifying MNIST digits. The input image is flattened to a 784-dim vector. We multiply by W1 of shape 784x256, add a bias, apply ReLU element-wise — that gives us a 256-dim hidden representation. Then we multiply by W2 of shape 256x10, add another bias, and we get 10 logits. A softmax converts those logits into a probability distribution over the ten digit classes. With a batch of 32, every operation just gets a leading batch dimension, so the input is (32, 784) and the output is (32, 10)."

**Q2. Why exactly do we need a nonlinearity between the two linear layers?**
> "Because composing linear maps gives you another linear map. If I drop the ReLU, then the whole thing reduces to multiplication by W2 W1 plus a constant — that's just a single linear layer with extra parameters. The nonlinearity is what gives the network the universal approximation property: with one hidden layer of finite but sufficient width, an MLP can approximate any continuous function on a compact domain. ReLU specifically is cheap, gradient-friendly, and breaks linearity in a way that's worked spectacularly in practice."

**Q3. How many parameters in a layer with 784 inputs and 256 outputs?**
> "It's 784 times 256 for the weight matrix plus 256 for the bias vector — so 200,704 plus 256, which is 200,960. People often forget the bias term, but in interviews you want to mention it because biases matter for shifting decision boundaries, and you decide whether to include them based on whether you've already got LayerNorm or BatchNorm doing centering for you."

**Q4. What does it mean for a neural net to be a universal function approximator?**
> "Cybenko showed in 1989, and Hornik generalized in 1991, that a feed-forward network with a single hidden layer of sufficient width and any non-polynomial activation can approximate any continuous function on a compact set to arbitrary precision. That's the 'universal' part. In practice this is more existence theorem than recipe — depth tends to be more parameter-efficient than width for hard tasks, which is why we go deep in modern nets."

---

## 1.3 Backpropagation — the chain rule, slowly

People are scared of backprop because they remember the calculus. But the *idea* is dead simple: backprop is just the chain rule from calculus, applied recursively from the loss backwards through the computational graph, with intermediate activations cached so we don't recompute them.

### The mental model

Think of the forward pass as a Rube Goldberg machine. Marbles roll through pipes (linear layers) and over hills (activations) until they land in a target bucket (the loss). The loss tells you "you missed by this much." Backprop is the question: if I nudge each parameter by epsilon, how does that affect where the marble lands? It walks backwards through every step, accumulating partial derivatives via the chain rule.

### The block diagram of one layer's backward pass

```
   FORWARD                            BACKWARD
   ───────                            ────────
   x  ──────▶ ┌──────┐                  ┌──────┐ ◀──── ∂L/∂a (signal from above)
              │ z=Wx │                  │      │
              │  +b  │                  │      │
              └──┬───┘                  └──┬───┘
                 │ z                       │ δ = (∂L/∂a) ⊙ σ'(z)
                 ▼                          │
              ┌──────┐                  ┌──▼───┐
              │ σ(z) │                  │      │
              └──┬───┘                  └──┬───┘
                 │ a                       │
                 ▼                          ▼
              (next layer)              ∂L/∂W = δ · xᵀ
                                        ∂L/∂b = δ
                                        ∂L/∂x = Wᵀ · δ   (sent to layer below)
```

### The three formulas you should never forget

In plain English: when a loss signal `delta` arrives at a layer:

1. The gradient w.r.t. that layer's weight matrix W is the outer product of delta with the layer's input x (transposed).
2. The gradient w.r.t. that layer's bias is delta itself.
3. The gradient passed back to the previous layer's output is W transpose times delta.

```
∂L/∂W = δ · xᵀ    where δ = ∂L/∂z = (∂L/∂a) ⊙ σ'(z)
∂L/∂b = δ
∂L/∂x = Wᵀ · δ    ← the signal propagated backwards
```

The element-wise multiply with sigma-prime is where activation choice matters: if sigma-prime is zero (which happens for ReLU when z<0), the gradient dies. That's the dying ReLU problem.

### Worked example with numbers

Let's hand-compute one backprop step for a tiny net. Forward: `x = [1, 2]`, `W = [[0.5, -0.5], [1.0, 0.0]]`, `b = [0, 0]`, ReLU activation. Suppose true label says we want output `[1, 1]` and we use MSE.

Forward:
- `z = Wx + b = [0.5*1 + (-0.5)*2, 1.0*1 + 0.0*2] = [-0.5, 1.0]`
- `a = ReLU(z) = [0, 1.0]`
- `loss = (1/2) * ((0-1)^2 + (1-1)^2) = 0.5`

Backward:
- `dL/da = a - target = [0-1, 1-1] = [-1, 0]`
- `sigma'(z) = [0, 1]` (ReLU derivative: 1 if z>0 else 0)
- `delta = dL/da .elementwise* sigma'(z) = [-1*0, 0*1] = [0, 0]`
- `dL/dW = delta * x^T = [[0,0],[0,0]]` — no gradient! That first ReLU was dead.

This is the dying ReLU in microcosm. The neuron's pre-activation was negative, ReLU killed it, and now no gradient flows back. If this kept happening across many neurons, the model wouldn't learn at all.

### Why training takes ~3x more memory than inference

Backprop needs every intermediate activation from the forward pass to compute gradients. A 70B-parameter model's forward pass uses model weights + activations; the backward pass needs the same activations again to compute weight gradients, plus optimizer states (Adam keeps two moments per parameter). That's why training a 70B model needs hundreds of GB of GPU memory while inference fits on two H100s.

### Common mistakes / gotchas

1. **Forgetting the activation derivative**. Backprop through tanh has factor `1 - tanh^2(z)`; through sigmoid `sigmoid(z)*(1-sigmoid(z))`; through ReLU `1 if z>0 else 0`. People mix these up under interview pressure.
2. **Confusing `dL/dx` and `dL/dW`**. The first is for sending signal further back; the second is for actually updating the weights. Both come from delta but in different ways.
3. **Assuming gradients exist everywhere**. ReLU is non-differentiable at 0; argmax is non-differentiable everywhere; rounded outputs have zero gradient. Real numerical libs use subgradients (ReLU gets 0 at z=0).

### Resume tie-in

> When I was building the XGBoost lambda for TrueBalance, gradient flow was conceptually similar — gradient boosting fits each tree to the negative gradient of the loss w.r.t. the previous ensemble's prediction. The math is different from backprop in a deep net, but the *mindset* — "compute gradient, take a step, repeat" — is the same, and that intuition is what makes me confident jumping between classical ML and deep learning at production scale.

### Interview Q&A

**Q1. What is backprop, plainly?**
> "Backprop is the chain rule from calculus applied recursively to a computational graph. We do a forward pass, cache every intermediate value, then walk the graph backwards, computing the gradient of the loss with respect to every parameter by repeatedly multiplying local Jacobians. The cleverness isn't mathematical — it's the realization that we can reuse intermediate values from the forward pass to make the backward pass have the same asymptotic cost as the forward pass."

**Q2. Why does training use about three times more memory than inference?**
> "Inference only needs the model weights and the current activation in flight. Training needs to keep every intermediate activation from the forward pass so backprop can use them — that's typically the same size as the activations themselves, doubling memory. Then optimizers like Adam store running first and second moments per parameter, which adds another two times the model size. So roughly 3x to 4x the inference footprint, which is why we use ZeRO sharding and activation checkpointing for big models."

**Q3. What's the dying ReLU problem and how do you fix it?**
> "If a ReLU neuron's input becomes consistently negative across all training examples, its derivative is zero everywhere it sees data, so its weights never update — it's stuck dead. This is more common with high learning rates or bad initialization. Fixes are Leaky ReLU, which lets a small slope through for negative inputs; ELU and GELU which have smooth nonzero gradients on the negative side; better initialization like He init that keeps pre-activation variance reasonable; and just lower learning rates. In modern LLMs SwiGLU and GELU sidestep this entirely."

**Q4. Why do we need residual connections in deep networks?**
> "Without skip connections, a gradient flowing back through N layers gets multiplied by N Jacobians — if those Jacobians have spectral radius less than 1, the gradient vanishes; if greater than 1, it explodes. Residual connections add an identity path through each block, so the gradient has at least one route back to early layers without any matrix-multiplications. That's why ResNet could train 152 layers when nobody could train 30 before, and it's why every transformer block is wrapped in `x + Attention(x)` and `x + FFN(x)`."

**Q5. Compare gradient clipping to gradient normalization.**
> "Gradient clipping caps the L2 norm of the full gradient at some threshold like 1.0 — if the norm exceeds the cap, you scale the whole gradient down. This prevents the catastrophic loss spikes that come from data outliers or bad batches. Gradient normalization is a stronger version where you always rescale to a fixed norm, which is more stable but can erase scale information that gradient descent uses. Production LLM training uses clipping, almost universally at norm 1.0, and you'll see this in every public training recipe from LLaMA to Mistral."

---

## 1.4 Activation functions — when to use what, and why

Activation functions get glossed over because they're "just one line of code," but the choice has a real impact on training stability and final quality. Let me walk you through the lineage.

### The story arc

The earliest neural nets used sigmoid because it was smooth and bounded — a natural choice for "soft on/off." But sigmoid saturates: if `|z| > 5`, the derivative is essentially zero, so gradients vanish through deep stacks. Tanh fixed the centering issue (sigmoid outputs are all positive, which biases downstream layers) but still saturated. ReLU broke this in 2010-12 — it's non-saturating for positive inputs and dirt cheap to compute. AlexNet's 2012 ImageNet win popularized it.

Then BERT and GPT-2 (2018-19) used GELU because it's smoother than ReLU and gave slightly better language modeling perplexity. SiLU/Swish appeared next (Google, 2017) — `x * sigmoid(x)` — used in LLaMA and Mistral. Today's frontier is **SwiGLU**, a gated variant where the FFN block has the form `Swish(xW_gate) * (xW_up) * W_down`. The gating lets the model dynamically scale information flow per-token, and empirically it beats plain ReLU/GELU by a small but real margin in perplexity.

### The cheat sheet table

| Activation | Formula | Pros | Cons | Used in |
|------------|---------|------|------|---------|
| Sigmoid | `1/(1+e^-x)` | Smooth, bounded [0,1] | Saturates, not zero-centered | Old MLPs, gates in LSTM |
| Tanh | `(e^x - e^-x)/(e^x + e^-x)` | Zero-centered, bounded [-1,1] | Saturates | LSTM, old RNNs |
| **ReLU** | `max(0, x)` | Cheap, no saturation for x>0 | Dying ReLU for x<0 | CNNs (ResNet), early transformers |
| Leaky ReLU | `max(0.01x, x)` | Fixes dying ReLU | Extra hyperparam | Some GANs |
| ELU | `x if x>0 else α(e^x-1)` | Smooth, no dying | More compute | Niche |
| **GELU** | `x · Φ(x)` | Smooth, probabilistic interpretation | More compute | BERT, GPT-2, GPT-3 |
| **SiLU/Swish** | `x · sigmoid(x)` | Smooth, non-monotonic | More compute | LLaMA, Mistral FFN gate |
| **SwiGLU** | `Swish(xW_g) ⊙ (xW_u)·W_d` | Gated, SOTA in LLMs | 1.5x FFN params | LLaMA-2/3, Mistral, Gemma |

### How to say this in an interview

> "The big-picture story is: we started with sigmoid and tanh because they were mathematically nice, ran into vanishing gradients, switched to ReLU which is non-saturating and cheap, and that unlocked deep CNNs. For language models we found that smoother activations like GELU and SiLU train slightly more stably and give a small perplexity bump. Modern LLMs use SwiGLU in the feed-forward block — the gating mechanism lets the model dynamically scale information per-token, which is empirically worth one to two perplexity points compared to plain GELU. The cost is an extra projection matrix, so the FFN has three weight matrices instead of two."

### Common mistakes / gotchas

1. **Using tanh in transformer FFN.** It saturates and trains worse. GELU/SiLU/SwiGLU are correct.
2. **Treating GELU as a drop-in replacement for ReLU**. They have different magnitudes; some hyperparameters need re-tuning when you swap.
3. **Forgetting that sigmoid and tanh are still useful** — sigmoid for gating in LSTM/Mamba, tanh for bounded outputs in RL value heads.

### Interview Q&A

**Q1. Why did the field move from ReLU to GELU?**
> "ReLU has a hard kink at zero — its derivative jumps from 0 to 1 — and the dead-neuron problem happens whenever a neuron's pre-activation stays negative. GELU is `x times the standard normal CDF of x`, which is a smooth approximation of ReLU. In practice GELU gives a small but consistent perplexity improvement on language modeling tasks, which is why BERT and GPT-2 adopted it. The cost is roughly 2-3x more compute per activation, but that's negligible compared to attention."

**Q2. What's special about SwiGLU?**
> "SwiGLU is a gated activation: the FFN computes `Swish(xW_gate)` element-wise multiplied by `xW_up`, and that product goes through a final projection `W_down`. The intuition is that the Swish branch acts as a learned, per-position gate that decides how much of `xW_up` to let through. Empirically this gives one or two perplexity points over GELU at the same parameter count. PaLM, LLaMA, Mistral, and Gemma all use SwiGLU. The catch is that you need a third weight matrix in the FFN, so to keep parameter count constant you shrink the hidden dimension from `4d` to roughly `8/3 * d`."

**Q3. When would you actually use sigmoid in a modern architecture?**
> "Sigmoid lives on as a gating function. In LSTMs and GRUs, the input/forget/output gates are sigmoids because you want a value in [0,1]. In Mamba and other state-space models, you'll see sigmoid gates too. And in any binary classification head, a sigmoid output gives you a probability directly. What you do *not* want sigmoid for is hidden activations in deep nets — it saturates and gradients die."

**Q4. Why is softmax considered an activation function?**
> "Softmax converts a vector of arbitrary scores into a probability distribution that sums to one. It's the exponentiate-then-normalize trick. We use it as the final activation for multiclass classification or as part of attention to convert similarity scores into attention weights. It's smooth and differentiable, so gradients flow through it cleanly. Two gotchas: it's translation-invariant — adding a constant to every input doesn't change the output, which is why we subtract the max for numerical stability — and the temperature parameter inside controls how peaked the distribution is."

---

## 1.5 Loss functions — the four you must know cold

A loss function tells the model how wrong its prediction is. Different tasks have different "wrongness" definitions, and choosing the right one is half the battle.

### Mental model

Think of the loss as a thermostat. The model adjusts its parameters to drive the thermostat reading down. If you pick the wrong thermostat — say MSE for a classification problem — the model still trains, but towards a sub-optimal target.

### The four-quadrant cheat sheet

```
   ┌────────────────────┬────────────────────┐
   │ Regression          │ Classification     │
   │                     │                    │
   │  MSE                │  Cross-entropy     │
   │  MAE / Huber        │  BCE (binary)      │
   │  Quantile loss      │  Focal loss        │
   ├────────────────────┼────────────────────┤
   │ Embedding /         │ RL / Preference    │
   │  Retrieval          │                    │
   │                     │                    │
   │  Triplet            │  Policy gradient   │
   │  InfoNCE            │  PPO with KL       │
   │  Margin             │  DPO logistic      │
   └────────────────────┴────────────────────┘
```

### 1.5.1 Cross-entropy — derived from maximum likelihood

This is *the* loss for classification and language modeling. Let me walk through why.

Suppose the true label is class k out of C classes, and your model outputs probabilities `p1, ..., pC`. The likelihood of the observed label under the model is `pk`. To maximize likelihood, you maximize `log(pk)`. To turn that into a *minimization* problem (which optimizers expect), you minimize `-log(pk)`. That's the cross-entropy loss for one example.

In one-hot notation where `yi = 1 if i == k else 0`:
```
L_CE = - Σ_i y_i · log(p_i)
```

For language modeling with next-token prediction, this becomes:
```
L = - Σ_t log P(x_t | x_<t)
```

And during training we sum over every token in every sequence. This is the *only* loss used to pretrain LLMs.

### Why softmax + CE is numerically safe

Logits can be huge (e.g., 100). `exp(100)` overflows float32. The "log-softmax" trick subtracts the max logit before exponentiating, then combines softmax and log into a single fused op called `log_softmax`. PyTorch's `nn.CrossEntropyLoss(input=logits, target=labels)` does exactly this — it expects raw logits, not probabilities. Forget that, apply softmax then CE manually, and you've double-softmaxed.

### 1.5.2 MSE — for regression

```
L_MSE = (1/N) · Σ_i (y_i - ŷ_i)^2
```

Penalizes large errors quadratically. If your data has outliers, MSE will bend the model to fit them. Use MAE (L1) or Huber (quadratic near zero, linear far from zero) for outlier robustness.

### 1.5.3 Triplet / contrastive — for embeddings

If you don't have absolute labels but you do know "A is more similar to P than to N," the right loss is triplet:

```
L_triplet = max(0, d(A, P) - d(A, N) + α)
```

Anchor `A`, positive `P`, negative `N`, margin `α`. The model learns to make `d(A,P) < d(A,N) - α`. This is the foundation of face recognition (FaceNet), embedding models, and metric learning.

InfoNCE is a generalization with N negatives instead of one:
```
L_InfoNCE = - log [ exp(sim(A,P)/τ) / Σ_j exp(sim(A, p_j)/τ) ]
```

We'll go deep on this in Chapter 04.

### Common mistakes / gotchas

1. **Using MSE for classification**. Mathematically valid but terrible — gradients are weak when the prediction is confidently wrong, slowing training.
2. **Forgetting class imbalance**. Vanilla CE on a 99:1 imbalanced dataset still trains, but the model is biased towards majority. Use class weights or focal loss.
3. **Mixing up logits and probabilities**. PyTorch's BCEWithLogitsLoss expects logits; BCELoss expects probabilities. Mismatch = NaN losses or wrong gradients.

### Interview Q&A

**Q1. Why is cross-entropy the right loss for classification rather than MSE?**
> "Cross-entropy is the negative log-likelihood under the softmax output, so minimizing it is exactly maximum-likelihood estimation. It also has a nicer gradient — the gradient of CE with respect to the pre-softmax logit is just `predicted_prob - true_label`, which is bounded and well-scaled regardless of how confident the wrong prediction is. MSE on softmax outputs gives a gradient that vanishes when the model is confidently wrong, because the softmax saturates. Empirically, CE trains faster and converges to better accuracy on every standard benchmark."

**Q2. Walk me through deriving cross-entropy from MLE.**
> "Sure. We're modeling P(y given x) as the softmax of the model's logits. The likelihood of observing the dataset is the product over all examples of the probability our model assigns to the true label. We maximize log-likelihood, which is the sum of log-probabilities. To turn it into a minimization, we negate, giving us the sum of negative log-probabilities of the true labels. That's the cross-entropy loss. So minimizing CE is identical to maximum likelihood — same recipe, different sign."

**Q3. Why is LLM pretraining loss just cross-entropy?**
> "Because next-token prediction is multiclass classification — at each position, you predict one of V vocabulary tokens, and the loss for that position is the negative log-probability the model assigned to the correct token. Sum over all positions in all sequences, divide by token count, and you get average cross-entropy per token, also known as nats per token. Exponentiating gives perplexity. So everything from GPT-2 to LLaMA-3 to Claude trains under the same loss — just at very different scales."

**Q4. When would you use MAE or Huber instead of MSE?**
> "When your data has outliers and you don't want them to dominate. MSE penalizes errors quadratically, so a single point with huge error pulls the fit hard. MAE penalizes linearly, which is robust but has discontinuous gradient at zero. Huber is the best of both: quadratic near zero so gradients are smooth, linear far from zero so outliers don't dominate. For real-world tabular regression — say, predicting LTV in our credit-scoring pipeline — Huber or quantile loss often beats MSE on heavy-tailed targets."

**Q5. What's a contrastive loss intuitively?**
> "It's a loss that doesn't need absolute labels — only relative. Pull positive pairs together, push negative pairs apart in embedding space. The simplest form is triplet: anchor, positive, negative, with a margin. The modern SOTA is InfoNCE, which uses one positive and many negatives, framed as a softmax over similarities. This is how every modern embedding model — SBERT, BGE, E5, OpenAI text-embedding-3 — is trained, and we'll go deep on it in chapter four."

**Q6. What is label smoothing and when do you use it?**
> "Label smoothing replaces hard one-hot labels with soft labels — instead of `[0, 0, 1, 0]` you train against `[0.025, 0.025, 0.925, 0.025]`. This prevents the model from becoming over-confident, which empirically improves calibration and sometimes accuracy. It's especially useful in machine translation and image classification where over-confident wrong predictions are a real failure mode. The tradeoff is a small bias — the model never quite outputs probability 1, which can hurt if downstream code thresholds at 0.99."

---

## 1.6 Word embeddings — before transformers ate the world

Before BERT and the transformer revolution, the central question of NLP was: how do I turn a word into a vector that has useful properties? You can't just dump words into a neural net; you need a numerical representation, and the better the representation, the easier the downstream task.

### 1.6.1 The naive baseline: one-hot

A vocabulary of 50,000 words means each word becomes a 50,000-dim vector with a single 1 and the rest zeros. The dot product of any two distinct words is zero, so "cat" is exactly as similar to "dog" as it is to "pizza." Useless for semantics. And catastrophic for memory.

### 1.6.2 Word2Vec — distributional semantics, finally working

Mikolov et al.'s 2013 paper was a landmark. The idea: train a shallow neural net to predict context from a word (or word from context), and use the hidden layer's weights as word vectors. The vectors learn semantic structure because words with similar contexts ("cat" and "dog" both appear near "fluffy," "pet," "bowl") end up with similar embeddings.

There are two architectures: CBOW and Skip-gram.

### CBOW (Continuous Bag of Words)

Predict the **center** word from its context.

```
   Context:  [the, quick, brown, ___, jumps, over, the, lazy]
                │      │      │            │     │     │     │
                ▼      ▼      ▼            ▼     ▼     ▼     ▼
            ┌─────────────────────────────────────────────────┐
            │  Sum or average of context word embeddings      │
            └────────────────────────┬────────────────────────┘
                                     │ (avg vector)
                              ┌──────▼───────┐
                              │  Linear → V  │  (V = vocab size)
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   Softmax    │
                              └──────┬───────┘
                                     │
                                     ▼
                       Predict: "fox"  (the missing center word)
```

Faster to train (averages context, smooths over rare patterns). Better for frequent words.

### Skip-gram

Predict each **context** word from the center word.

```
                                       ┌──▶ Predict "the"
                                       ├──▶ Predict "quick"
            ┌────────────┐    ┌────┐   ├──▶ Predict "brown"
   "fox" ──▶│ Embedding  │───▶│ FC │───┤
            │  E[fox]    │    └────┘   ├──▶ Predict "jumps"
            └────────────┘             ├──▶ Predict "over"
                                       └──▶ ...
```

Slower but better for rare words because each (center, context) pair is a separate training example.

### The negative sampling trick

The vocab is 50K-1M words. Computing softmax over all of them per training example is brutal. Negative sampling reformulates the problem: instead of "which of V words is the context," ask "is this (center, context) pair real or fake?" For each positive pair, sample 5-20 negative pairs (random words). The loss becomes:

```
L = - log σ(v_center · v_context_pos) - Σ_neg log σ(-v_center · v_neg)
```

This is logistic regression on (real, fake) pairs. Much cheaper and works empirically nearly as well as full softmax.

### The famous analogy property

Word2Vec embeddings have linear structure: `king - man + woman ≈ queen`, `Paris - France + Italy ≈ Rome`. This emerged from training, not by design, and it stunned the field. The intuition: gendered/locational/temporal axes line up consistently across the embedding space, because the same contextual cues distinguish them.

### 1.6.3 GloVe — the matrix-factorization view

Stanford, 2014. Pennington, Socher, Manning. Their argument: Word2Vec is local — it slides a window over the corpus. Why not look at the global statistics of word co-occurrence directly?

Build the matrix X where `X_ij` = number of times word j appears in the context of word i across the entire corpus. Then learn embeddings such that:

```
w_i^T · w_j + b_i + b_j ≈ log(X_ij)
```

The loss is a weighted MSE that down-weights rare co-occurrences:
```
L = Σ_ij f(X_ij) · (w_i^T w_j + b_i + b_j - log X_ij)^2
```

GloVe often beats Word2Vec slightly on analogy tasks but is more memory-hungry to train (need the full co-occurrence matrix). Word2Vec dominates in practice because it streams.

### 1.6.4 FastText — handling OOV via subwords

Word2Vec assigns each whole word an embedding. New word? Tough luck. Mis-spelled word? Same. Facebook's FastText (Bojanowski et al., 2016) fixes this by representing each word as the sum of its character n-grams.

```
"apple" → < <a, ap, pp, pl, le, e> > + 3-grams + 4-grams + ...
        = sum of n-gram embeddings
```

Now `"apppple"` (a typo) decomposes into mostly the same n-grams as `"apple"` and gets a sensible embedding. Even an unseen word like `"blockchain"` gets a reasonable vector because its subwords appeared in training.

This is huge for morphologically rich languages — Arabic, Finnish, Turkish — where one root has dozens of inflected forms, and Word2Vec would have to learn them all separately.

### 1.6.5 Static vs contextual embeddings

Word2Vec, GloVe, FastText all produce **static** embeddings — one vector per word, regardless of context. So `bank` in "river bank" and `bank` in "deposit money at the bank" gets the same vector. That's wrong, and fixing it required transformers.

```
Static:                      Contextual:
"river bank"   ─▶ v_bank     "river bank"   ─▶ v_bank_1   (river-ish)
"savings bank" ─▶ v_bank     "savings bank" ─▶ v_bank_2   (financial)
                              (BERT computes a different vector per occurrence)
```

Modern transformers (BERT, GPT, Claude) produce a different vector for every token at every layer, depending on its context. This is what makes them so much more powerful than Word2Vec.

### 1.6.6 When would I still use Word2Vec / GloVe today?

1. **Latency-critical retrieval** where running BERT is too slow. TF-IDF + averaged Word2Vec can be 100x faster.
2. **Edge devices** without GPUs.
3. **As features for classical ML** like XGBoost or logistic regression.
4. **Baselines in research** to show your fancy model actually beats the simple thing.

### Common mistakes / gotchas

1. **Confusing CBOW and Skip-gram directions**. CBOW = context-to-center, Skip-gram = center-to-context. People flip these in interviews.
2. **Forgetting the negative sampling**. Without it, training is intractable, and many people just gloss over how the loss works.
3. **Treating Word2Vec as "learning a language model"**. It's not — it's learning representations from co-occurrence. The resulting vectors are useful, but the "model" isn't doing language modeling.

### Resume tie-in

> When I built the NER lender extractor at TrueBalance, I started with a baseline that used FastText embeddings averaged into a simple bidirectional LSTM tagger. That got us to about 30% F1 on lender entities. Switching to a fine-tuned BERT — i.e., contextual embeddings — pushed us to 68%. The jump was almost entirely about disambiguation: same word, different meaning depending on neighbor tokens, which static embeddings simply can't represent.

### Interview Q&A

**Q1. Explain Word2Vec in plain English.**
> "Word2Vec is a shallow neural net trained to predict context from a word, or a word from its context. The hidden-layer weights end up being good word embeddings. Words appearing in similar contexts — like 'cat' and 'dog' both appearing near 'fluffy' — end up with similar vectors. The famous result is that simple arithmetic in this space reveals semantic structure: `king - man + woman` lands near `queen`. There are two flavors: CBOW predicts center from context, faster but slightly worse on rare words; Skip-gram predicts context from center, slower but better for rare words. Both use negative sampling so we don't softmax over the full vocab every step."

**Q2. What does the negative sampling objective actually look like?**
> "It's binary classification on (real, fake) pairs. For each (center, true_context), sample five to twenty (center, random_word) pairs as negatives. The loss is `-log sigmoid(v_center dot v_pos) - sum over negatives of log sigmoid(-v_center dot v_neg)`. This pulls real co-occurring pairs closer in dot-product space and pushes random pairs apart. It's much cheaper than full softmax over a 1M-word vocab — that's the whole reason Word2Vec was fast enough to train on web-scale corpora."

**Q3. Why does GloVe sometimes beat Word2Vec?**
> "GloVe explicitly factorizes the global word co-occurrence matrix, so it leverages corpus-wide statistics directly. Word2Vec is local: it slides a window over text and only sees pairs within that window. On analogy tasks, GloVe typically scores a couple points higher because the global view captures relationships that aren't always captured by local windows. The catch is that GloVe needs you to materialize the co-occurrence matrix, which is memory-hungry. Word2Vec streams. In practice today, neither matters much because contextual embeddings have replaced both."

**Q4. Why was FastText a big deal?**
> "FastText represents each word as the sum of its character n-gram embeddings. So `'apple'` is the sum of n-grams like `'app'`, `'ppl'`, `'ple'`. Two big benefits. First, out-of-vocabulary handling: a typo or new word still gets a reasonable embedding because its n-grams appeared in training. Second, morphological richness: Finnish, Turkish, Arabic have hundreds of word forms per root, and FastText handles them naturally. Word2Vec would need to see every form. The cost is a larger model and slower training, but for multilingual or noisy text it's the right baseline."

**Q5. Why did contextual embeddings replace static ones?**
> "Polysemy. Words like 'bank' or 'bat' or 'lead' have multiple meanings, and a single vector can't represent all of them. ELMo in 2018 was the first to show that running a bidirectional LSTM over the sentence and using its hidden state per token gives you a context-aware embedding. BERT generalized this with attention, and now every modern LLM does this internally — every hidden state at every layer is a contextual embedding. We use Word2Vec only as a baseline now."

**Q6. Can Word2Vec handle out-of-vocabulary words?**
> "No — and that's its big weakness. Word2Vec assigns embeddings only to words that appeared in the training vocabulary. New words, typos, brand names, hashtags — all get an `<UNK>` token or are dropped. FastText fixes this by composing word embeddings from character n-grams, so any string of characters gets a representation. Modern subword tokenizers like BPE solve the same problem differently, by splitting unknown words into known subword pieces."

---

## 1.7 Optimizers — pick the right hammer

The optimizer is the algorithm that turns gradients into parameter updates. For LLMs, the choice is essentially settled: **AdamW**. But you should know why.

### The lineage

- **SGD**: `theta = theta - lr * gradient`. Brutally simple. Slow on noisy or ill-conditioned losses.
- **SGD with momentum**: `v = beta * v + g; theta = theta - lr * v`. Accumulates a velocity that pushes through saddle points and noisy gradients. Default for CV (ResNet etc.).
- **Adam**: maintains per-parameter running averages of gradients and squared gradients (`m` and `v`), uses them to compute an adaptive learning rate per parameter. Robust to varied loss landscapes. Default for everything except CNNs.
- **AdamW**: Adam with **decoupled weight decay**. The fix that matters for transformers.

### Why AdamW and not Adam for LLMs

In vanilla Adam, L2 weight decay is added to the gradient, then divided by `sqrt(v)` like everything else. That means weight decay's effective strength varies per parameter — which breaks regularization. AdamW separates weight decay from the gradient update: `theta = theta - lr * (Adam_step + weight_decay * theta)`. Now decay is applied uniformly. Empirically this is consistently better for transformers, and every public LLM training recipe uses it.

### Cheat sheet

| Optimizer | Memory cost | Best for | Notes |
|-----------|------------|----------|-------|
| SGD + momentum | 1x params | CNNs, simple tasks | Needs careful LR tuning |
| Adam | 2x params (m, v) | Most ML | Adaptive LR per param |
| **AdamW** | 2x params | **Transformers, LLMs** | Decoupled weight decay |
| Lion | 1x params (only sign of momentum) | Memory-bound large training | Sign-based, surprisingly competitive |
| Adafactor | Sub-linear | Very large models | Used in T5; trades memory for stability |
| Sophia | 2x params + Hessian estimate | Cutting-edge LLM training | ~2x faster convergence in some recipes |

### Common mistakes / gotchas

1. **Using Adam (not AdamW) and being confused why weight decay isn't working.** Always AdamW for transformers.
2. **Forgetting LR warmup**. AdamW's initial steps are noisy because `m` and `v` haven't accumulated. Linear warmup over 500-2000 steps is standard.
3. **Default beta values aren't always right**. `beta1=0.9, beta2=0.95` is common for LLMs; `beta2=0.999` (the default) can be slow to adapt to changing loss landscape.

### Interview Q&A

**Q1. Why AdamW and not plain Adam?**
> "Vanilla Adam folds L2 weight decay into the gradient, which then gets scaled by the per-parameter adaptive learning rate. That means parameters with large `v` get less effective decay, and parameters with small `v` get more — inconsistent regularization. AdamW decouples weight decay from the gradient update: it scales the parameters down by `lr * weight_decay * theta` independently of the Adam step. This restores uniform regularization, and empirically it's consistently better for transformers. Every public LLM recipe — LLaMA, Mistral, Qwen — uses AdamW."

**Q2. Why momentum helps SGD.**
> "Momentum is exponential averaging of gradients. The update direction is no longer just the current gradient but a smoothed history. This helps in three ways: it pushes through narrow ravines where the gradient flips sign each step, it helps escape saddle points, and it accelerates progress along consistent directions. In CV, SGD with momentum (typically 0.9) plus cosine LR schedule is still the recipe of choice for ResNet-style training. For LLMs we use AdamW because the loss landscape is more heterogeneous and adaptive LR per parameter helps more."

**Q3. What's a typical LLM training schedule?**
> "AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, weight_decay=0.1. Linear warmup for the first 500 to 2000 steps, then cosine decay to 10% of the peak LR over the remaining tokens. Gradient clipping at norm 1.0. Mixed precision in BF16. That's basically the LLaMA recipe and most labs use a close variant. The numbers come from extensive empirical work — small changes to beta2 or warmup steps can break long training runs."

**Q4. Have you used Lion or Sophia?**
> "I've experimented with Lion on smaller models. It's appealing because it only stores the sign of the momentum, halving optimizer memory — useful when you're memory-constrained on H100. The reported quality is comparable to AdamW. Sophia uses a Hessian estimate and claims faster convergence; some papers show ~2x training speedup on LLMs but production recipes haven't fully migrated. AdamW remains the safe default I'd pick for a production training run unless we had a specific memory or speed requirement."

---

## 1.8 Normalization — BatchNorm, LayerNorm, RMSNorm

Normalization layers stabilize training by keeping activations from exploding or vanishing across layers. The choice is dictated by your data shape and architecture.

### The three flavors

```
BatchNorm:  normalize across the BATCH for each feature
            (x - μ_batch_per_feature) / σ_batch_per_feature

LayerNorm:  normalize across FEATURES for each sample
            (x - μ_per_sample) / σ_per_sample

RMSNorm:    no mean centering, just scale by RMS
            x / RMS(x)   where RMS(x) = sqrt(mean(x^2) + eps)
```

### Visualization

```
Input shape (B, T, D): batch=B, seq_len=T, features=D

BatchNorm:    average over (B, T)              for each D
              (assumes B is meaningful, fixed-length sequence; bad for NLP)

LayerNorm:    average over D                   for each (B, T)
              (per-token, per-sample; works with variable lengths)

RMSNorm:      same axes as LayerNorm but no mean subtraction
              (cheaper, no quality loss in practice)
```

### Why BatchNorm fails in transformers

- Sequences vary in length; padding tokens skew the batch statistics.
- Inference often uses batch=1 (single user request), where BatchNorm degenerates.
- Sequences in a batch have different roles (chat vs document), so per-feature stats don't share well.

LayerNorm solves all of these by normalizing per-token, per-sample. It works at any batch size and handles variable lengths naturally.

### Why RMSNorm is taking over

Zhang and Sennrich (2019) showed that LayerNorm's mean-centering step contributes little to its effectiveness. Drop the mean subtraction and you save a few percent compute, no quality loss. LLaMA-2/3, Mistral, Qwen, Gemma all use RMSNorm.

### Pre-LN vs Post-LN

```
Post-LN (original "Attention Is All You Need"):
    x ──▶ Attention ──▶ + ──▶ LayerNorm ──▶ next
                        ▲
                        x

Pre-LN (modern):
    x ──▶ LayerNorm ──▶ Attention ──▶ + ──▶ next
                                       ▲
                                       x
```

Post-LN performs slightly better when training is stable but is hard to train at depth — it needs careful LR warmup. Pre-LN is more stable for deep stacks (100+ layers) and is the modern default.

### Interview Q&A

**Q1. Why LayerNorm and not BatchNorm in transformers?**
> "Three reasons. First, sequences are variable length and padding skews batch statistics. Second, inference is often batch=1, where BatchNorm has no meaningful batch to average over. Third, transformers process per-token, and the natural axis to normalize is the feature dimension within a token, not the batch. LayerNorm fits all three. RMSNorm goes further by dropping mean-centering for a small speedup with no quality loss — that's why LLaMA, Mistral, and most modern LLMs use RMSNorm."

**Q2. Pre-LN vs Post-LN — which to use?**
> "Pre-LN places the norm inside the residual block, before the attention or FFN. Post-LN places it after the residual addition. Pre-LN is more stable for deep nets — the gradient has a clean residual path that doesn't go through the norm. Post-LN gives slightly better final quality when training is stable, but requires careful LR warmup or scaled initialization to avoid divergence. Modern LLMs universally use Pre-LN. There's a hybrid called sandwich-LN that some labs use, but it's niche."

**Q3. What's the actual formula for RMSNorm?**
> "It's `x / sqrt(mean(x^2) + epsilon) times gamma`, where gamma is a learnable scale parameter. There's no mean subtraction and no learnable bias, unlike LayerNorm which has both. The reasoning is that mean-centering costs computation and barely changes results — Zhang and Sennrich showed this empirically in 2019. For a 70B model, RMSNorm is a few percent faster end-to-end with no quality loss, which adds up over a multi-week training run."

**Q4. Why does normalization help in the first place?**
> "Two reasons. First, it stabilizes the scale of activations across layers — without it, values can grow or shrink exponentially through a deep stack, and the optimizer has to fight that. Second, it enables larger learning rates by keeping the loss landscape better-conditioned in the directions normalization controls. The third underrated benefit is that it makes the model more robust to bad initialization — even a poorly-initialized network can train with normalization, where it would diverge without."

---

## 1.9 Bias-variance tradeoff — the diagram everyone draws

```
                              Total Error
        Error
          │       ╲                  ╱
          │        ╲    Variance  ╱
          │ Bias²   ╲           ╱
          │          ╲         ╱
          │           ╲       ╱
          │            ╲     ╱
          │             ╲___/   ◀── sweet spot
          │
          └──────────────────────────────▶
            simple model        complex model
            (underfit)          (overfit)
```

- **High bias / underfit**: train loss high, val loss high. Solutions: add capacity, train longer, better features.
- **High variance / overfit**: train loss low, val loss high. Solutions: more data, regularization (weight decay, dropout), early stopping, simpler model.

### LLMs break the classical curve

Modern LLMs sit far past the classical "optimum" — they have orders of magnitude more parameters than data points and yet generalize. Phenomena like **double descent** (loss has a second descent past the interpolation threshold) and **grokking** (validation accuracy jumps long after train loss plateaus) muddle the classical view. The modern recipe is: scale data and model together, regularize lightly, and trust the scaling laws.

### Interview Q&A

**Q1. Your model overfits — what do you try, in priority order?**
> "First, more data — that's the single most impactful lever, every time. Second, regularization: weight decay, dropout, label smoothing. Third, simplify the model if data is genuinely scarce. Fourth, early stopping — monitor validation loss and stop when it plateaus. Fifth, data augmentation if it's a CV or speech task. For LLMs specifically, the answer is almost always more and better data before scaling the model — that's what the Chinchilla paper showed and what every frontier lab has internalized."

**Q2. What's double descent?**
> "Classical theory says val loss is U-shaped in model complexity — first decreases, then increases past the sweet spot. But for very over-parameterized models, val loss has a second descent: it goes up, peaks at the interpolation threshold (the point where the model can perfectly fit train data), then comes back down as you keep growing the model. This is one reason LLMs work so well despite having far more parameters than tokens — they live well past the classical sweet spot, in the second-descent regime."

---

## 1.10 Evaluation metrics — classification & retrieval

### Classification

- **Accuracy**: `(TP + TN) / total`. Useless on imbalanced data.
- **Precision**: `TP / (TP + FP)`. "Of what I said yes, how many were right?"
- **Recall**: `TP / (TP + FN)`. "Of all true yes, how many did I catch?"
- **F1**: harmonic mean of P and R. Balanced view.
- **AUC-ROC**: area under TPR-FPR curve. Threshold-independent.
- **AUC-PR**: better for imbalanced data (fraud, default prediction at TrueBalance).

### Retrieval / ranking

- **Recall@k**: of all relevant docs, what fraction made top-k.
- **Precision@k**: of top-k, what fraction are relevant.
- **MRR** (mean reciprocal rank): `1 / position_of_first_relevant`, averaged. Good for "one right answer" tasks.
- **nDCG@k**: discounted cumulative gain. Industry standard for ranking.

### LLM / generation

- **Perplexity** = `exp(avg cross-entropy)`. Lower = better LM.
- **BLEU / ROUGE**: n-gram overlap. Brittle for open-ended generation.
- **LLM-as-judge**: dominant in 2025-2026 for open-ended quality.
- **RAGAS / TruLens**: factuality, faithfulness, retrieval quality for RAG.

### Common gotcha

> Accuracy on imbalanced data lies. If 99% of loans don't default, predicting "no default" always scores 99%. Always check precision, recall, PR-AUC for fraud-style tasks.

### Interview Q&A

**Q1. When does accuracy mislead?**
> "On imbalanced data. If 99% of users don't default, a model that always predicts 'no default' has 99% accuracy and zero usefulness. You want precision, recall, and PR-AUC instead. At TrueBalance, the default rate on certain segments was about 3%, and the difference between a 97% accurate model and a 90% accurate one was almost entirely in how well it caught actual defaulters. So we tracked PR-AUC and recall at fixed precision, never raw accuracy."

**Q2. AUC-ROC vs AUC-PR — when to use which?**
> "AUC-ROC plots true positive rate vs false positive rate. It's threshold-independent and works well when classes are balanced. But on heavily imbalanced data, FPR stays small even when the model is bad on the minority class, so AUC-ROC looks artificially high. AUC-PR (precision vs recall) is much more sensitive to the minority class and is the better metric for fraud, churn, or rare-event detection. I default to PR-AUC for any task with imbalance worse than 10:1."

**Q3. Why is nDCG the standard for ranking?**
> "Because it weighs higher positions more heavily — getting the right answer at rank 1 is worth more than at rank 5. The formula sums `relevance_i / log2(i+1)` over the top-k, normalized by the ideal DCG. So a perfect ranking scores 1.0, and you can compare across queries. Recall@k just asks 'is the right answer in top-k' which doesn't care about ordering within top-k. For RAG, I usually report both — nDCG@10 for ranking quality and Recall@10 for coverage."

**Q4. Why is perplexity the standard LLM metric?**
> "Perplexity is `exp(average cross-entropy per token)`. It's interpretable as 'how many tokens would the model be uniformly uncertain among, on average.' Perplexity 10 means the model is, on average, choosing among 10 equally-likely tokens at each position. Lower is better. The reason it's standard is that it's directly tied to the training loss — you can compare apples-to-apples across models on the same data. It does have limits — perplexity doesn't capture instruction-following or reasoning quality, which is why we also rely on LLM-as-judge and human eval for downstream tasks."

---

## 1.11 Resume tie-ins

> When asked about foundations, here's how I'd weave my actual experience:

- **TrueBalance XGBoost lambda (p99 < 500ms)**: Tree-based, not neural, but the bias-variance intuition applies directly — XGBoost's depth, learning rate, and regularization (alpha for L1, lambda for L2) are exactly the levers you'd discuss in a deep-learning context. I tuned these against PR-AUC because we had a 3% positive class.
- **NER lender extractor (29.7% → 68% F1)**: A classic sequence-labeling task. We tried averaged FastText + BiLSTM as a baseline, then moved to fine-tuned BERT, and that contextual-vs-static jump was where the bulk of the F1 gain came from. Great example of static vs contextual embeddings in production.
- **ResMed RAG chatbot**: Embeddings everywhere — sentence-BERT for retrieval, contrastive fine-tuning on medical (query, passage) pairs. Most of chapter 04 is what we lived.
- **Sopra Steria CV/OCR**: Different domain (vision) but same fundamentals — CNN backbone, BatchNorm because batch=32 was reasonable, cross-entropy for character classification. The transferable lesson is that the right normalization and loss matters as much as architecture.

---

Continue to **[Chapter 02 — Transformers Deep Dive](02_transformers.md)**.
