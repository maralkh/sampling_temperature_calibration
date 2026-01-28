# Derivation of Optimal Temperature for Noisy Softmax

## Setup

- Clean logits: $z \in \mathbb{R}^V$ with $\text{Var}(z) = \tau^2$
- Noise: $\varepsilon \in \mathbb{R}^V$ with $\text{Var}(\varepsilon_i) = \sigma^2$, i.i.d., zero mean
- Noisy logits: $\tilde{z} = z + \varepsilon$
- Clean distribution: $p = \text{softmax}(z)$
- Noisy distribution at temperature $T$: $q = \text{softmax}(\tilde{z}/T)$

**Goal:** Find $T^*$ that minimizes $\text{KL}(p \| q)$.

---

## Step 1: Fisher Information Approximation

For small perturbations, KL divergence can be approximated using Fisher information:

$$\text{KL}(p \| q) \approx \frac{1}{2} \delta^T F \delta$$

where $F$ is the Fisher information matrix and $\delta$ is the difference in logit space.

---

## Step 2: Define δ

We compare $p = \text{softmax}(z)$ with $q = \text{softmax}((z+\varepsilon)/T)$.

The "effective" difference in logit space:

$$\delta = z - \frac{z+\varepsilon}{T} = z\left(1 - \frac{1}{T}\right) - \frac{\varepsilon}{T}$$

---

## Step 3: Fisher Matrix for Softmax

For softmax, the Fisher information matrix is:

$$F_{ij} = p_i(\delta_{ij} - p_j)$$

---

## Step 4: Compute $\delta^T F \delta$

$$\delta^T F \delta = \sum_{i,j} \delta_i F_{ij} \delta_j = \sum_i p_i \delta_i^2 - \left(\sum_i p_i \delta_i\right)^2 = \text{Var}_p(\delta)$$

So:

$$\text{KL}(p \| q) \approx \frac{1}{2} \text{Var}_p(\delta)$$

---

## Step 5: Expand $\text{Var}_p(\delta)$

With $\delta_i = z_i(1-1/T) - \varepsilon_i/T$:

$$\mathbb{E}_p[\delta] = \left(1-\frac{1}{T}\right)\mathbb{E}_p[z] - \frac{1}{T}\mathbb{E}_p[\varepsilon]$$

$$\mathbb{E}_p[\delta^2] = \left(1-\frac{1}{T}\right)^2\mathbb{E}_p[z^2] + \frac{1}{T^2}\mathbb{E}_p[\varepsilon^2] - 2\left(1-\frac{1}{T}\right)\frac{1}{T}\mathbb{E}_p[z\varepsilon]$$

---

## Step 6: Independence Assumption

Assuming $\varepsilon$ is independent of $z$:

$$\mathbb{E}_p[z\varepsilon] = \mathbb{E}_p[z] \cdot \mathbb{E}[\varepsilon] = 0$$

(since $\mathbb{E}[\varepsilon] = 0$)

---

## Step 7: Take Expectation over ε

$$\mathbb{E}_\varepsilon[\mathbb{E}_p[\varepsilon]] = \sum_i p_i \mathbb{E}[\varepsilon_i] = 0$$

$$\mathbb{E}_\varepsilon[\mathbb{E}_p[\varepsilon^2]] = \sum_i p_i \sigma^2 = \sigma^2$$

---

## Step 8: Expected $\text{Var}_p(\delta)$

$$\mathbb{E}_p[\delta] = \left(1-\frac{1}{T}\right)\mathbb{E}_p[z]$$

$$\mathbb{E}_p[\delta]^2 = \left(1-\frac{1}{T}\right)^2 \mathbb{E}_p[z]^2$$

$$\mathbb{E}_\varepsilon[\mathbb{E}_p[\delta^2]] = \left(1-\frac{1}{T}\right)^2\mathbb{E}_p[z^2] + \frac{\sigma^2}{T^2}$$

Therefore:

$$\mathbb{E}_\varepsilon[\text{Var}_p(\delta)] = \left(1-\frac{1}{T}\right)^2 \underbrace{(\mathbb{E}_p[z^2] - \mathbb{E}_p[z]^2)}_{\text{Var}_p(z)} + \frac{\sigma^2}{T^2}$$

---

## Step 9: Define Constants

Let:
- $A = \text{Var}_p(z)$ (variance of logits weighted by softmax probabilities)
- $B = \sigma^2$ (noise variance)

$$\mathbb{E}[\text{KL}] \approx \frac{1}{2}\left[ A\left(1-\frac{1}{T}\right)^2 + \frac{B}{T^2} \right]$$

---

## Step 10: Optimize over T

Let $u = 1/T$:

$$f(u) = A(1-u)^2 + Bu^2$$

$$\frac{df}{du} = -2A(1-u) + 2Bu = 0$$

$$-A + Au + Bu = 0$$

$$u(A + B) = A$$

$$u = \frac{A}{A+B}$$

Therefore:

$$\frac{1}{T^*} = \frac{A}{A+B}$$

$$T^* = \frac{A+B}{A} = 1 + \frac{B}{A}$$

---

## Step 11: Final Formula

$$\boxed{T^* = 1 + \frac{\sigma^2}{\text{Var}_p(z)}}$$

---

## Step 12: Approximation

If $\text{Var}_p(z) \approx \tau^2 = \text{Var}(z)$ (uniform vs softmax-weighted variance are similar):

$$T^* \approx 1 + \frac{\sigma^2}{\tau^2} = 1 + \alpha$$

where $\alpha = \sigma^2/\tau^2$ is the noise-to-signal ratio.

---

## Comparison with Variance Matching

**Variance matching** (wrong approach):

$$\text{Var}(\tilde{z}/T) = \text{Var}(z) \implies T^* = \sqrt{1+\alpha}$$

**KL minimization** (this derivation):

$$T^* = 1 + \alpha$$

For small $\alpha$:
- $\sqrt{1+\alpha} \approx 1 + \alpha/2$
- $1 + \alpha$ is approximately $2\times$ larger correction

This explains the empirical factor of ~2 observed in experiments.

---

## Note

The key difference is that:
- Variance matching works in **logit space** with uniform weights
- KL minimization works in **probability space** via Fisher information with softmax weights

The softmax nonlinearity causes the factor of 2 difference.
