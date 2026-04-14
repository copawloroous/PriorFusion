# III. METHODOLOGY

## A. Problem Overview

Image fusion aims to synthesize a fused image $\mathbf{I}_f \in \mathbb{R}^{H \times W \times 3}$ from a pair of co-registered source images $\mathbf{I}_A \in \mathbb{R}^{H \times W \times C_A}$ and $\mathbf{I}_B \in \mathbb{R}^{H \times W \times C_B}$, where the fused result aggregates complementary information from both modalities. The fusion process can be generally formulated as:

$$\mathbf{I}_f = \mathcal{F}_\theta(\mathbf{I}_A, \mathbf{I}_B), \tag{1}$$

where $\mathcal{F}_\theta$ denotes the fusion network parameterized by $\theta$.

Recent studies have demonstrated that integrating semantic priors from Vision Foundation Models (VFMs) can substantially improve fusion quality by providing high-level scene understanding. However, the prevailing paradigms for injecting VFM features into the fusion backbone suffer from a fundamental deficiency that we term the **semantic prior incompatibility problem**, which arises from two sources:

**(i) Channel-wise semantic misalignment.** Given a VFM encoder $\mathcal{E}_{\text{VFM}}$ and the fusion backbone encoder $\mathcal{E}_{\text{fuse}}$, their respective feature representations $\mathbf{F}_{\text{VFM}} \in \mathbb{R}^{C_v \times H_p \times W_p}$ and $\mathbf{F}_{\text{fuse}} \in \mathbb{R}^{C_f \times H_f \times W_f}$ reside in fundamentally different channel semantics. The physical meaning encoded in each channel of $\mathbf{F}_{\text{VFM}}$ bears no correspondence to any channel of $\mathbf{F}_{\text{fuse}}$, as the two encoders are trained with entirely different objectives.

**(ii) Conflicting optimization objectives.** VFMs are optimized for discrimination-friendly features that prioritize multi-view, texture-invariant, and environment-invariant semantic consistency, deliberately discarding pixel-level texture and detail information. Conversely, the fusion backbone is optimized to preserve pixel-level fidelity including textures, edges, and fine-grained details. These opposing optimization directions create an adversarial interference when features from both domains are naively combined.

Two prevalent injection paradigms exist in the literature, both of which fail to resolve this incompatibility:

**Paradigm I: Direct Feature Mixing.** Concatenation or element-wise operations between $\mathbf{F}_{\text{VFM}}$ and $\mathbf{F}_{\text{fuse}}$ along the channel dimension:

$$\mathbf{F}_{\text{mixed}} = \text{Conv}([\mathbf{F}_{\text{VFM}}; \mathbf{F}_{\text{fuse}}]), \tag{2}$$

which forces cross-channel interaction between semantically misaligned feature spaces, leading to feature corruption and adversarial interference.

**Paradigm II: SPADE-style Modulation.** Spatially-Adaptive Normalization encodes VFM features as affine transformation parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$:

$$\mathbf{F}_{\text{out}} = \boldsymbol{\gamma}(\mathbf{F}_{\text{VFM}}) \odot \text{IN}(\mathbf{F}_{\text{fuse}}) + \boldsymbol{\beta}(\mathbf{F}_{\text{VFM}}), \tag{3}$$

where $\text{IN}(\cdot)$ denotes Instance Normalization. While more principled than direct mixing, the modulation parameters $\boldsymbol{\gamma}, \boldsymbol{\beta}$ are derived from convolutions operating on $\mathbf{F}_{\text{VFM}}$, which inherently encodes channel semantics of the VFM space. The resulting scale and shift operations still operate under the assumption that VFM channel information can meaningfully modulate fusion features—an assumption that the channel misalignment problem directly violates.

**Our key insight** is that while VFM features and fusion backbone features are channel-incompatible, there exists a representation that is **invariant to the choice of feature space**: the **attention affinity matrix**. For any feature representation $\mathbf{F} \in \mathbb{R}^{N \times C}$ consisting of $N$ spatial tokens, the affinity matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ captures pairwise relational structure:

$$\mathbf{A}_{ij} = \text{sim}(\mathbf{f}_i, \mathbf{f}_j), \tag{4}$$

where $\mathbf{f}_i$ is the feature vector at spatial position $i$. Critically, affinity matrices encode the **relative relationships** and **contextual dependencies** among spatial tokens—an inductive bias that remains consistent across different feature spaces. Whether computed from VFM features or fusion backbone features, the affinity matrix reflects the same underlying semantic structure: which patches are semantically related and which are not. This property makes affinity matrices a natural "lingua franca" for cross-space prior injection, entirely circumventing the channel alignment problem.

Based on this insight, we propose PriorFusion, which transforms VFM semantic priors into an affinity matrix and injects it directly as an additive bias into the backbone's self-attention mechanism, achieving theoretically principled prior integration without any cross-channel interaction.


## B. VFM-based Semantic Prior Extraction

### 1) Patch Token Extraction

We employ a frozen Vision Foundation Model $\mathcal{V}$ (specifically, DINOv2 with registers) as the semantic prior source. Given a source image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, the VFM partitions it into non-overlapping patches of size $P \times P$ and produces patch-level tokens:

$$\mathbf{T} = \mathcal{V}(\phi(\mathbf{I})) \in \mathbb{R}^{N \times D}, \tag{5}$$

where $\phi(\cdot)$ denotes the standard ImageNet normalization, $N = H_p \times W_p$ with $H_p = \lfloor H/P \rfloor$, $W_p = \lfloor W/P \rfloor$, and $D$ is the token dimension. The CLS token and register tokens are discarded, retaining only spatial patch tokens. Notably, the VFM employs Rotary Position Embeddings (RoPE), which naturally generalizes to arbitrary input resolutions without requiring resizing to a fixed resolution (e.g., $224 \times 224$).

### 2) Intra-Modal De-Meaning

A critical preprocessing step is intra-modal de-meaning. In our preliminary experiments, we observed that the raw token representations are dominated by a modality-specific mean direction: for tokens $\{\mathbf{t}_i\}_{i=1}^{N}$ extracted from a single image, the pairwise cosine similarities $\cos(\mathbf{t}_i, \mathbf{t}_j)$ are uniformly high ($> 0.8$) regardless of semantic content. This phenomenon occurs because each modality induces a strong global bias in the feature space, masking the underlying semantic structure.

To expose the true semantic variations, we perform de-meaning:

$$\hat{\mathbf{T}} = \mathbf{T} - \frac{1}{N}\sum_{i=1}^{N} \mathbf{t}_i \cdot \mathbf{1}^{\top} = \mathbf{T} - \bar{\mathbf{T}}, \tag{6}$$

where $\bar{\mathbf{T}} \in \mathbb{R}^{1 \times D}$ is the spatial mean token. The de-meaned residuals $\hat{\mathbf{T}}$ capture the relative semantic deviations from the modality-specific baseline, which constitute the informationally relevant signal for semantic grouping.

### 3) Affinity Matrix Computation

From the de-meaned tokens, we compute the cosine affinity matrix that captures the pairwise semantic relatedness among all spatial patches:

$$\tilde{\mathbf{T}} = \text{L2Norm}(\hat{\mathbf{T}}), \quad \mathbf{A} = \tilde{\mathbf{T}} \tilde{\mathbf{T}}^{\top} \in \mathbb{R}^{N \times N}, \tag{7}$$

where $\text{L2Norm}(\cdot)$ normalizes each token to unit length, and $\mathbf{A}_{ij} \in [-1, 1]$ represents the cosine similarity between the semantic residuals of patches $i$ and $j$. High values indicate patches belonging to the same semantic region, while low or negative values indicate semantic dissimilarity.

### 4) Adaptive Cross-Modal Affinity Fusion

For a pair of source images $\mathbf{I}_A$ and $\mathbf{I}_B$, we compute their respective affinity matrices $\mathbf{A}_A$ and $\mathbf{A}_B$ independently using Eqs. (5)-(7). The two modalities typically exhibit asymmetric semantic expressiveness in the VFM feature space—empirically, we observe that the standard deviation of visible-spectrum affinity matrices is approximately $4\times$ larger than that of infrared, indicating richer semantic structure in the visible domain.

To account for this asymmetry, we introduce adaptive weighting based on affinity dispersion:

$$\sigma_A = \text{std}(\mathbf{A}_A), \quad \sigma_B = \text{std}(\mathbf{A}_B), \tag{8}$$

$$w_A = \frac{\sigma_A}{\sigma_A + \sigma_B + \epsilon}, \quad w_B = 1 - w_A, \tag{9}$$

$$\mathbf{A}_{\text{joint}} = w_A \cdot \mathbf{A}_A + w_B \cdot \mathbf{A}_B, \tag{10}$$

where $\sigma_{(\cdot)}$ computes the global standard deviation over all entries, $\epsilon$ is a small constant for numerical stability, and $\mathbf{A}_{\text{joint}} \in \mathbb{R}^{N \times N}$ is the fused affinity matrix. The modality with higher affinity dispersion—and hence richer semantic differentiation—receives a proportionally larger weight.

### 5) Spatial Saliency Derivation

From the joint affinity matrix, we derive a spatial saliency map that identifies semantically prominent regions. Each patch's saliency is defined as its average affinity with all other patches:

$$s_i = \frac{1}{N}\sum_{j=1}^{N} [\mathbf{A}_{\text{joint}}]_{ij}, \tag{11}$$

$$\mathbf{S} = \text{Reshape}\left(\frac{\mathbf{s} - \min(\mathbf{s})}{\max(\mathbf{s}) - \min(\mathbf{s}) + \epsilon}\right) \in \mathbb{R}^{1 \times H_p \times W_p}, \tag{12}$$

where $\mathbf{s} = [s_1, \ldots, s_N]^{\top}$ is normalized to $[0, 1]$ per sample. Patches with high average affinity correspond to regions that are semantically coherent with the global scene context, typically representing salient objects such as pedestrians or vehicles in infrared-visible fusion.


## C. Affinity-Guided Attention Injection

### 1) Theoretical Foundation

The core of our approach rests on the observation that the self-attention affinity in transformer blocks and the VFM-derived affinity matrix share the same **inductive bias**: both encode pairwise relational structure among spatial tokens.

In a standard spatial self-attention mechanism, the attention logits are computed as:

$$\mathbf{Q} = \mathbf{W}_Q \mathbf{X}, \quad \mathbf{K} = \mathbf{W}_K \mathbf{X}, \quad \mathbf{V} = \mathbf{W}_V \mathbf{X}, \tag{13}$$

$$\text{Attn}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}\right)\mathbf{V}, \tag{14}$$

where $\mathbf{X} \in \mathbb{R}^{N \times C}$ is the input feature, $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ are learnable projections, and $d$ is the head dimension. The pre-softmax logit matrix $\mathbf{L} = \mathbf{Q}\mathbf{K}^{\top}/\sqrt{d} \in \mathbb{R}^{N \times N}$ is itself an affinity matrix—it measures the compatibility between every pair of spatial positions.

Our key observation is that both $\mathbf{L}$ and $\mathbf{A}_{\text{joint}}$ are **$N \times N$ pairwise relational matrices over the same set of spatial positions**, encoding the same type of information: which patches should attend to which. The only difference is their source—$\mathbf{L}$ is derived from learnable projections of the backbone features, while $\mathbf{A}_{\text{joint}}$ is derived from frozen VFM features. Crucially, neither involves channel-level semantics in the injection operation: the additive combination occurs entirely in the **spatial relational space** $\mathbb{R}^{N \times N}$, which is invariant to the specific channel semantics of either encoder.

### 2) Additive Affinity Bias

Based on this analysis, we inject the VFM semantic prior as an additive bias to the attention logits:

$$\text{Attn}_{\text{prior}}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}} + \alpha \cdot \mathbf{A}_{\text{bias}}\right)\mathbf{V}, \tag{15}$$

where $\mathbf{A}_{\text{bias}} \in \mathbb{R}^{N \times N}$ is derived from the joint affinity matrix $\mathbf{A}_{\text{joint}}$ (with spatial interpolation if necessary), and $\alpha$ is a learnable scalar controlling the injection strength. The parameter $\alpha$ is constrained to be non-negative via the softplus function:

$$\alpha = \text{softplus}(\alpha_{\text{raw}}) = \log(1 + e^{\alpha_{\text{raw}}}), \tag{16}$$

with an upper bound clamp $\alpha \leq \alpha_{\max}$ to prevent the prior from overwhelming the backbone's intrinsic attention patterns. We set $\alpha_{\max} = 2.0$, since a logit shift of $2.0$ already amplifies the relative attention weight by a factor of $e^2 \approx 7.4$, beyond which marginal returns diminish and the network's local attention capacity degrades.

**Initialization.** The raw parameter is initialized as $\alpha_{\text{raw}} = -0.69$, yielding $\text{softplus}(-0.69) \approx 0.5$. This ensures that at training onset, the prior injection provides a moderate semantic guidance without disrupting the pretrained backbone dynamics.

**Spatial alignment.** When the VFM patch grid $(H_p \times W_p)$ differs from the backbone's bottleneck resolution $(H_b \times W_b)$, the affinity matrix is resized via bilinear interpolation:

$$\mathbf{A}_{\text{bias}} = \text{Interp}_{N_b \times N_b}\left(\mathbf{A}_{\text{joint}}\right), \tag{17}$$

where $N_b = H_b \times W_b$. In our default setting with $128 \times 128$ inputs and patch size $P = 16$, the VFM produces an $8 \times 8$ patch grid ($N = 64$), which exactly matches the bottleneck resolution of our 4-level encoder, making interpolation unnecessary.


## D. Saliency-Weighted Skip Connection

At the intermediate decoder level (Level 3), we apply a lightweight saliency-guided modulation to the skip connection features. Unlike the bottleneck affinity injection, which operates in the relational space, this module operates as a simple multiplicative enhancement in the spatial domain:

$$\hat{\mathbf{F}}_{\text{skip}} = \mathbf{F}_{\text{skip}} \odot \left(1 + \beta \cdot \text{Up}(\mathbf{S})\right), \tag{18}$$

where $\mathbf{F}_{\text{skip}} \in \mathbb{R}^{C \times H_3 \times W_3}$ is the fused skip feature at Level 3, $\text{Up}(\mathbf{S})$ upsamples the saliency map to match the spatial resolution $(H_3, W_3)$, and $\beta = \text{softplus}(\beta_{\text{raw}}) \geq 0$ is a learnable gating parameter. The formulation ensures that: (1) when $\beta \to 0$, the module degenerates to an identity mapping, preserving the baseline behavior; (2) the enhancement is strictly additive ($1 + \beta \cdot \mathbf{S} \geq 1$ since $\mathbf{S} \in [0, 1]$ and $\beta \geq 0$), meaning salient regions are amplified while non-salient regions remain unchanged rather than being suppressed.


## E. Network Architecture

The overall architecture of PriorFusion adopts a dual-encoder, single-decoder U-Net structure with four hierarchical levels. Let $\mathcal{E}_A$ and $\mathcal{E}_B$ denote the encoders for modalities A and B, respectively, each producing multi-scale features $\{\mathbf{F}_A^l\}_{l=1}^{4}$ and $\{\mathbf{F}_B^l\}_{l=1}^{4}$.

**Encoding.** Each encoder consists of cascaded Transformer blocks with progressive channel expansion (dim $= [C, 2C, 4C, 8C]$) and spatial downsampling via pixel-unshuffle. Channel-spatial attention modules (CBAM) are applied at Levels 2-4 to refine modality-specific features before fusion.

**Bottleneck fusion.** At Level 4 (lowest resolution), cross-attention first enables inter-modal feature exchange:

$$\hat{\mathbf{F}}_A^4, \hat{\mathbf{F}}_B^4 = \text{CrossAttn}(\mathbf{F}_A^4, \mathbf{F}_B^4), \tag{19}$$

followed by channel concatenation and $1 \times 1$ projection for initial fusion:

$$\mathbf{F}^4 = \text{Conv}_{1\times1}([\hat{\mathbf{F}}_A^4; \hat{\mathbf{F}}_B^4]). \tag{20}$$

The affinity-guided spatial self-attention (Eq. 15) is then applied to $\mathbf{F}^4$, constituting the primary prior injection point.

**Skip fusion.** At Levels 1-3, features from both encoders are fused via channel projection:

$$\mathbf{F}^l = \text{Conv}_{1\times1}([\mathbf{F}_A^l; \mathbf{F}_B^l]), \quad l = 1, 2, 3. \tag{21}$$

At Level 3, the saliency-weighted modulation (Eq. 18) is additionally applied.

**Decoding.** The decoder progressively upsamples via pixel-shuffle and integrates skip connections:

$$\mathbf{D}^l = \mathcal{T}^l(\text{Conv}([\text{Up}(\mathbf{D}^{l+1}); \mathbf{F}^l])), \quad l = 3, 2, 1, \tag{22}$$

where $\mathcal{T}^l$ denotes the Transformer decoder blocks at level $l$, and $\text{Up}(\cdot)$ is pixel-shuffle upsampling. The final fused image is produced by a $3 \times 3$ convolution mapping the Level 1 features to 3-channel RGB output.


## F. Loss Function

The total training loss consists of four complementary terms:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{int}} + \lambda_2 \mathcal{L}_{\text{ssim}} + \lambda_3 \mathcal{L}_{\text{grad}} + \lambda_4 \mathcal{L}_{\text{color}}, \tag{23}$$

**Intensity preservation loss.** Encourages the fused image to retain the maximal intensity from both sources:

$$\mathcal{L}_{\text{int}} = \frac{1}{HW}\|\mathbf{I}_f - \max(\mathbf{I}_A, \mathbf{I}_B)\|_1, \tag{24}$$

where the $\max$ operation selects the brighter pixel element-wise, guided by a grayscale comparison mask.

**Structural similarity loss.** Preserves structural correspondence with both sources:

$$\mathcal{L}_{\text{ssim}} = (1 - \text{SSIM}(\mathbf{I}_A, \mathbf{I}_f)) + (1 - \text{SSIM}(\mathbf{I}_B^{\text{gray}}, \mathbf{I}_f^{\text{gray}})), \tag{25}$$

where $\text{SSIM}(\cdot, \cdot)$ is computed with a window size of 48.

**Gradient preservation loss.** Retains the maximal edge information:

$$\mathcal{L}_{\text{grad}} = \frac{1}{HW}\left(\|\nabla_x \mathbf{I}_f - \max(\nabla_x \mathbf{I}_A, \nabla_x \mathbf{I}_B)\|_1 + \|\nabla_y \mathbf{I}_f - \max(\nabla_y \mathbf{I}_A, \nabla_y \mathbf{I}_B)\|_1\right), \tag{26}$$

where $\nabla_x$ and $\nabla_y$ denote horizontal and vertical Sobel gradient operators.

**Color consistency loss.** Ensures chrominance fidelity with the visible source:

$$\mathcal{L}_{\text{color}} = \frac{1}{HW}\left(\|C_b(\mathbf{I}_f) - C_b(\mathbf{I}_A)\|_1 + \|C_r(\mathbf{I}_f) - C_r(\mathbf{I}_A)\|_1\right), \tag{27}$$

where $C_b(\cdot)$ and $C_r(\cdot)$ extract the chrominance channels in YCbCr space. By constraining only the Cb and Cr components, the luminance channel remains unconstrained, allowing the infrared modality's intensity information to contribute freely to the fused result.

The loss weights are set as $\lambda_1 = 8$, $\lambda_2 = 1$, $\lambda_3 = 10$, $\lambda_4 = 12$ throughout all experiments.


## G. Discussion: Why Affinity Injection Resolves the Incompatibility

We conclude this section with a theoretical discussion on why the proposed affinity-based injection fundamentally differs from and improves upon existing paradigms.

**Proposition 1 (Space Invariance of Affinity Structure).** Let $\mathbf{F}_1 \in \mathbb{R}^{N \times C_1}$ and $\mathbf{F}_2 \in \mathbb{R}^{N \times C_2}$ be two feature representations of the same set of $N$ spatial positions, produced by different encoders. If both encoders capture the same underlying semantic grouping structure, then their respective affinity matrices $\mathbf{A}_1 = \tilde{\mathbf{F}}_1 \tilde{\mathbf{F}}_1^{\top}$ and $\mathbf{A}_2 = \tilde{\mathbf{F}}_2 \tilde{\mathbf{F}}_2^{\top}$ are structurally correlated, regardless of $C_1 \neq C_2$ or the specific channel semantics.

This proposition is supported by our empirical observation that the affinity matrices computed from DINOv2 features and from the trained fusion backbone features exhibit high structural correlation (measured via Centered Kernel Alignment), despite operating in entirely different channel spaces.

**Consequence.** The additive injection $\mathbf{L} + \alpha \cdot \mathbf{A}_{\text{joint}}$ operates entirely within $\mathbb{R}^{N \times N}$—the space of pairwise spatial relations. No cross-channel interaction is performed, no channel alignment is assumed, and no channel-space translation is needed. The VFM prior simply reinforces the backbone's spatial attention by telling it "these patches are semantically related," using a language (pairwise similarity scores) that both spaces inherently understand. This is why PriorFusion succeeds where concatenation and SPADE-style modulation fail.
