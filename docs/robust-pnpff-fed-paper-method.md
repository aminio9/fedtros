# Paper-Ready Method Naming and Description

## Recommended method name

**Fed-RPNP: Federated Robust Positive-Negative Prototype Fusion for Open-Set Intrusion Detection**

Recommended acronym: **Fed-RPNP**

This name is preferable for a journal manuscript because it is concise, distinguishes the proposed
federated adaptation from the original PNPFF method, and does not imply that the method is an exact
reproduction of APNPFF or APNPFF++. The word *robust* refers specifically to representation
normalization, class-balanced optimization, known-derived boundary regularization, calibrated
rejection, and detector health validation.

Alternative title suitable for a broader Q1 journal submission:

> **Privacy-Preserving Open-Set Intrusion Detection via Federated Knowledge Distillation and Robust
> Positive-Negative Prototype Fusion**

The complete system may be described as **DKD-FedOS with a Fed-RPNP global detector**. DKD-FedOS
denotes the federated student-learning framework, whereas Fed-RPNP denotes the post-aggregation
open-set decision module. Keeping these names separate makes the contribution boundaries clear.

## Paper positioning

Fed-RPNP is a privacy-preserving adaptation of the Positive-Negative Prototypes Fusion Framework
(PNPFF) proposed by Zhong and Cui [1]. The original PNPFF jointly learns a deep embedding, multiple
positive prototypes, and one negative prototype per known class. Directly attaching prototypes to a
frozen federated classifier is not equivalent to that formulation. In particular, a closed-set
student may map an unseen attack into a highly confident known-class region, after which a post-hoc
prototype layer cannot recover an appropriate open-set geometry.

Fed-RPNP addresses this federated deployment gap without transferring client samples. After the
federated rounds are complete, the server constructs a private copy of the aggregated student
backbone and adapts that copy jointly with the PNPFF projection and prototypes. Adaptation uses only
a shared known-class fitting partition and privacy-safe pseudo-unknown samples derived from that
partition. Real unknown test samples are excluded from training, model selection, score orientation,
and threshold calibration.

The method should be presented as an **equation-faithful PNPFF core with an explicitly identified
robust federated extension**, not as an architecture-identical reproduction of the original image
classification experiments.

## System model and federated location

Assume (C) federated clients and (K) known traffic classes. Client (c) owns a private dataset
(mathcal{D}_c). During federated learning, every client trains its local teacher and the shared
student according to DKD-FedOS. Only the permitted student parameters and privacy-sanitized scalar
statistics are communicated to the server. Raw traffic records, labels, class histograms, local
prototypes, and pseudo-unknown samples remain local.

After the final aggregation round, the server obtains the global student
(f_{\bar{\theta}}). Fed-RPNP is then fitted once using a shared known-only dataset
(mathcal{D}_{g}). A deterministic stratified split produces:


- (mathcal{D}_{fit}), containing 70% of each known class, for representation and prototype fitting;
- (mathcal{D}_{cal}), containing the remaining 30%, for checkpoint selection, score calibration,
  detector validation, and threshold selection.


Unknown-class samples are forbidden in both partitions. Fed-RPNP is not trained or executed
independently by Flower clients. Open-set rejection is performed globally after aggregation.

## Adapted embedding and prototype representation

Let (h_{\bar{\theta}}(x)) be the feature vector produced by a private copy of the aggregated
student backbone. The fitting-partition mean (\mu_h) and standard deviation (\sigma_h) are stored
and used to standardize the representation. The adapted embedding is


\[
z(x)=\operatorname{norm}_2\!\left(
W\frac{h_{\theta_r}(x)-\mu_h}{\sigma_h+\epsilon}
\right),
\]


where (W) is an identity-initialized projection, (\theta_r) is initialized from the final
aggregated backbone, and (\operatorname{norm}_2) denotes unit (L_2) normalization. The normalized
embedding prevents prototype norms and dot-product logits from growing without bound.

For each known class (k\in\{1,\ldots,K\}), the detector learns (V=7) positive prototypes
({P_{k,v}}_{v=1}^{V}) and one negative prototype (N_k). Prototypes are projected back to the unit
hypersphere after every optimization step.

Following PNPFF, the combined Euclidean and directional distance is


\[
d(z,p)=\frac{1}{m}\lVert z-p\rVert_2^2-z^\top p,
\]


where (m=128) is the embedding dimension. The distance to the positive prototype set is


\[
d(z,P_k)=\min_{v\in\{1,\ldots,V\}}d(z,P_{k,v}).
\]


## Positive and negative prototype learning

The positive class probability is


\[
p_P(y=k\mid x)=
\frac{\exp[-\gamma_1d(z(x),P_k)]}
{\sum_{j=1}^{K}\exp[-\gamma_1d(z(x),P_j)]},
\]


and its classification loss is


\[
\mathcal{L}_P=-\log p_P(y\mid x).
\]


The moving-radius loss encourages compact known-class support:


\[
\mathcal{L}_O=
\max\left(0,
\frac{1}{m}\lVert z(x)-\bar{P}_y\rVert_2^2-R
\right),
\]


where (\bar{P}_y) is the mean positive prototype of class (y). The radius is parameterized as
(R=\operatorname{softplus}(r)), ensuring that it remains positive while retaining a stable gradient
near zero.

Prototype diversity contains local separation and balanced assignment terms. For a sample from
class (y), its assignment probability is


\[
a_v(x)=
\frac{\exp[-T d(z(x),P_{y,v})]}
{\sum_{u=1}^{V}\exp[-T d(z(x),P_{y,u})]}.
\]


The negative sign is deliberate. Equation 8 of the source paper prints a positive sign, while its
distance definition makes smaller values more similar and its accompanying text states that
assignment should favor the most similar prototype. Using (-Td) is therefore the mathematically
consistent interpretation. The batch-average assignment is regularized toward the uniform prior so
that all prototype slots remain active.

The negative class probability is


\[
p_N(y=k\mid x)=
\frac{\exp[\gamma_2d(z(x),N_k)]}
{\sum_{j=1}^{K}\exp[\gamma_2d(z(x),N_j)]},
\]


with loss


\[
\mathcal{L}_N=-\log p_N(y\mid x).
\]


Positive and negative learning are alternated for every mini-batch. The positive step updates the
adapted backbone, projection, positive prototypes, and radius. The negative step updates the adapted
backbone, projection, and negative prototypes. Separate SGD optimizers, momentum, gradient clipping,
and prototype retraction are used. Class-balanced sampling prevents the majority Normal class from
dominating either update.

## Privacy-safe open-space regularization

Base PNPFF learns exclusively from known samples and may remain vulnerable when an unseen attack is
embedded inside a known-class region. Fed-RPNP therefore introduces an APNPFF++-inspired, but much
simpler, known-derived boundary regularizer.

Two fitting examples from different known classes are selected and mixed:


\[
\tilde{x}=\lambda x_i+(1-\lambda)x_j+\varepsilon,
\qquad y_i\neq y_j,
\qquad \lambda\sim\operatorname{Beta}(\alpha,\alpha).
\]


Random feature masking and small Gaussian noise are optionally applied. The resulting
(\tilde{x}) is treated as a pseudo-unknown boundary sample, not as a real unknown class. For
(U_K), the uniform distribution over the known classes, the open-space loss is


\[
\mathcal{L}_U=
D_{KL}(p_P(\cdot\mid\tilde{x})\parallel U_K)
+D_{KL}(p_N(\cdot\mid\tilde{x})\parallel U_K).
\]


The complete robust objective is


\[
\mathcal{L}=
\mathcal{L}_P
+\lambda_0\mathcal{L}_N
+\lambda_1\mathcal{L}_O
+\lambda_2\mathcal{L}_{div}
+\lambda_u\mathcal{L}_U.
\]


Default values are (\gamma_1=\gamma_2=1), (\lambda_0=\lambda_1=0.1),
(\lambda_2=0.01), (\lambda_u=0.1), (T=10), and (V=7). The stable post-federation learning
rate is 0.01. The original paper learning rate of 0.1 remains an explicit reproduction setting.

## Fused class score and calibrated rejection

The paper-style fused known-class score is


\[
S_k(x)=\eta p_P(y=k\mid x)+\omega p_N(y=k\mid x),
\]


with (\eta=\omega=0.5). The candidate known-class prediction is


\[
\hat{y}_{PNP}(x)=\arg\max_k S_k(x),
\]


and the raw paper-derived unknown score is


\[
u_{raw}(x)=1-\max_k S_k(x).
\]


Softmax confidence alone can be severely miscalibrated for out-of-distribution traffic. Fed-RPNP
therefore fits a non-decreasing isotonic function (g) using only held-out known scores and
independently generated held-out pseudo-unknown scores:


\[
u(x)=g(u_{raw}(x)).
\]


The operational score (u(x)\in[0,1]) is always defined so that larger values indicate stronger
unknown evidence. Test labels are never used to choose or reverse score direction.

The rejection threshold is selected to maximize pseudo-unknown F1 subject to


\[
\frac{1}{|\mathcal{D}_{cal}|}
\sum_{x\in\mathcal{D}_{cal}}\mathbb{1}[u(x)>\tau]
\leq \rho,
\]


where (\rho=0.05) is the target known false-unknown rate. If no useful feasible operating point is
available, the threshold falls back to the corresponding known-score quantile. A sample is rejected
when (u(x)>\tau); otherwise, `prototype_rank` returns (\hat{y}_{PNP}(x)).

## Model selection and failure detection

Ordinary validation accuracy is inappropriate for the strongly imbalanced traffic distribution.
Fed-RPNP records known balanced accuracy (B_e) and pseudo-unknown AUROC (A_e) at every epoch. A
checkpoint is eligible only if its balanced accuracy is within two percentage points of the best
known-only checkpoint. Among eligible checkpoints, the selected objective is


\[
H_e=\frac{2B_eA_e}{B_e+A_e}.
\]


Before open-test evaluation, the detector must satisfy all of the following calibration-only checks:


1. scores, backbone parameters, projections, radii, and prototypes are finite;
2. pseudo-unknown AUROC is at least 0.55;
3. the pseudo-unknown score median exceeds the known score median;
4. the measured known false-unknown rate is within the configured tolerance.


When Fed-RPNP is the selected detector, a failed check terminates evaluation with an explicit reason.
For mean, weighted, or maximum score fusion, an unhealthy prototype component is excluded instead of
silently injecting an inverted score.

## Algorithm summary

**Input:** final aggregated student, shared known-only data, (K) known classes.  
**Output:** adapted global PNPFF detector and calibrated rejection threshold.


1. Deterministically divide the shared known data into stratified fitting and calibration partitions.
2. Copy the final student backbone; compute and store fitting-feature normalization statistics.
3. Initialize the projection, seven positive prototypes per class, one negative prototype per class,
   and a positive moving radius.
4. Generate deterministic cross-class pseudo-unknown fitting samples.
5. For every class-balanced mini-batch, alternate positive and negative optimization, add the
   corresponding pseudo-unknown uniformity term, clip gradients, and normalize prototypes.
6. Select the checkpoint using constrained balanced accuracy and pseudo-unknown AUROC.
7. Generate a separate pseudo-unknown calibration set and fit the monotonic unknown-score calibrator.
8. Select (\tau) under the known false-unknown constraint and run detector health checks.
9. Evaluate the mixed known/unknown test set exactly once using the final aggregated checkpoint.

## Claimed contributions and novelty boundaries

The manuscript may claim the following contributions:


- a privacy-preserving integration of positive-negative prototype fusion with a federated
  knowledge-distillation intrusion detector;
- post-aggregation joint representation-prototype adaptation that avoids the failure of fitting
  prototypes over a frozen closed-set geometry;
- normalized and class-balanced alternating prototype optimization for imbalanced network traffic;
- known-derived open-space regularization and held-out monotonic score calibration without real
  unknown samples;
- calibration-only health validation that detects score inversion before open-test evaluation;
- a final-only global fitting protocol that avoids redundant client rejection and repeated
  per-round detector training.


The manuscript should not claim that Fed-RPNP reproduces the original VGG32 architecture, implements
the APNPFF GAN, or trains from real FoT samples. It should also distinguish the implemented
known-derived mixup regularizer from the complete APNPFF++ method.

## Recommended experimental reporting

Report closed-set and open-set behavior together. At minimum, include known accuracy, known balanced
accuracy, macro-F1, unknown precision/recall/F1, AUROC, AUPRC, FPR95, known false-unknown rate, and
OSCR. Report the raw and calibrated Fed-RPNP scores separately so that score calibration cannot be
mistaken for representation improvement.

Recommended ablations are:


1. frozen versus adapted aggregated backbone;
2. unnormalized versus normalized embeddings and prototypes;
3. imbalanced versus class-balanced batches;
4. printed (+Td) versus similarity-consistent (-Td) assignment;
5. without versus with pseudo-unknown regularization;
6. fixed (\tau=0.5), known quantile, and constrained pseudo-unknown calibration;
7. PNPFF alone versus generator, energy, PROSER, and combined fusion;
8. per-round refitting cost versus final-only global fitting.


For every experiment, publish the run seed, known/unknown label split, number of clients, federated
rounds, local episodes, IID/non-IID setting, PNPFF checkpoint epoch, threshold metadata, detector
health report, and per-component AUROC. Historical results from the former post-hoc detector must be
reported as a separate baseline because the meaning of `prototype_rank` has changed.

## Reproducibility statement

Fed-RPNP fitting is deterministic for a fixed software environment, checkpoint, dataset, and seed.
The fitting/calibration split, balanced sampler, prototype initialization, manifold-mixup pairs,
masking, noise, and calibration pseudo-unknowns are all seeded. The saved `pnpff_state.pt` artifact
contains the adapted backbone, projection, prototypes, feature statistics, radius, isotonic
calibrator, threshold, best epoch, configuration, training history, and detector health metadata.

## Reference

[1] X. Zhong and J. Cui, “Positive-negative prototypes fusion framework for open set recognition,”
*Scientific Reports*, vol. 15, art. 23815, 2025. DOI: 10.1038/s41598-025-09625-4.
