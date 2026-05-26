# Proposed Method

This draft is written in manuscript style for a high-quality journal paper. It is aligned with the implemented framework in `cf_marlos`, whose full name is **Cooperative Federated Multi-Agent Reinforcement Learning with Learnable Aggregation for Open-Set Blockchain Intrusion Detection**. The method combines local CVAE-DQN learning, federated aggregation, generator-based reconstruction, and EVT-based open-set calibration for B-NAT blockchain traffic.

## 4. Proposed Method

We consider a horizontal federated learning setting with `N` clients. Client `i` owns a local dataset `D_i`, while the server has access only to aggregated model updates and a validation set `D_val`. The learning objective is to preserve known-class classification quality, remain stable under non-IID client partitions, and reject previously unseen attacks rather than forcing them into known classes.

Each client trains a CVAE-DQN agent locally. The shared trainable parameters are aggregated by FedAvg, FedProx, or FMRL-LA. After federated training, the reconstruction behavior of the global model is calibrated with EVT on validation data to produce a thresholded unknown-rejection rule.

### Notation

- `theta = {theta_p, theta_r, theta_Q, theta_G}`: federated parameters of the prior network, recognition network, main Q-network, and generator.
- `theta_Q^-`: local target Q-network parameters.
- `K`: number of federated rounds.
- `mu`: FedProx proximal coefficient.
- `M_evt`: class-conditional EVT models.
- `delta`: global open-set rejection threshold.

### 4.1 System Overview

The complete pipeline starts with raw B-NAT traffic and ends with calibrated open-set inference. Raw records are preprocessed into tensor datasets, then partitioned into client shards using a non-IID split. Each client performs local CVAE-DQN training on its private shard, and the server aggregates the resulting updates into a global model. Once the shared representation has converged, EVT is fitted on validation reconstruction errors to transform the reconstruction score into a calibrated unknown probability.

![Overview of the proposed method](D:/Research/cf_marlos/docs/figures/proposed_method_overview.png)

*Figure 4.1. Overview of the cooperative federated open-set intrusion detection pipeline.*

**Algorithm 1 belongs here** because it captures the full end-to-end workflow.

```text
Algorithm 1: Federated CVAE-DQN Training and Open-Set Calibration

Input:
    Client datasets {D_i}_{i=1}^N
    Validation set D_val
    Federated rounds K
    Local training configuration H
    Aggregation strategy S in {FedAvg, FedProx, FMRL-LA}

Output:
    Final global model theta*
    Class-conditional EVT models M_evt
    Global rejection threshold delta

1:  Initialize global parameters theta^0 = {theta_p^0, theta_r^0, theta_Q^0, theta_G^0}
2:  for round k = 1, ..., K do
3:      Server samples available clients C_k
4:      Server broadcasts theta^{k-1} to each client in C_k
5:      for each client i in C_k in parallel do
6:          Set mu_i = mu if S = FedProx, otherwise mu_i = 0
7:          theta_i^k, m_i^k <- ClientLocalUpdate(D_i, theta^{k-1}, H, mu_i)
8:      end for
9:      if S = FMRL-LA then
10:         theta^k <- FMRLLAUpdate(theta^{k-1}, {theta_i^k, m_i^k}_{i in C_k})
11:     else
12:         theta^k <- sum_{i in C_k} n_i theta_i^k / sum_{i in C_k} n_i
13:     end if
14: end for
15: theta* <- theta^K
16: M_evt, delta <- EVTCalibration(D_val, theta*)
17: return theta*, M_evt, delta
```

This algorithm is the correct top-level statement of the method because it makes the dependency chain explicit: local training precedes aggregation, and aggregation precedes open-set calibration. It also shows that FedAvg and FedProx share the same aggregation skeleton, while FMRL-LA replaces uniform averaging with utility-aware aggregation.

### 4.2 Local CVAE-DQN Agent

Each client is modeled as a local agent that interacts with its private shard of traffic data. Given a traffic feature vector `s_t`, the agent selects a class action `a_t` using an epsilon-greedy policy and receives a reward that is positive when the predicted class matches the true class and negative otherwise. This formulation allows the classification task to be trained as a reinforcement learning problem while still preserving the structure of supervised class prediction.

The client architecture contains four learned modules. The prior network `p_theta(z|s)` maps state features into a latent distribution, the recognition network `q_phi(z|s,a)` conditions the latent code on the state-action pair, the main Q-network `Q_psi(z,s)` predicts known-class values, and the target Q-network `Q_psi^-` stabilizes temporal-difference learning. In addition, the client maintains a replay buffer for experience replay and an epsilon-greedy policy for exploration.

The local optimization objective is a sum of three interacting components. First, the prior network is updated by minimizing the KL divergence between the posterior and prior latent distributions. Second, the recognition network and main Q-network are updated using a Double-DQN temporal-difference loss. Third, when FedProx is active, a proximal penalty is added to constrain local drift from the current global parameters. In compact form:

```text
L_prior = KL(q_phi(z|s,a*) || p_theta(z|s))
L_TD    = Huber(Q_psi(z,s,a) - y)
y       = r + gamma (1 - d) Q_psi^-(z', s', argmax_a Q_psi(z', s', a))
L_prox  = (mu / 2) ||theta - theta^{k-1}||_2^2
```

The target Q-network is not federated as a separate global object. It is a local stabilization copy that is synchronized from the main Q-network after aggregation and refreshed periodically during local training.

**Algorithm 2 belongs here** because it describes the exact local update procedure.

```text
Algorithm 2: Client-Side CVAE-DQN Local Update with Optional FedProx

Input:
    Local client dataset D_i
    Broadcast global parameters theta^{k-1}
    Local training configuration H
    Proximal coefficient mu_i

Output:
    Updated client parameters theta_i^k
    Client diagnostics m_i^k

1:  Load theta^{k-1} into p_theta, q_phi, Q_psi, and G_omega
2:  Set local target network Q_psi^- <- Q_psi
3:  Store theta^{k-1} as the proximal reference
4:  Initialize replay buffer B_i and epsilon-greedy policy pi_i
5:  for each local episode e = 1, ..., E do
6:      Reset the local environment and observe s_t
7:      for each step t = 1, ..., T do
8:          Select action a_t using pi_i
9:          Observe r_t, s_{t+1}, done flag d_t, and true class a_t*
10:         Store (s_t, a_t, r_t, s_{t+1}, d_t, a_t*) in B_i
11:         if |B_i| is sufficient then
12:             Sample mini-batch b from B_i
13:             L_p <- KL(q_phi(z|s,a*) || p_theta(z|s))
14:             if mu_i > 0 then
15:                 L_p <- L_p + (mu_i / 2)||theta_p - theta_p^{k-1}||_2^2
16:             end if
17:             Update theta_p using L_p
18:             y <- r + gamma(1-d) Q_psi^-(z', s', argmax_a Q_psi(z', s', a))
19:             L_Q <- Huber(Q_psi(z,s,a) - y)
20:             if mu_i > 0 then
21:                 L_Q <- L_Q + (mu_i / 2)(
                          ||theta_r - theta_r^{k-1}||_2^2
                        + ||theta_Q - theta_Q^{k-1}||_2^2)
22:             end if
23:             Update theta_r and theta_Q using L_Q
24:             Periodically update Q_psi^- from Q_psi
25:         end if
26:     end for
27: end for
28: Build C_i = correctly classified known samples from D_i
29: if generator training is enabled and |C_i| is sufficient then
30:     Train G_omega on C_i using reconstruction loss MSE(G_omega(z,a), s)
31:     Add FedProx generator penalty if mu_i > 0
32: end if
33: Return theta_i^k = {theta_p, theta_r, theta_Q, theta_G} and diagnostics m_i^k
```

The value of this algorithm is that it makes the hybrid nature of the client update explicit. It is not a standard classifier update, nor is it a purely RL update. It combines latent-variable modeling, Double-DQN learning, replay-based optimization, and optional proximal regularization in a single local procedure. The generator is trained only after the client identifies correctly classified known samples, which protects the reconstruction model from contamination by ambiguous or misclassified traffic.

### 4.3 Generator-Based Reconstruction

The generator `G_omega(z,a)` reconstructs the input feature vector from a latent code and a class condition. In this framework, the generator is not used as a standalone decoder for data synthesis; instead, it serves as the bridge between the latent representation and the open-set detector. During training, the generator is updated only on correctly classified known samples. This design choice is important because the reconstruction error must represent the geometry of known classes, not the noise introduced by incorrect labels or unknown traffic.

For a sample `x`, reconstruction error is computed as:

```text
e(x) = MSE(x, G_omega(q_phi(x, a_hat), a_hat))
```

where `a_hat` is the predicted known class. Known samples should yield low reconstruction error, while unknown samples should typically produce larger error because they do not lie on the learned known-class manifold.

No separate algorithm is required here because generator training is already embedded in Algorithm 2. A separate algorithm would duplicate the client-side update.

### 4.4 Federated Parameter Sharing

The federated contract is deliberately narrow. Only trainable parameters that define the shared representation are exchanged across clients. Local state, calibration artifacts, and privacy-sensitive buffers remain private to each client.

| Federated component | Role |
|---|---|
| Prior network `p_theta(z|s)` | Learns shared latent structure from traffic features |
| Recognition network `q_phi(z|s,a)` | Supports latent inference and reconstruction |
| Main Q-network `Q_psi(z,s)` | Learns the global known-class decision policy |
| Generator `G_omega(z,a)` | Learns shared reconstruction behavior for open-set detection |

| Not federated | Role |
|---|---|
| Target Q-network `Q_psi^-` | Local stabilization copy of the main Q-network |
| Replay buffer | Stores local transitions and interaction history |
| Optimizer states | Local training state, not required for aggregation |
| EVT models and thresholds | Calibration artifacts fitted after training |
| Raw local data | Remains on the client for privacy and federation assumptions |

In the current implementation, the target Q-network is synchronized locally after aggregation and should be described as a local stabilizer, not as a separately shared model. Likewise, EVT is a calibration layer rather than a federated representation learner, so it should remain outside the aggregation loop.

### 4.5 FedAvg and FedProx Baselines

FedAvg serves as the canonical federated baseline. After local training, the server aggregates client parameters by sample count:

```text
theta^k = sum_i n_i theta_i^k / sum_i n_i
```

FedProx retains the same server-side averaging rule but modifies the local objective by penalizing drift from the broadcast global model:

```text
F_i^prox(theta) = F_i(theta) + (mu / 2) ||theta - theta^{k-1}||_2^2
```

In this repository, the proximal term is applied on the client side to the federated blocks learned during local training. This makes FedProx more stable under non-IID client data, but also more conservative when the proximal coefficient is too large. For that reason, FedAvg and FedProx should be presented as optimization baselines rather than as separate algorithm blocks.

### 4.6 FMRL-LA Cooperative Aggregation

FMRL-LA is the main federated contribution of the method. It replaces uniform aggregation with a two-phase cooperative protocol that estimates client utility before deciding which updates should influence the global model. The server does not treat every client as equally informative; instead, it scores each client using latent summaries and training diagnostics that reflect reward, classification quality, stability, novelty, and data coverage.

The client-side audit vector combines the mean latent summary from the prior network with scalar diagnostics derived from local training. These diagnostics include recent reward, historical reward, macro F1, accuracy, TD stability, novelty, class entropy, label coverage, generator quality, and interaction count. The server uses these signals to estimate client utility through a learned critic and then selects the clients whose updates are most likely to improve the global objective.

The aggregation step is utility-weighted rather than purely sample-weighted:

```text
theta^k = theta^{k-1} + eta * sum_{i in A_k} u_i (theta_i^k - theta^{k-1}) / sum_{i in A_k} u_i
```

where `A_k` is the selected client set and `u_i` is the estimated utility of client `i`. During early rounds, the utility model is warmed up so that selection does not become overly aggressive before the server has enough evidence about client quality.

**Algorithm 3 belongs here** because it is the novel cooperative federated mechanism.

```text
Algorithm 3: FMRL-LA Utility-Aware Client Selection and Aggregation

Input:
    Previous global parameters theta^{k-1}
    Client updates and diagnostics {theta_i^k, m_i^k}_{i in C_k}
    Minimum selected clients q_min
    Maximum selected fraction rho_max
    Utility threshold lambda
    Aggregation step size eta

Output:
    Updated global parameters theta^k

Phase A: utility estimation and client selection
1:  for each client i in C_k do
2:      Extract latent summary h_i
3:      Extract scalar diagnostics x_i from m_i^k:
            reward, historical reward, F1, accuracy, TD stability,
            novelty, class entropy, label coverage, generator quality,
            and local step count
4:      Estimate utility u_i = C_i(h_i, x_i)
5:      Clip or temperature-scale u_i
6:  end for
7:  Select A_k = {i : u_i >= lambda}
8:  if |A_k| < q_min then
9:      Add highest-utility clients until |A_k| = q_min
10: end if
11: Limit |A_k| so that |A_k| <= ceil(rho_max |C_k|)

Phase B: utility-weighted aggregation
12: for each selected client i in A_k do
13:     Delta_i <- theta_i^k - theta^{k-1}
14: end for
15: theta^k <- theta^{k-1}
              + eta * sum_{i in A_k} u_i Delta_i / sum_{i in A_k} u_i
16: Compute round-level system utility
17: Update server-side critics and centralized mixer
18: Save theta^k and monitoring records
19: return theta^k
```

The advantage of FMRL-LA is that it can suppress low-quality or unstable client updates under heterogeneous data distributions. The tradeoff is additional communication and coordination overhead, because the method requires audit metadata, client selection, cached uploads, and server-side utility learning. This overhead is justified when robustness and final utility are more important than minimizing coordination cost.

### 4.7 EVT Open-Set Calibration

The final stage converts the trained classifier into an open-set detector. EVT is fitted after federated training, using only validation samples that are both known and correctly classified. For each known class, the reconstruction errors form a class-conditional distribution, and the upper tail of that distribution is modeled with a generalized Pareto distribution. This yields a probabilistic unknown score for each sample.

The calibration procedure is:

1. Run the trained global model on validation data.
2. Keep reconstruction errors only for correctly classified known samples.
3. Fit an EVT tail model per class.
4. Estimate per-class unknown probabilities from reconstruction error.
5. Set the global threshold `delta` using validation known-sample probabilities.

At inference time, a sample is rejected as unknown when its EVT score exceeds the calibrated threshold. If no EVT tail is available for the predicted class, the safe behavior is to treat the sample as unknown rather than silently accepting it as known.

```text
Algorithm 4: EVT Calibration and Open-Set Inference

Input:
    Validation set D_val
    Sample x for inference
    Trained global model theta*
    EVT tail fraction alpha_evt
    Target known false-positive rate beta

Output:
    EVT models M_evt
    Global threshold delta
    Final prediction y_hat

Calibration:
1:  Initialize empty reconstruction-error set E_c for each known class c
2:  for each validation sample (s,y) in D_val do
3:      c_hat <- argmax_c Q_psi(p_theta(s), s, c)
4:      if c_hat = y then
5:          z <- q_phi(s,c_hat)
6:          s_hat <- G_omega(z,c_hat)
7:          e <- MSE(s_hat, s)
8:          Add e to E_{c_hat}
9:      end if
10: end for
11: for each known class c with sufficient errors E_c do
12:     Choose tail threshold u_c from the upper alpha_evt tail of E_c
13:     Fit a generalized Pareto distribution to excesses e - u_c
14:     Store fitted model M_evt[c]
15: end for
16: Compute validation unknown probabilities using M_evt
17: Set delta to the (1 - beta) quantile of validation unknown probabilities

Inference:
18: c_hat <- argmax_c Q_psi(p_theta(x), x, c)
19: z <- q_phi(x,c_hat)
20: x_hat <- G_omega(z,c_hat)
21: e_x <- MSE(x_hat, x)
22: p_u <- M_evt[c_hat](e_x)
23: if p_u >= delta then
24:     y_hat <- Unknown
25: else
26:     y_hat <- c_hat
27: end if
28: return M_evt, delta, y_hat
```

This stage is critical for the security setting because it provides a calibrated rejection mechanism instead of a raw confidence threshold. The main benefit is that unknown detection is tied to the reconstruction geometry of the known classes. The main limitation is sensitivity to validation quality and EVT tail size, so the results section should report both threshold-independent metrics and threshold-dependent metrics.

## Summary

The proposed method is intentionally structured as one pipeline rather than a collection of isolated modules. Local CVAE-DQN learning provides the known-class representation, federated aggregation shares useful knowledge across clients, FMRL-LA improves robustness under heterogeneity, and EVT adds a calibrated open-set rejection layer. Together, these components form the cooperative federated open-set intrusion detection framework described by the repository.
