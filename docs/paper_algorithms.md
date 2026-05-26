# Paper Algorithms for the Methodology Section

This draft contains the algorithm blocks that are most appropriate for the paper. The standard FedAvg and FedProx rules are included inside the main training algorithm and discussion, but they are not shown as a separate algorithm because they are baseline aggregation methods.

## Notation

- `N`: number of clients.
- `K`: number of federated rounds.
- `D_i`: local training dataset of client `i`.
- `D_val`: validation/calibration dataset.
- `theta = {theta_p, theta_r, theta_Q, theta_G}`: federated parameters of the prior network, recognition network, main Q-network, and generator.
- `theta_Q^-`: local target Q-network parameters.
- `mu`: FedProx proximal coefficient. `mu = 0` for FedAvg and FMRL-LA in the current implementation.
- `S`: aggregation strategy, where `S in {FedAvg, FedProx, FMRL-LA}`.
- `M_evt`: class-conditional EVT models.
- `delta`: calibrated global open-set rejection threshold.

## Algorithm 1: Federated CVAE-DQN Training and Open-Set Calibration

```text
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
2:  for each federated round k = 1, ..., K do
3:      Server selects or samples available clients C_k
4:      Server broadcasts theta^{k-1} to clients in C_k
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

### Analysis

This algorithm should be the first algorithm in the paper because it connects all major stages of the proposed method. It shows that federated learning is used for trainable neural parameters, while EVT is applied after training as a calibration stage. FedAvg and FedProx are handled as baseline strategies through the weighted average in line 12; the difference is that FedProx activates the local proximal coefficient in line 6. FMRL-LA replaces the standard average with a utility-aware update in lines 9-10. This framing makes the comparison fair because all strategies use the same client partitions, local model, validation set, and open-set calibration procedure.

## Algorithm 2: Client-Side CVAE-DQN Local Update with Optional FedProx

```text
Input:
    Local client dataset D_i
    Broadcast global parameters theta^{k-1}
    Local training configuration H
    Proximal coefficient mu_i

Output:
    Updated client parameters theta_i^k
    Client diagnostics m_i^k

1:  Load theta^{k-1} into prior p_theta(z|s), recognition q_phi(z|s,a),
    main Q-network Q_psi(z,s), and generator G_omega(z,a)
2:  Set target Q-network parameters theta_Q^- <- theta_Q
3:  Store theta^{k-1} as the FedProx reference
4:  Initialize local replay buffer B_i and epsilon-greedy policy pi_i
5:  for each local episode e = 1, ..., E do
6:      Reset local environment and observe state s_t
7:      for each step t = 1, ..., T do
8:          Select action a_t using pi_i and Q_psi
9:          Observe reward r_t, next state s_{t+1}, done flag d_t,
            and true class label a_t*
10:         Store transition (s_t, a_t, r_t, s_{t+1}, d_t, a_t*) in B_i
11:         if |B_i| >= minimum buffer size then
12:             Sample mini-batch b from B_i
13:             Compute prior loss:
                    L_p = KL(q_phi(z|s,a*) || p_theta(z|s))
14:             if mu_i > 0 then
15:                 L_p <- L_p + (mu_i / 2) ||theta_p - theta_p^{k-1}||_2^2
16:             end if
17:             Update theta_p using gradient descent on L_p
18:             Compute Double-DQN target:
                    y = r + gamma(1-d) Q_{theta_Q^-}(z', s',
                        argmax_a Q_psi(z', s', a))
19:             Compute TD loss:
                    L_Q = Huber(Q_psi(z, s, a) - y)
20:             if mu_i > 0 then
21:                 L_Q <- L_Q + (mu_i / 2)(
                          ||theta_r - theta_r^{k-1}||_2^2
                        + ||theta_Q - theta_Q^{k-1}||_2^2)
22:             end if
23:             Update theta_r and theta_Q using gradient descent on L_Q
24:             Periodically update target network:
                    theta_Q^- <- tau theta_Q + (1 - tau) theta_Q^-
25:         end if
26:     end for
27: end for
28: Build correctly classified set:
        C_i = {(s,a*) in D_i : argmax_a Q_psi(p_theta(s), s, a) = a*}
29: if generator training is enabled and |C_i| is sufficient then
30:     for each mini-batch (s,a*) from C_i do
31:         Sample latent z from q_phi(z|s,a*)
32:         Reconstruct s_hat = G_omega(z,a*)
33:         Compute reconstruction loss L_G = MSE(s_hat, s)
34:         if mu_i > 0 then
35:             L_G <- L_G + (mu_i / 2)||theta_G - theta_G^{k-1}||_2^2
36:         end if
37:         Update theta_G using gradient descent on L_G
38:     end for
39: end if
40: Set theta_i^k = {theta_p, theta_r, theta_Q, theta_G}
41: Compute diagnostics m_i^k including reward, accuracy, F1, TD loss,
    KL loss, generator quality, class entropy, label coverage, and step count
42: return theta_i^k, m_i^k
```

### Analysis

This algorithm is the most important client-side method because it explains why the local update is more than a standard supervised classifier update. The prior network is trained by matching the recognition posterior, while the recognition and main Q-network are trained with a Double-DQN TD objective. The target Q-network is deliberately local: it stabilizes the TD target but is not federated as a separate parameter block. The generator is trained only on correctly classified known samples, which reduces reconstruction contamination and supports the EVT open-set detector. The FedProx terms are optional and appear only when `mu_i > 0`, so the same algorithm covers FedAvg, FedProx, and the client update used inside FMRL-LA.

## Algorithm 3: FMRL-LA Utility-Aware Client Selection and Aggregation

```text
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
2:      Extract latent summary h_i from local prior statistics
3:      Extract scalar diagnostics x_i from m_i^k:
            reward, historical reward, F1, accuracy, TD stability,
            novelty, class entropy, label coverage, generator quality,
            and local step count
4:      Estimate client utility u_i = C_i(h_i, x_i) using the client critic
5:      Clip or temperature-scale u_i for numerical stability
6:  end for
7:  Select clients A_k = {i : u_i >= lambda}
8:  if |A_k| < q_min then
9:      Add highest-utility clients until |A_k| = q_min
10: end if
11: Limit |A_k| so that |A_k| <= ceil(rho_max |C_k|)

Phase B: utility-weighted model update
12: for each selected client i in A_k do
13:     Compute local parameter delta Delta_i = theta_i^k - theta^{k-1}
14: end for
15: Compute utility-weighted update:
        theta^k = theta^{k-1}
                  + eta * sum_{i in A_k} u_i Delta_i / sum_{i in A_k} u_i
16: Compute round-level system utility from reward, F1, accuracy,
    TD stability, novelty, and communication efficiency
17: Update the server-side critics and centralized mixer using system utility
18: Save theta^k and FMRL-LA monitoring records
19: return theta^k
```

### Analysis

This is the algorithm that most clearly distinguishes the proposed cooperative federated approach from standard FL. FedAvg assumes every participating client contributes according to sample count, while FMRL-LA estimates whether an update is useful for the global objective before using it. This is especially important under non-IID partitions, where some clients may have missing classes, unstable TD updates, or poor generator quality. The main benefit is robustness: high-quality and informative updates can receive more influence, while low-utility updates can be ignored or down-weighted. The tradeoff is overhead. FMRL-LA requires audit metadata, server-side critics, a mixer, and a two-phase round structure, so it should be evaluated with both accuracy and communication cost.

## Algorithm 4: EVT Calibration and Open-Set Inference

```text
Input:
    Validation set D_val
    Test or deployment sample x
    Trained global model theta*
    Tail fraction alpha_evt
    Target known false-positive rate beta

Output:
    EVT models M_evt
    Global rejection threshold delta
    Final prediction y_hat

Calibration:
1:  Initialize empty error sets E_c for each known class c
2:  for each validation sample (s,y) in D_val do
3:      Predict known class c_hat = argmax_c Q_psi(p_theta(s), s, c)
4:      if c_hat = y then
5:          Compute latent code z = q_phi(s,c_hat)
6:          Reconstruct s_hat = G_omega(z,c_hat)
7:          Compute reconstruction error e = MSE(s_hat, s)
8:          Add e to E_{c_hat}
9:      end if
10: end for
11: for each known class c with sufficient errors E_c do
12:     Choose threshold u_c from the upper alpha_evt tail of E_c
13:     Fit a generalized Pareto distribution to excesses e - u_c
14:     Store the fitted EVT model M_evt[c]
15: end for
16: Compute unknown probabilities for correctly classified validation samples
17: Set delta to the (1 - beta) quantile of validation unknown probabilities

Inference:
18: For sample x, predict c_hat = argmax_c Q_psi(p_theta(x), x, c)
19: Reconstruct x_hat using q_phi(x,c_hat) and G_omega
20: Compute reconstruction error e_x = MSE(x_hat, x)
21: Compute unknown probability p_u = M_evt[c_hat](e_x)
22: if p_u >= delta then
23:     y_hat <- Unknown
24: else
25:     y_hat <- c_hat
26: end if
27: return M_evt, delta, y_hat
```

### Analysis

This algorithm should appear because it explains how the method moves from closed-set classification to open-set intrusion detection. EVT is not federated as a trainable model; it is fitted after global training using validation reconstruction errors. Only correctly classified known samples are used for tail fitting, which makes the reconstruction-error distribution class-consistent and reduces calibration noise. The benefit is a thresholded unknown-rejection mechanism that is more principled than softmax confidence. The main limitation is calibration sensitivity: tail size, validation quality, and the target known false-positive rate can change the unknown/known tradeoff. For this reason, the paper should report threshold-independent metrics such as AUROC/AUPRC and threshold-dependent metrics such as unknown F1 and known accuracy after rejection.

## Recommended Placement In The Paper

- Use Algorithm 1 at the start of the methodology to summarize the full training and calibration pipeline.
- Use Algorithm 2 in the local learning subsection because it defines the CVAE-DQN update, generator training, and FedProx term.
- Use Algorithm 3 in the federated optimization subsection because FMRL-LA is the custom cooperative aggregation method.
- Use Algorithm 4 in the open-set detection subsection.
- Describe FedAvg and FedProx in text with their equations rather than giving them a separate algorithm block.
