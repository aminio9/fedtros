# توضیح دقیق FMRL-AVA با تجمیع هم‌راستای بردار آپدیت

نام کامل روش:

`Federated Multi-Agent Reinforcement Learning with Adaptive Vector-Aligned Aggregation`

این روش بر پایه ترکیب دو مقاله ساخته شده است: از FMRL-LA ایده ارتباط دو مرحله‌ای، critic ناهمگام و mixer مرکزی گرفته شده، و از FedAWA ایده استفاده از client vector یا بردار آپدیت برای تنظیم وزن aggregation وارد شده است. پیاده‌سازی فعلی یک نسخه اختصاصی برای CF-MARLOS-AVA است و با هیچ‌کدام از دو مقاله مرجع برابر نیست. نگاشت دقیق کد به مقاله‌ها در `docs/fmrl-ava-source-mapping-fa.md` آمده است.

این نسخه، پیاده‌سازی فعلی FMRL-AVA را بعد از اصلاح جدید توضیح می‌دهد. تغییر اصلی این است که وزن aggregation دیگر فقط از reward یا utility دستی نمی‌آید. روش جدید ایده FedAWA-style را وارد می‌کند: هر کلاینت علاوه بر تعداد نمونه و utility، بر اساس هم‌راستایی بردار آپدیت مدلش با جهت کلی آپدیت همان round وزن می‌گیرد.

هدف طراحی:

1. در حالت IID رفتار نزدیک FedAvg بماند.
2. در حالت non-IID کلاینت‌هایی که آپدیت ناسازگار یا مخالف جهت کلی دارند، وزن کمتری بگیرند.
3. client selection قبلی حفظ شود تا کلاینت‌های بسیار ضعیف اصلا وارد upload نشوند.
4. reward تیمی validation-based فقط برای آموزش/مانیتورینگ critic و mixer استفاده شود، نه منبع مستقیم وزن aggregation.

## 1. ساختار دو فاز

هر round منطقی FMRL-AVA دو فاز دارد.

```text
Phase A:
server -> clients: global model
clients -> server: audit metadata

Phase B:
server -> selected clients: upload request
selected clients -> server: cached model weights
server: vector-aligned aggregation
```

در Phase A همه کلاینت‌های sampled محلی آموزش می‌بینند، اما وزن کامل مدل را نمی‌فرستند. فقط خلاصه‌های سبک مانند hidden state، reward، F1، accuracy، TD stability، novelty، entropy، coverage و کیفیت generator ارسال می‌شود.

در Phase B فقط کلاینت‌های انتخاب‌شده وزن‌های cache شده را upload می‌کنند. سپس سرور aggregation را با وزن هم‌راستایی بردار آپدیت انجام می‌دهد.

## 2. محاسبه utility برای client selection

برای هر کلاینت \(i\)، سرور دو نوع score می‌سازد.

اول score ممیزی قطعی:

$$
q_i^k =
0.25F1_i
+0.20Acc_i
+0.20S_{TD,i}
+0.15Cov_i
+0.10\rho_i
+0.05\bar{\rho}_i
+0.05G_i
$$

که در آن:

- \(F1_i\): F1 محلی یا audit F1
- \(Acc_i\): accuracy محلی یا policy accuracy
- \(S_{TD,i}=1/(1+\max(e_{TD,i},0))\): پایداری TD
- \(Cov_i\): کیفیت پوشش داده، از entropy و label coverage
- \(\rho_i\): reward اخیر نرمال‌شده
- \(\bar{\rho}_i\): reward تاریخی نرمال‌شده
- \(G_i\): نسبت نمونه‌های درست برای آموزش generator

دوم score critic:

$$
c_i^k =
\frac{
C_i(h_i^k,x_i^k)
}{
1+C_i(h_i^k,x_i^k)
}
$$

خروجی critic فقط residual محدود است. score نهایی:

$$
z_i^k =
(1-\beta)q_i^k + \beta c_i^k
$$

سپس utility حول میانگین همان round مرکزدهی می‌شود:

$$
u_i^k =
\mathrm{clip}
\left(
1+2\gamma(z_i^k-\bar{z}^k),
u_{\min},
u_{\max}
\right)
$$

اگر score کلاینت هم از آستانه کمتر باشد و هم از میانگین round پایین‌تر باشد، utility صفر می‌شود:

$$
u_i^k = 0
\quad
\text{if}
\quad
z_i^k < \lambda
\quad
\text{and}
\quad
z_i^k < \bar{z}^k
$$

این یعنی client selection هنوز فعال است: کلاینتی که utility صفر دارد لازم نیست در Phase B وزن کامل مدل را upload کند.

## 3. وزن پایه شبیه FedAvg

برای کلاینت‌های انتخاب‌شده \(A_k\)، ابتدا وزن پایه ساخته می‌شود:

$$
b_i^k = n_i u_i^k
$$

اینجا \(n_i\) تعداد نمونه‌های کلاینت است. بنابراین prior اصلی FedAvg حفظ می‌شود. اگر داده IID باشد و utilityها نزدیک 1 باشند:

$$
b_i^k \approx n_i
$$

پس روش به FedAvg نزدیک می‌شود.

## 4. بردار آپدیت کلاینت

هر کلاینت انتخاب‌شده مدل محلی cache شده را upload می‌کند. سرور delta را نسبت به مدل جهانی قبلی می‌سازد:

$$
\Delta_i^k =
\Theta_i^k - \Theta^{k-1}
$$

این همان client vector است. در non-IID، فقط خوب بودن metric محلی کافی نیست؛ جهت حرکت مدل هم مهم است. ممکن است یک کلاینت روی داده خودش خوب باشد، اما آپدیتش با جهت کلی مدل جهانی ناسازگار باشد.

## 5. جهت مرجع round

سرور ابتدا جهت مرجع را با وزن پایه می‌سازد:

$$
\bar{\Delta}^k =
\frac{
\sum_{j \in A_k} b_j^k \Delta_j^k
}{
\sum_{j \in A_k} b_j^k
}
$$

این جهت همان چیزی است که اگر alignment فعال نبود، مدل جهانی بر اساس آن حرکت می‌کرد.

## 6. هم‌راستایی FedAWA-style

برای هر کلاینت cosine similarity بین آپدیت خودش و جهت مرجع محاسبه می‌شود:

$$
s_i^k =
\cos(\Delta_i^k,\bar{\Delta}^k)
$$

سپس multiplier هم‌راستایی ساخته می‌شود:

$$
m_i^k =
\mathrm{clip}
\left(
\exp(\kappa s_i^k),
m_{\min},
m_{\max}
\right)
$$

اگر کلاینت با جهت کلی هم‌راستا باشد، \(s_i^k\) مثبت و \(m_i^k\) بزرگ‌تر از 1 می‌شود. اگر مخالف باشد، \(s_i^k\) منفی و \(m_i^k\) کمتر از 1 می‌شود. اگر \(\kappa=0\)، همه multiplierها 1 می‌شوند و این بخش عملا خاموش است.

## 7. وزن نهایی aggregation

وزن نهایی کلاینت:

$$
a_i^k = b_i^k m_i^k = n_i u_i^k m_i^k
$$

و آپدیت مدل جهانی:

$$
\Theta^k =
\Theta^{k-1}
+
\alpha
\frac{
\sum_{i \in A_k} a_i^k\Delta_i^k
}{
\sum_{i \in A_k} a_i^k
}
$$

این فرمول سه منطق را با هم ترکیب می‌کند:

1. \(n_i\): حفظ رفتار FedAvg و اثر تعداد نمونه‌ها
2. \(u_i^k\): انتخاب و تعدیل کیفیت کلاینت
3. \(m_i^k\): اعتماد بیشتر به آپدیت‌هایی که با جهت کلی مدل هم‌راستا هستند

## 8. نقش reward تیمی بعد از اصلاح

reward تیمی validation-based حذف نشده، اما دیگر وزن مستقیم aggregation را تعیین نمی‌کند. نقش آن این است که critic و mixer سمت سرور را آموزش دهد و برای monitoring کیفیت round استفاده شود.

اگر metricهای validation موجود باشند:

$$
R_{\mathrm{val}} =
\eta_1F1_{\mathrm{macro}}^{\mathrm{val}}
+\eta_2BAcc^{\mathrm{val}}
+\eta_3AUROC_{\mathrm{open}}
+\eta_4F1_{\mathrm{unknown}}
+\eta_5Rej_{\mathrm{open}}
$$

اگر بعضی metricها در حالت ارزیابی فعلی تولید نشوند، از مخرج وزن‌دهی حذف می‌شوند و صفر حساب نمی‌شوند. بنابراین closed-set یا IID به خاطر نبودن metricهای open-set جریمه مصنوعی نمی‌شود.

یک support reward هم برای fallback و پایداری باقی می‌ماند:

$$
R_{\mathrm{sup}} =
\xi_1F1_{\mathrm{local}}
+\xi_2BAcc_{\mathrm{local}}
+\xi_3S_{TD}
+\xi_4Cov
+\xi_5G
+\xi_6C_{\mathrm{comm}}
$$

هدف آموزش mixer:

$$
U_{\mathrm{mixer}} =
\lambda R_{\mathrm{val}}
+(1-\lambda)R_{\mathrm{sup}}
$$

اما نکته مهم این است:

$$
U_{\mathrm{mixer}}
\neq
a_i^k
$$

یعنی reward تیمی فقط target آموزشی/مانیتورینگ است، نه وزن aggregation.

## 9. چرا این روش منطقی‌تر است؟

روش قبلی بیشتر به reward و metricهای دستی وابسته بود. این برای intrusion detection مفید است، اما برای aggregation فدرال یک ضعف دارد: metric خوب محلی همیشه به معنی آپدیت مفید جهانی نیست.

روش vector-aligned بهتر است چون:

- تصمیم وزن‌دهی از خود parameter update می‌آید.
- در non-IID، آپدیت‌های پرت یا مخالف جهت کلی کمتر اثر می‌گذارند.
- در IID، چون آپدیت‌ها هم‌جهت هستند، multiplierها تقریبا مشابه می‌شوند و بعد از نرمال‌سازی رفتار نزدیک FedAvg می‌ماند.
- با FedAWA/FedLAW سازگارتر است، چون aggregation weight را به learnable/adaptive weighting نزدیک می‌کند.
- reward validation-based همچنان برای آموزش critic/mixer و تحلیل open-set حفظ می‌شود، اما aggregation را به یک proxy دستی وابسته نمی‌کند.

## 10. فیلدهای مهم در کد

تنظیمات اصلی در `src/configs/federated/default.yaml`:

```yaml
alignment_strength: 0.50
min_alignment_multiplier: 0.50
max_alignment_multiplier: 2.00
validation_reward_blend: 0.85
validation_reward_ema_decay: 0.80
```

فیلدهای monitoring در `fmrl_ava_monitoring.jsonl`:

- `base_aggregation_weight`: مقدار \(b_i=n_i u_i\)
- `alignment_cosine`: مقدار \(s_i\)
- `alignment_multiplier`: مقدار \(m_i\)
- `aggregation_weight`: مقدار نهایی \(a_i=n_i u_i m_i\)
- `validation_team_reward`: reward validation برای mixer
- `support_reward`: reward پشتیبان

## 11. جمع‌بندی

نسخه جدید FMRL-AVA این‌گونه است:

1. Phase A با audit و critic کلاینت‌های ضعیف را انتخاب/حذف می‌کند.
2. Phase B فقط از کلاینت‌های انتخاب‌شده weight upload می‌گیرد.
3. وزن پایه مثل FedAvg برابر \(n_i\) است، اما با utility تعدیل می‌شود.
4. وزن نهایی با هم‌راستایی بردار آپدیت اصلاح می‌شود.
5. reward تیمی برای آموزش critic/mixer و گزارش کیفیت باقی می‌ماند، نه برای تعیین مستقیم aggregation.

این طراحی برای non-IID منطقی‌تر است و در IID همچنان رفتار FedAvg-like دارد.

## FMRL-AVA-GLOW patch: research-grade non-IID alpha=0.1 mode

FMRL-AVA-GLOW is the fixed FMRL-AVA configuration used for severe Dirichlet label skew. GLOW means: Gradient-safe contextual-bandit local RL, Local proximal and latent regularization, Outcome-rewarded server critic trained from validation advantage, and Warm FedAvg-anchored weakly aligned aggregation.

The local traffic environment is treated as a contextual-bandit classification problem. A client action predicts the label of the current traffic sample, but it does not cause the next traffic sample. For that reason `training.rl_mode` is `contextual_bandit` and `training.gamma` is set to `0.0`. The old TD path remains in the code with a small weight for backward compatibility, but the main local Q signal is now full-action bandit supervision: the true class receives `reward.correct`, locally present wrong classes receive `reward.incorrect`, and locally absent classes receive no Q-gradient.

Missing-class protection is active only during local client training. Each client reports its local label histogram; the agent stores `local_class_counts` and masks absent local logits in the supervised classification loss using `training.missing_class_gradient.mask_value`. The epsilon-greedy policy also restricts random and greedy choices to labels available on the client shard. Global evaluation and inference do not apply this mask.

Class-aware aggregation multipliers are disabled for FMRL-AVA-GLOW because prior project experiments showed they hurt this method. The class-aware code remains available for FedMADE-style experiments, but `profile_balance_strength`, `profile_quality_blend`, and `profile_cluster_strength` are set to zero and the profile multiplier bounds are fixed to `1.0` for GLOW.

Server selection is coverage-safe. Warmup rounds select all clients. After warmup, the server may drop only low-utility clients while keeping at least 90% of the sampled clients and preserving nonzero aggregate support for every class. For the standard 10-client alpha=0.1 setting this keeps 9-10 clients instead of turning selection into a class-collapse machine, because apparently non-IID data was not chaotic enough already.

Vector alignment is now FedAvg-anchored. The reference delta is computed from plain sample-count FedAvg before utility, drift, profile, or alignment modifiers. Alignment is a weak bounded stability signal only, with default bounds `[0.95, 1.05]` in GLOW.

The server critic/mixer is trained from outcome reward. `_current_utility_tensor()` now uses differentiable critic outputs, and `_train_server_models()` optimizes mixer MSE plus critic MSE targets adjusted by validation/support advantage. Critic influence on selection is delayed by `critic_activation_round`; before activation, selection uses the deterministic audit score. The critic is auxiliary and lightly blended after activation, not a magical aggregation oracle wearing a lab coat.

Server Adam/Yogi is disabled in the first fixed method. GLOW uses `server_optimizer: none` and `aggregation_lr: 1.0`, so the server applies the bounded weighted delta directly. Adaptive server optimizers remain in code for later ablations, but they are not active in the first stability patch.

Primary metrics are macro-F1, balanced accuracy, worst-class F1, and minority-class recall. Overall accuracy is secondary because the dataset is imbalanced and majority-class accuracy can look impressive while minority classes quietly burn.

### GLOW ablation plan

B0: FedAvg current baseline.
B1: FedAvg plus fixed contextual-bandit RL.
B2: FMRL-AVA current pre-GLOW behavior.
B3: FMRL-AVA plus contextual-bandit RL.
B4: add missing-class gradient mask.
B5: add local proximal regularization.
B6: add coverage-safe selection.
B7: add weak FedAvg-anchored vector alignment.
B8: add trained server critic after round 40.

### Validation commands

```bash
python -m pytest tests -q
python run.py experiment=smoke +method=fmrl_ava_glow runtime=tiny output=tiny
python run.py experiment=exp3 +method=fmrl_ava_glow seed=42 dataset.preprocessing.alpha=0.1 runtime=tiny output=tiny
```

Fair FedAvg comparison with the same local RL fixes:

```bash
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1 \
  training.rl_mode=contextual_bandit \
  training.gamma=0.0 \
  training.epsilon_start=0.30 \
  training.epsilon_end=0.02 \
  training.epsilon_decay_rate=0.97 \
  training.loss_weights.prior_kl=0.5 \
  training.loss_weights.q_td=0.25 \
  training.loss_weights.bandit_q=1.0 \
  training.loss_weights.classification=2.0 \
  training.imbalance.enabled=true \
  training.imbalance.weight_mode=effective_number \
  training.imbalance.effective_number_beta=0.999 \
  training.imbalance.min_weight=0.3 \
  training.imbalance.max_weight=3.0 \
  training.imbalance.class_balanced_sampling=true \
  training.imbalance.weighted_reward=true \
  training.classification_loss.focal_gamma=1.5 \
  training.auxiliary_losses.supervised_contrastive_lambda=0.02 \
  training.auxiliary_losses.center_loss_lambda=0.01 \
  training.kl.free_nats=0.25 \
  training.kl.warmup_steps=200 \
  training.missing_class_gradient.enabled=true \
  training.missing_class_gradient.mask_value=-20.0
```

### Sources used for method justification

- Wong, H. Y., Lim, C. K., Chan, C. S. “Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling.” Pattern Recognition, 2026. DOI: 10.1016/j.patcog.2026.113900.
- Chowdhury, S., et al. “Confusion-Calibrated Cross-Entropy and Class-Specialized Aggregation for Robust Federated Learning under Extreme Data Heterogeneity.” Knowledge-Based Systems, 2026. DOI: 10.1016/j.knosys.2026.115497.
- Saha, P., Mishra, D., Wagner, F., Kamnitsas, K., Noble, J. A. “FedExIT - Missing Class-agnostic Semi-Supervised Federated Learning with Extreme Imbalance Tackling Scheme.” Information Fusion, 2026. DOI: 10.1016/j.inffus.2025.104080.
- Wang, X., Wu, Z., Zhu, J. “FedAPE: Heterogeneous Federated Learning with Attention-guided Aggregation and Prototype Enhancement.” Future Generation Computer Systems, 2026. DOI: 10.1016/j.future.2026.108417.
- Jing, Y., Guo, B., Li, N., Xu, R., Yu, Z. “Federated Multi-Agent Reinforcement Learning: A Comprehensive Survey of Methods, Applications and Challenges.” Expert Systems with Applications, 2025. DOI: 10.1016/j.eswa.2025.128729.
- Giuseppi, A., Menegatti, D., Pietrabissa, A. “Enhancing Federated Reinforcement Learning: A Consensus-based Approach for Both Homogeneous and Heterogeneous Agents.” Machine Intelligence Research, 2025. DOI: 10.1007/s11633-025-1550-8.
- Mnih, V., et al. “Human-level control through deep reinforcement learning.” Nature, 2015. DOI: 10.1038/nature14236.
- van Hasselt, H., Guez, A., Silver, D. “Deep Reinforcement Learning with Double Q-learning.” AAAI, 2016. DOI: 10.1609/aaai.v30i1.10295.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. “Focal Loss for Dense Object Detection.” ICCV, 2017. DOI: 10.1109/ICCV.2017.324.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S. “Class-Balanced Loss Based on Effective Number of Samples.” CVPR, 2019. DOI: 10.1109/CVPR.2019.00949.
- Hou, W., Chen, T., Wang, F., Wu, T., Zheng, Z., Tang, S., Lim, W. Y. B. “FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization.” CVPR, 2026. Use the official CVPR OpenAccess URL if no DOI is available; do not invent one.
