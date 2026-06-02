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
