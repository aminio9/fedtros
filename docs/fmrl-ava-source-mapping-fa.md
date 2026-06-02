# نگاشت منبع و پیاده‌سازی روش FMRL-AVA

نام جدید روش پیشنهادی `FMRL-AVA` است:

`Federated Multi-Agent Reinforcement Learning with Adaptive Vector-Aligned Aggregation`

این روش پیاده‌سازی مستقیم هیچ‌کدام از دو مقاله مرجع نیست. FMRL-AVA ترکیبی از ایده‌های `FMRL-LA` و `FedAWA` است که برای مسئله تشخیص نفوذ مجموعه‌باز در CF-MARLOS-AVA بازطراحی شده است.

## رفرنس‌های اصلی

1. Zhang et al., "Towards Cost-Efficient Federated Multi-agent RL with Learnable Aggregation", PAKDD 2024.
   DOI: `10.1007/978-981-97-2253-2_14`

2. Shi et al., "FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors", CVPR 2025.

## نگاشت ایده به کد

| بخش پیاده‌سازی | فایل کد | منبع اصلی | تغییر اختصاصی در CF-MARLOS-AVA |
| --- | --- | --- | --- |
| دور دو مرحله‌ای Phase A/Phase B | `src/federated/server.py`, `src/federated/client.py` | FMRL-LA | به‌جای محیط MARL اصلی، کلاینت CVAE-DQN محلی روی داده تشخیص نفوذ آموزش می‌بیند. در Phase A فقط audit metadata ارسال می‌شود و در Phase B فقط کلاینت‌های منتخب وزن cache شده را آپلود می‌کنند. |
| منتقد ناهمگام هر کلاینت | `src/federated/server_models.py::AsyncCritic` | FMRL-LA | ورودی critic از latent prior summary و ویژگی‌های ممیزی مثل F1، accuracy، TD stability، novelty، entropy، coverage و generator quality ساخته می‌شود. |
| تجمیع‌گر/میکسر مرکزی | `src/federated/server_models.py::CentralizedAggregator` | FMRL-LA | خروجی mixer برای آموزش و پایش utility استفاده می‌شود، اما وزن مستقیم aggregation نیست. این کار از وابستگی بیش از حد به reward دستی جلوگیری می‌کند. |
| utility و client selection | `src/federated/selection_utils.py`, `src/federated/server.py` | FMRL-LA | utility با audit score قطعی و critic residual محدود ترکیب می‌شود. utility پایین‌تر از آستانه می‌تواند upload را حذف کند، ولی `min_selected_clients` از خالی شدن round جلوگیری می‌کند. |
| client vector / update vector | `src/federated/server.py` | FedAWA | بردار کلاینت به شکل `Delta_i = theta_i^k - theta^{k-1}` ساخته می‌شود. برخلاف FedAWA اصلی، وزن‌ها با optimization جداگانه روی proxy data یاد گرفته نمی‌شوند؛ چون پروژه داده proxy ندارد و حفظ حریم خصوصی مهم است. |
| ضریب هم‌راستایی بردار | `src/federated/selection_utils.py::alignment_multiplier` | FedAWA | از cosine بین `Delta_i` و جهت مرجع همان round استفاده می‌شود. ضریب به شکل bounded exponential محاسبه می‌شود تا updateهای هم‌راستا تقویت و updateهای مخالف تضعیف شوند. |
| وزن نهایی aggregation | `src/federated/server.py` | ترکیب FMRL-LA و FedAWA | وزن پایه `b_i = n_i u_i` از sample count و utility می‌آید؛ سپس ضریب هم‌راستایی `m_i` اضافه می‌شود و وزن نهایی `a_i = b_i m_i` است. |
| reward تیمی validation/support | `src/federated/selection_utils.py`, `src/federated/server.py` | تغییر اختصاصی | reward تیمی فقط target آموزش/مانیتورینگ critic/mixer است و مستقیما وزن aggregation را تعیین نمی‌کند. برای open-set، termهای AUROC، unknown F1 و rejection quality در validation reward لحاظ می‌شوند. |
| مانیتورینگ | `fmrl_ava_monitoring.jsonl` | تغییر اختصاصی | علاوه بر utility و selected fraction، فیلدهای `alignment_cosine`, `alignment_multiplier`, `base_aggregation_weight`, و `aggregation_weight` ذخیره می‌شوند. |

## فرمول پیاده‌سازی

برای کلاینت منتخب \(i\):

$$
\Delta_i = \theta_i^k - \theta^{k-1}
$$

وزن پایه:

$$
b_i = n_i u_i
$$

جهت مرجع round:

$$
\bar{\Delta} =
\frac{\sum_{j \in A_k} b_j \Delta_j}
{\sum_{j \in A_k} b_j}
$$

ضریب هم‌راستایی بردار:

$$
m_i =
\mathrm{clip}
\left(
\exp(\kappa \cos(\Delta_i,\bar{\Delta})),
m_{\min},
m_{\max}
\right)
$$

وزن نهایی:

$$
a_i = b_i m_i
$$

آپدیت سرور:

$$
\theta^k =
\theta^{k-1}
+ \eta
\frac{\sum_{i \in A_k} a_i \Delta_i}
{\sum_{i \in A_k} a_i}
$$

## چرا این تغییر منطقی‌تر است؟

در حالت non-IID، FedAvg فقط از تعداد نمونه استفاده می‌کند و نمی‌فهمد update یک کلاینت با جهت کلی آموزش سازگار است یا نه. FMRL-LA اصلی utility را وارد انتخاب و وزن‌دهی می‌کند، اما جهت واقعی پارامترها را نمی‌سنجد. FedAWA نشان می‌دهد بردار update کلاینت اطلاعات مهمی درباره تفاوت داده محلی دارد. FMRL-AVA این دو ایده را ترکیب می‌کند: اول clientهای کم‌فایده را با utility کنترل می‌کند، سپس بین کلاینت‌های منتخب، updateهای هم‌راستا را بیشتر و updateهای ناسازگار را کمتر وارد مدل جهانی می‌کند.

در حالت IID، چون scoreها و جهت updateها به هم نزدیک‌اند، utilityها حول 1 می‌مانند و alignment multiplierها تقریبا مشابه می‌شوند؛ بنابراین aggregation به رفتار FedAvg نزدیک می‌شود.

