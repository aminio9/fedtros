# توضیح دقیق FMRL-LA مطابق پیاده‌سازی CF-MARLOS

این نسخه، روش FMRL-LA را مطابق کد فعلی توضیح می‌دهد. هدف طراحی این است که در حالت IID مثل FedAvg رفتار کند، اما در حالت non-IID بتواند کلاینت‌های ناپایدار، کم‌پوشش یا کم‌کیفیت را کمتر وارد مدل جهانی کند.

ایده اصلی این است:

- پایه تجمیع همچنان تعداد نمونه‌های کلاینت است، مثل FedAvg.
- utility فقط یک ضریب تعدیل محدود است، نه جایگزین کامل FedAvg.
- اگر همه کلاینت‌ها شبیه هم باشند، utility همه تقریبا 1 می‌شود.
- اگر یک کلاینت نسبت به بقیه کیفیت پایین‌تری داشته باشد، وزنش کم می‌شود یا در حالت خیلی ضعیف upload نمی‌کند.

## 1. ساختار دو فازه

هر دور منطقی FMRL-LA در کد با دو round در Flower اجرا می‌شود.

فاز A:

```text
server -> clients: global parameters
clients -> server: audit metadata only
```

در این فاز کلاینت مدل را محلی آموزش می‌دهد، وزن‌های جدید را cache می‌کند، اما وزن‌های کامل را نمی‌فرستد. فقط metadata سبک برای محاسبه utility ارسال می‌شود.

فاز B:

```text
server -> selected clients: upload request
selected clients -> server: cached model parameters
```

در این فاز فقط کلاینت‌های انتخاب‌شده وزن‌های cacheشده را upload می‌کنند.

## 2. اطلاعاتی که کلاینت در فاز A می‌فرستد

برای هر کلاینت، سرور دو نوع اطلاعات می‌گیرد.

اول، خلاصه نهفته مدل محلی:

$$
h_i^k =
\frac{1}{|B_i|}
\sum_{s \in B_i}
\mu_{\mathrm{prior}}(s)
$$

این همان `avg_mu_vector` در کد است.

دوم، چند شاخص عددی از کیفیت آموزش و داده محلی:

$$
x_i^k =
[
\rho_i,
\bar{\rho}_i,
F1_i,
Acc_i,
S_i^{TD},
N_i,
H_i,
Cov_i,
G_i,
Step_i
]
$$

اما نکته مهم این است که این بردار مستقیما به‌تنهایی تصمیم‌گیرنده نیست. از آن یک امتیاز audit محافظه‌کارانه ساخته می‌شود:

$$
q_i =
0.25F1_i
+0.20Acc_i
+0.20S_i^{TD}
+0.15D_i
+0.10\rho_i
+0.05\bar{\rho}_i
+0.05G_i
$$

که در آن:

$$
D_i = 0.5H_i + 0.5Cov_i
$$

و:

$$
S_i^{TD} =
\frac{1}{1+\max(e_i^{TD},0)}
$$

این طراحی منطقی‌تر از اتکا به critic خام است، چون:

- F1 و accuracy کیفیت پیش‌بینی محلی را نشان می‌دهند.
- TD stability نشان می‌دهد local RL update چقدر پایدار بوده است.
- entropy و label coverage نشان می‌دهند داده کلاینت چقدر نماینده‌تر است.
- reward و history reward رفتار RL را وارد score می‌کنند.
- generator quality فقط سهم کوچک دارد تا اگر generator غیرفعال یا ضعیف بود کل روش خراب نشود.

## 3. نقش critic یا $C_i$

در کد، $C_i$ همان `AsyncCritic` اختصاصی کلاینت $i$ است. critic هنوز استفاده می‌شود، اما دیگر تصمیم اصلی فقط بر اساس خروجی خام آن نیست.

خروجی critic:

$$
\tilde{u}_i^k =
\mathrm{softplus}
\left(
C_i([h_i^k,x_i^k])
\right)
$$

سپس به یک امتیاز محدود در بازه $[0,1]$ تبدیل می‌شود:

$$
c_i^k =
\frac{\tilde{u}_i^k / \tau}
{1 + \tilde{u}_i^k / \tau}
$$

امتیاز نهایی قبل از centering ترکیب audit score و critic score است:

$$
z_i^k =
(1-\beta)q_i + \beta c_i^k
$$

در پیکربندی پیش‌فرض:

$$
\beta = 0.15
$$

پس critic فقط یک residual محدود است. این کار باعث می‌شود اول آموزش یا در حالت IID، critic تصادفی باعث خراب شدن FedAvg نشود.

## 4. utility نهایی و حفظ رفتار FedAvg در IID

در هر round، سرور میانگین score کلاینت‌ها را حساب می‌کند:

$$
\bar{z}^k =
\frac{1}{|C_k|}
\sum_{i \in C_k}
z_i^k
$$

سپس utility نهایی نسبت به میانگین همان round ساخته می‌شود:

$$
u_i^k =
\mathrm{clip}
\left(
1 + 2\gamma(z_i^k-\bar{z}^k),
u_{\min},
u_{\max}
\right)
$$

در پیکربندی پیش‌فرض:

$$
\gamma=0.75,\quad
u_{\min}=0.25,\quad
u_{\max}=2.0
$$

اگر کلاینت‌ها IID باشند، scoreها نزدیک هم هستند:

$$
z_i^k \approx \bar{z}^k
$$

پس:

$$
u_i^k \approx 1
$$

بنابراین FMRL-LA در IID تقریبا به FedAvg برمی‌گردد.

اگر یک کلاینت نسبت به بقیه کیفیت پایین‌تری داشته باشد:

$$
z_i^k < \bar{z}^k
$$

utility آن کمتر از 1 می‌شود. اگر score هم خیلی پایین‌تر از آستانه باشد، utility صفر می‌شود:

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

این شرط دوم مهم است: اگر همه کلاینت‌ها در یک round ضعیف ولی مشابه باشند، همه حذف نمی‌شوند. حذف فقط وقتی اتفاق می‌افتد که یک کلاینت هم به شکل مطلق ضعیف باشد و هم نسبت به بقیه همان round ضعیف‌تر باشد.

## 5. انتخاب کلاینت

در roundهای warm-up، همه کلاینت‌ها انتخاب می‌شوند. بعد از warm-up:

$$
A_k =
\{i \in C_k \mid u_i^k > 0\}
$$

اگر تعداد کلاینت‌های انتخاب‌شده کمتر از `min_selected_clients` باشد، بهترین کلاینت‌ها اضافه می‌شوند تا round قفل نشود. سپس حداکثر تعداد انتخاب‌شده با `max_selected_fraction` محدود می‌شود:

$$
|A_k|
\le
\lceil \rho_{\max}|C_k| \rceil
$$

اگر هیچ utility مثبتی در round باقی نماند، سرور همان مجموعه sampled را با utility یکنواخت 1 ادامه می‌دهد تا آموزش collapse نکند.

## 6. aggregation نهایی

تغییر مهم جدید این است که aggregation فقط با utility انجام نمی‌شود. وزن واقعی هر کلاینت برابر است با:

$$
a_i^k = n_i u_i^k
$$

که $n_i$ تعداد نمونه‌های کلاینت است.

هر کلاینت انتخاب‌شده delta مدل خود را نسبت به مدل جهانی قبلی می‌فرستد:

$$
\Delta_i^k =
\Theta_i^k - \Theta^{k-1}
$$

مدل جهانی به شکل زیر به‌روزرسانی می‌شود:

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

اگر داده IID باشد و همه utilityها تقریبا 1 باشند:

$$
a_i^k \approx n_i
$$

پس فرمول بالا همان FedAvg sample-weighted روی deltaها می‌شود. در non-IID، $u_i^k$ اثر کلاینت‌های ناپایدار یا کم‌کیفیت را کم می‌کند.

## 7. Centralized Aggregator

`CentralizedAggregator` در کد یک mixer شبیه QMIX است. ورودی آن utilityهای کلاینت‌ها و global state ساخته‌شده از featureهای کلاینتی است:

$$
s_k =
[
h_1^k,x_1^k,
h_2^k,x_2^k,
\ldots,
h_n^k,x_n^k
]
$$

خروجی mixer:

$$
Q_{\mathrm{total}}^k =
M([u_1^k,\ldots,u_n^k],s_k)
$$

در پیاده‌سازی فعلی، mixer مستقیما وزن aggregation همان round را تعیین نمی‌کند. وزن aggregation از $a_i^k=n_i u_i^k$ می‌آید. mixer و criticها بعد از aggregation با utility سطح سیستم آموزش داده می‌شوند تا در roundهای بعد بهتر عمل کنند.

## 8. utility سطح سیستم

چون مسئله تشخیص نفوذ reward تیمی واقعی مثل Dec-POMDP مقاله اصلی ندارد، در کد یک utility ترکیبی ساخته شده است:

$$
U_{\mathrm{sys}} =
\frac{
\beta_r c_r
+\beta_f c_f
+\beta_a c_a
+\beta_{td} c_{td}
+\beta_n c_n
+\beta_c c_c
}{
\beta_r+\beta_f+\beta_a+\beta_{td}+\beta_n+\beta_c
}
$$

که در آن $c_c$ کارایی ارتباطی است:

$$
c_c =
1 -
\frac{|A_k|}{|C_k|}
$$

تابع loss سمت سرور:

$$
\mathcal{L}_{server}
=
\left(
Q_{\mathrm{total}}^k
-
U_{\mathrm{sys}}
\right)^2
$$

## 9. جمع‌بندی

نسخه اصلاح‌شده FMRL-LA این ویژگی‌ها را دارد:

1. در IID به FedAvg نزدیک می‌شود، چون utilityها حول 1 مرکزدهی می‌شوند.
2. در non-IID کلاینت‌های با کیفیت پایین، TD ناپایدار یا پوشش داده ضعیف اثر کمتری دارند.
3. critic هنوز وجود دارد، اما فقط سهم محدود دارد تا خروجی تصادفی اولیه باعث خراب شدن aggregation نشود.
4. وزن نهایی aggregation برابر $n_i u_i^k$ است، پس تعداد نمونه‌ها مثل FedAvg حفظ می‌شود.
5. client selection فقط کلاینت‌هایی را حذف می‌کند که هم مطلقا ضعیف باشند و هم نسبت به بقیه همان round ضعیف‌تر باشند.
