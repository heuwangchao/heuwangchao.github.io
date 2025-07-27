---
title: Hello World

---

**<u>论文</u>**：https://arxiv.org/pdf/2006.11239

**<u>代码</u>**：

- 原作tensorflow: https://github.com/hojonathanho/diffusion
- pytorch: https://github.com/lucidrains/denoising-diffusion-pytorch

## 背景

<u>**解决了什么问题？**</u>

DDPM(Denoising Diffusion Probabilistic Model)也被称为**扩散模型**(Diffusion Model)，主要用于从噪声（标准正态分布）生成高质量的图片。

**<u>总体结构</u>**

DDPM主要包含**前向过程**(forward process)和**反向过程**(reverse process)，无论前向过程还是反向过程都是一个参数化的马尔可夫链(Markov chain)，如图所示：

<img src="D:\MyBlog\source\images/DDPM-1.png" alt="1753326991141" style="zoom:40%;" />

- **<u>前向过程</u>**：从$$x_0$$逐步添加噪声到$$\boldsymbol{x}_T$$的过程，其中$$\boldsymbol{x}_0$$为原始图像，$$\boldsymbol{x}_T$$为高斯白噪声
- **<u>反向过程</u>**：将$$\boldsymbol{x}_T$$逐步去噪，还原出原始图像$$\boldsymbol{x}_0$$的过程

## 前期知识储备

**<u>正态分布及其性质</u>**

假设随机变量$$\mathbf X$$满足均值$$\mu$$和方差$$\sigma$$的正态(高斯)分布，则其概率密度函数可以表示为：
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$
此时，变量$$\mathbf X$$被称为正太随机变量，记作$\mathbf X \sim  \mathcal{N}(\mu, \sigma^2)$。其中，均值$$\mu$$也被称为位置参数（对称轴），$$\sigma$$被称为尺度参数（$$\sigma$$越小概率密度曲线越陡峭）。当$$\mu=0, \sigma=1$$时被称为标准正态分布，记作：$\mathbf X \sim  \mathcal{N}(0, 1)$。

正态分布满足以下特性：

- 对$$\mathbf X$$做线性变换有：$a \mathbf X + b \sim  \mathcal{N}(a\mu + b, a^2\sigma^2)$
- 独立正态分布的可加性，加和仍是正态分布：$\mathbf X_1 + \mathbf X_2 \sim  \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$

**<u>贝叶斯分布及其性质</u>**

假设$A,B$是两个独立的随机事件，且$P(B)>0$，$P(A|B)$表示在事件$B$发生的情况下事件$A$发生的概率，此时有：
$$
P(A \mid B) = \frac{P(A)P(B \mid A)}{P(B)}
$$
该公式也被称为**贝叶斯公式**，其中$P(A)$也被称为先验概率(prior)，$P(A \mid B)$为后验概率(posterior)，$P(B|A)$表示$B$事件对$A$事件的证据强度(likelihood)。该公式可以由下面的文氏图(Venn diagram)简单推导：

<img src="D:\MyBlog\source\images/DDPM-2.png" alt="1753341115986" style="zoom:67%;" />

在贝叶斯公式的基础上，可以引入其他事件$C$，在已知$C$事件存在的情况下，贝叶斯公式可以表示为：
$$
P(A \mid B, C) = \frac{P(B \mid A, C) P(A \mid C)}{P(B \mid C)}
$$
**<u>马尔科夫链(Markov Chain)</u>**

**<u>定义</u>**：状态空间中从一个状态到另一个状态的转换具备**“无记忆性“**，即：<u>下一状态的概率分布只能由当前状态决定，在时间序列中与它前面事件无关</u>。也可以解释为：过去所有的信息都已经被保存到了现在的状态，基于现在就可以预测未来。

> The future is independent of the past given the present
> 未来独立于过去，只基于当下。

## DDPM算法原理

### 前向过程

前向过程就是向原始图像$$\boldsymbol{x}_0$$中逐渐加入高斯白噪声使其变模糊的过程。加载过程中任意时刻的图像$$\boldsymbol{x}_t$$只与前一时刻的图像$$\boldsymbol{x}_{t-1}$$有关，该过程可以视为马尔可夫过程。加噪过程中相邻时刻关系如下：
$$
\boldsymbol{x}_t = \sqrt{\beta_t} \times \boldsymbol{\epsilon}_t + \sqrt{1 - \beta_t} \times \boldsymbol{x}_{t-1}
$$
其中，$\boldsymbol{\epsilon_t} \sim  \mathcal{N}(0, 1)$，$\beta$满足$0 < \beta_1 < \beta_2 < \beta_3 < \beta_{t-1} < \beta_t < 1$，可以看到随着迭代步数增加，$\beta$逐渐趋近于1，扩散是加速进行的。

为了简化推导，令$\alpha_t=1 - \beta_t$得：
$$
\boldsymbol{x}_t = \sqrt{1 - \alpha_t} \times \boldsymbol{\epsilon}_t + \sqrt{\alpha_t} \times \boldsymbol{x}_{t-1}
$$
已知$$\boldsymbol{x}_0$$时刻的图像，如何直接得到$$\boldsymbol{x}_t$$时刻图像的表达式呢？**<u>推导过程</u>**如下：
$$
\boldsymbol{x}_{t-1}\rightarrow \boldsymbol{x}_t:\quad \boldsymbol{x}_t = \sqrt{1 - \alpha_t} \times \boldsymbol{\epsilon}_t + \sqrt{\alpha_t} \times \boldsymbol{x}_{t-1}
$$

$$
\boldsymbol{x}_{t-2}\rightarrow \boldsymbol{x}_{t-1}:\quad \boldsymbol{x}_{t-1} = \sqrt{1 - \alpha_{t-1}} \times \boldsymbol{\epsilon}_{t-1} + \sqrt{\alpha_{t-1}} \times \boldsymbol{x}_{t-2}
$$

带入消除$\boldsymbol{x}_{t-1}$可知，$\boldsymbol{x}_{t-2}\rightarrow \boldsymbol{x}_{t}$直接跳步可以通过下面的表达式进行：
$$
\boldsymbol{x}_t = \sqrt{\alpha_t(1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-1} + \sqrt{1 - \alpha_t} \times \boldsymbol{\epsilon}_t + \sqrt{\alpha_t \alpha_{t-1}} \times \boldsymbol{x}_{t-2}
$$
由于 $\boldsymbol{\epsilon}_{t-1},\boldsymbol{\epsilon}_{t}\sim N(0, 1)$，故其概率密度分布曲线满足正态分布的线性变换和可加性，带入可得：
$$
\sqrt{\alpha_t(1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-1} + \sqrt{1 - \alpha_t} \times \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \alpha_t - \alpha_t \alpha_{t-1}) + \mathcal{N}(0, 1 - \alpha_t) = \mathcal{N}(0, 1 - \alpha_t \alpha_{t-1})
$$
采用重参数化，令$\boldsymbol{\epsilon} \sim \mathcal{N}(0, 1 - \alpha_t \alpha_{t-1})$，代入可得：
$$
\boldsymbol{x}_t = \sqrt{1 - \alpha_t \alpha_{t-1}} \, \boldsymbol{\epsilon} + \sqrt{\alpha_t \alpha_{t-1}} \, \boldsymbol{x}_{t-2}
$$
同理，带入$\boldsymbol{x}_{t-3}$的表达式可得$\boldsymbol{x}_{t-3}\rightarrow \boldsymbol{x}_t$可以表示为：
$$
\boldsymbol{x}_t = \sqrt{1 - \alpha_t \alpha_{t-1} \alpha_{t-2}} \times \boldsymbol{\epsilon} + \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}} \times \boldsymbol{x}_{t-3},\quad \mathcal{N}(0, 1 - \alpha_t \alpha_{t-1} \alpha_{t-2})
$$
因此，使用数学归纳法，可以得到$\boldsymbol{x}_{0}\rightarrow \boldsymbol{x}_t$的表达式为：
$$
\boldsymbol{x}_t = \sqrt{1 - \alpha_t \alpha_{t-1} \alpha_{t-2} \alpha_{t-3} \dots \alpha_2 \alpha_1} \times \boldsymbol{\epsilon} + \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2} \alpha_{t-3} \dots \alpha_2 \alpha_1} \times \boldsymbol{x}_0
$$
令   $\bar{\alpha}_t = \alpha_t \alpha_{t-1} \alpha_{t-2} \alpha_{t-3} \dots \alpha_2 \alpha_1$，则上式可以简化为：
$$
\boldsymbol{x}_t = \sqrt{1 - \bar{\alpha}_t} \times \boldsymbol{\epsilon} + \sqrt{\bar{\alpha}_t} \times \boldsymbol{x}_0,\quad \bar{\alpha}_t = \alpha_t \alpha_{t-1} \alpha_{t-2} \alpha_{t-3} \dots \alpha_2 \alpha_1
$$

### 反向过程

反向去噪的过程就是在已知噪声图片$\boldsymbol{x}_T$，通过逐步去噪恢复出原图$\boldsymbol{x}_0$的过程。由于从$\boldsymbol{x}_{t-1}\rightarrow \boldsymbol{x}_t$加入的是随机噪声，所以从$\boldsymbol{x}_{t}\rightarrow \boldsymbol{x}_{t-1}$也是一个随机过程，反向过程的目标是在已知$\boldsymbol{x}_t$的情况下求解前一时刻$\boldsymbol{x}_{t-1}$的图像，即求解 $P(x_{t-1}|x_t)$。

由贝叶斯公式可知，在相同的$\boldsymbol{x}_0$下，给定$\boldsymbol{x}_t$其前一时刻$\boldsymbol{x}_{t-1}$的概率可以表示为：
$$
P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{P(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0)}{P(\boldsymbol{x}_t \mid \boldsymbol{x}_0)}
$$
由前向过程可知：
$$
\begin{align}
\boldsymbol{x}_t = \sqrt{1 - \alpha_t} \times \boldsymbol{\epsilon}_t + \sqrt{\alpha_t} \times \boldsymbol{x}_{t-1} \quad &\Rightarrow \quad P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) \sim\mathcal{N}\left( \sqrt{\alpha_t} \boldsymbol{x}_{t-1}, 1 - \alpha_t \right) \\

\boldsymbol{x}_t = \sqrt{1 - \bar{\alpha}_t} \times \boldsymbol{\epsilon} + \sqrt{\bar{\alpha}_t} \times \boldsymbol{x}_0 \quad  &\Rightarrow \quad P(\boldsymbol{x}_{t} \mid \boldsymbol{x}_0) \sim\mathcal{N}\left( \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0, 1 - \bar{\alpha}_t \right) \\
\end{align}
$$
同理，$P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0) \sim\mathcal{N}\left( \sqrt{\bar{\alpha}_{t-}} \boldsymbol{x}_0, 1 - \bar{\alpha}_{t-1} \right)$，替换为其对应的概率密度函数代入$P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) $的表达式可化简得：
$$
P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) \sim \mathcal{N}\left( \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} \boldsymbol{x}_0, \left( \frac{\sqrt{1 - \alpha_t} \sqrt{1 - \bar{\alpha}_{t-1}}}{\sqrt{1 - \bar{\alpha}_t}} \right)^2 \right)
$$
反向过程就是需要得到$\boldsymbol{x}_{t}\rightarrow \boldsymbol{x}_{t-1}$的关系，最后从$\boldsymbol{x}_T$不断迭代直到恢复出原图$\boldsymbol{x}_0$，上面式子中$\boldsymbol{x}_0$是未知的，但是在前向过程中有：
$$
\boldsymbol{x}_t = \sqrt{1 - \bar{\alpha}_t} \times \boldsymbol{\epsilon} + \sqrt{\bar{\alpha}_t} \times \boldsymbol{x}_0 \Rightarrow \boldsymbol{x}_0 = \frac{\boldsymbol{x}_t - \sqrt{1 - \bar{\alpha}_t} \times \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}
$$
将$\boldsymbol{x}_0$带入$P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$可以得到不含未知数的迭代表达式：
$$
P(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) \sim \mathcal{N}\left( 
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \boldsymbol{x}_t + \frac{\sqrt{\alpha_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} \times \frac{\boldsymbol{x}_t - \sqrt{1 - \bar{\alpha}_t} \times \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}, 
\left( \sqrt{\frac{\beta_t(1 - \alpha_{t-1})}{1 - \bar{\alpha}_t}} \right)^2 
\right)
$$
从前向过程可知，任意一个$\boldsymbol{x}_t$都可以原图$\boldsymbol{x}_0$直接加噪得到，而只要知道$\boldsymbol{x}_{0}\rightarrow \boldsymbol{x}_t$加入的噪声$\boldsymbol{\epsilon}$，就能得到$\boldsymbol{x}_t$的前一时刻$\boldsymbol{x}_{t-1}$的概率分布。因此，反向过程的目标变为：

> **训练一个神经网络模型来预测图像$\boldsymbol{x}_t$相对于某个原图$\boldsymbol{x}_0$加入的噪声$\boldsymbol{\epsilon}$，然后通过$\boldsymbol{\epsilon}$来预测前一时刻的图像的概率分布，最后通过随机采样得到前一时刻的图像$\boldsymbol{x}_{t-1}$。继续将$\boldsymbol{x}_{t-1}$输入模型中得到噪声$\boldsymbol{\epsilon}$，反复迭代就可以得到原图$\boldsymbol{x}_0$。**

迭代的起点$\boldsymbol{x}_T$可以由标准正态分布随机采样得到。

















