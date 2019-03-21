[转载来自EternalFeather大佬]细数生成对抗网络和自然语言处理的那些恩怨情仇
===
# 文本生成的基础模型（Introduction）
在众多NLP的task中，**文本生成(Text Generation)** 是一种结合了机器学习和自然语言处理等技术层面而衍生出来的应用。他的诞生在一定程度上诠释了当今这些技术的发展水平。例如前不久出现在网络上的Drumpf推文产生器。

![](https://i.imgur.com/ir8H6tc.png)

## 文本生成分类
文本生成的应用大体上可以分为以下几类：
- 机器翻译（machine translation）
- 问答系统（Question-Answering System）
- 对话生成（dialogue generation）

## 核心模型介绍
使用在文本生成的模型不唯一，比较常用的是循环神经网络（RNN），使用的Cell包括了LSTM，GRU等。作为序列生成模型的核心框架，encode-decode framework开始出现在人们的视线范围之内，并在短时间内成为了广为人知的核心模型框架。

![](https://i.imgur.com/2E2JNNG.png)

类似于编码的原理，通过encoder将一个句子序列x（可以理解为utf-8编码）编码成为数字向量（也可以是矩阵或者Tensor，可以理解为unicode）。之后decoder利用这个编码后的context information将信息还原成为另一个对应的序列y（可以理解为big5编码）。虽然编码的过程是计算机通过改变初始化的超参数改变特征在隐层的映射机理，因此难以探究其具体含义，但是这也正应证了特征的分布是真实存在的（可以通过控制变量的方式探究每一个参数的具体含义）。

对于文字生成的任务，首先最重要的一点就是如何表示文字。我们知道图像是由像素（pixel）组成的，而像素在计算机中是被编码成数字的形式，因此我们可见的颜色图案都是由0和1这些数字构成的。虽然文字本身也被赋予了各自的编码（如utf-8等），但是这些编码却很难被应用在神经网络的参数传递中。通常的做法是建立一个词库字典（vocabulary）涵盖所有可能出现的字，而这些对应的字都会表示成one-hot vector的形式（不是唯一）作为模型的输入。每一次模型都会通过当前输入的字去预测下一个时间点会出现的字最可能是什么，从而一步一步得到完整的句子序列。

而我们往往会选择使用Softmax表示所有最终的结果，通过概率表示每一个字对应的概率值来选取最有可能的候选字作为输出。就这样，模型每次以一组“样本（sample：x）-标签（label：y）”作为训练的资料，通过循环神经网络主导的encoder和decoder将x转换成y。

# 传统GANs的数学定义和问题（Related Works）
GANs顾名思义是由两个模型构成的：1、生成器（Generator） 2、判别器（Discriminator）。其中生成器的目标是产生最接近真实样本的分布，从而骗过判别器的判断。而判别器相反就是尽可能准确地区分出真实样本和生成样本。

## 损失函数
在GANs中对于判别器而言，想要达到相应的目标就应该尽可能把真实样本当成正例，而把生成样本当成反例：

![](https://i.imgur.com/b18Xa3Q.png)

而对于生成器而言其损失函数有两种：

![](https://i.imgur.com/ln5gMA0.png)和改进之后的![](https://i.imgur.com/YY1COzI.png)

## 存在的问题
相信喜欢研究GANs的你也经常看到一些这样的说法“判别器训练的越好，生成器的模型就会越不稳定越南收敛”。传统的文字故难表达其中蕴含的奥秘，因此我们来通过数学方式了解一下。（推荐文章：令人拍案叫绝的Wasserstein GAN）

首先通过![](https://i.imgur.com/ln5gMA0.png)可以知道，当判别器达到最优化状态（我们想要的理想收敛状态，处于saddle point梯度为0）时，我们可以得到：

![](https://i.imgur.com/tQM7NGV.png)

我们对其求梯度（gradient）得到：

![](https://i.imgur.com/8DbmSgz.png)

化简之后得到：

![](https://i.imgur.com/53B6aGY.png)

这个公式中的 **P** 可以理解为通过x序列得到的y属于真实样本还是生成样本的概率值。当两者相等，则结果是0.5证明模型无法给出判断，因此为欠拟合状态。

如果此时我们将判别器训练的太好，那么生成器的loss就会降不下去（来自loss vanishing的问题）。我们给![](https://i.imgur.com/ln5gMA0.png)加上一项与生成器无关的项（虽然改变了loss，但是却不改变优化生成器的梯度，同时也是判别器损失函数的反），得到：

![](https://i.imgur.com/u4eMnkB.png)

化简之后得到：

![](https://i.imgur.com/xvXF6no.png)

这个式子涉及到的物理意义被称为KL散度（Kullback-Leibler divergence）和JS散度（Jensen-Shannon divergence）两个相似度衡量指标。具体的数学表达式如下：

- KL divergence

![](https://i.imgur.com/zHItvfh.png)

- JS divergence

![](https://i.imgur.com/KPwBQhJ.png)

将散度的公式代入增加了辅助项后的生成器优化函数，我们可以得到：

![](https://i.imgur.com/KCea4mp.png)

现在我们明确了一个道理：那就是在GANs训练过程中，当判别器到达了一个最佳状态的时候，生成器的目标就相当于最小化真实分布和生成分布之间的JS散度。**我们越是训练判别器，生成器的损失函数（loss function）就会越接近JS散度的值**。

那么这又会引起什么问题呢？我们知道越是减小误差，JS散度就会越小，生成分布就会被带向真实分布。但是假如两个分布本身就存在维度差或者说两个分布本身就没有重合的部分呢？这个问题比较抽象，下面通过一些例子来介绍：

首先介绍一下什么叫做两个分布没有重合的部分呢？维度差有是什么呢？怎么样的分布之间才算得上具有维度差呢？Wasserstein GAN告诉我们当![](https://i.imgur.com/u2XkwvK.png)和![](https://i.imgur.com/NJKd0XN.png)的支撑集（support）是高维空间中的低维流形（manifold）时，两者重合的部分的测度（measure）为0。

解释几个专有名词：
- 支撑集（Support）：函数的非零部分的子集，例如ReLU的支撑集就是![](https://i.imgur.com/8fTdAaI.png)，一个概率分布的支撑集就是他所有概率密度非零值得集合。
- 流形（manifold）：高维空间中的曲线、曲面的概念延伸。当两个分布的本质维度（intrinsic dimension）不同时，他们的自由度（二维的自由度包括长和宽）就会不同。
- 测度（measure）：高维空间中的长度、面积、体积等测量指标的概念延伸。

**打个比方**，如果我们要比较一个正方体和一个三角形的相似程度，我们就会陷入一个问题。正方体的维度是3，而三角形的维度是2，相差了“高”这一维度特征。如果把像和不像作为分类问题来看，前者比后者多出了整整一个特征，在三维的分类空间（往往会取最大的维度空间）中他们的重合部分仅仅是一个平面。这也就以为这如果三角形的底边和高都等于正方体的边长，但是我们仍然无法通过某种方法让三角形“逼近”（可以理解为变换）正方体。

在文字生成的任务中，由于GANs的生成器采用从一些杂讯信号中随机抽样组合，再通过类似DNN的全连接层进行维度扩展，从而达到能够在维度空间匹配最终结果的一个输出信号。但是不要忽略了一点，这些生成信号的本质（特征集合）还是停留在原先的杂讯信号的基础上没有改变。换句话说，即使进行了维度提升，但若是全连接层的神经元不足够强大，就没有办法还原所有的特征从而让两个比较的结果处于同一个维度空间中。

其次，从离散的角度分析，对于一个x输入信号可能会有以下几种可能：
- ![](https://i.imgur.com/laWZZ2F.png)
- ![](https://i.imgur.com/ho6eo5j.png)
- ![](https://i.imgur.com/6qEmE69.png)
- ![](https://i.imgur.com/smaw7Mr.png)

第一种情况不存在两个分布的距离这样的说法（因为两个分布本身就不存在），因此对JS散度的计算没有帮助。第二种情况和第三种情况，代入公式可以得到
![](https://i.imgur.com/Niw4Dfp.png)
和
![](https://i.imgur.com/SkFNT8j.png)
而唯有第四种情况的两个分布才有可能存在交集从而达到减小JS散度的目的。

说完了JS散度在分布呈现不同状态时存在的问题，那么怎么样联想到GANs中呢？可以试想一下，如果现在的![](https://i.imgur.com/GQp7abc.png)就是![](https://i.imgur.com/u2XkwvK.png)，而![](https://i.imgur.com/9Tmbfhd.png)就是![](https://i.imgur.com/NJKd0XN.png)。那么当判别器训练到最佳情况的时候，![](https://i.imgur.com/NJKd0XN.png)就会为0（相当于第二种情况），此时JS散度就是一个常数![](https://i.imgur.com/2J1WgZu.png)，这也就意味着生成器的损失函数对于他本身的超参数的偏微分结果为0——没有梯度！这样一来每一次的对抗学习都会让生成器陷入一个迷茫的状态，也就是梯度的消失（gradient vanishing）。

总结一下，生成器的第一种损失函数是因为在判别器接近最优状态时，生成器的误差就相当于![](https://i.imgur.com/u2XkwvK.png)和![](https://i.imgur.com/NJKd0XN.png)的JS散度。而所有的情况都指明了，在这种最优条件下，无论两个分布怎么变换，都会让梯度变成0或者无意义。这样一来**GANs训练不稳定的原因也就清楚了**。判别器训练得太好，生成器的梯度就会消失，判别器训练得不好，生成器的梯度就会不准确，四处乱跑。只有当判别器训练刚刚好火候的时候，GANs才能平稳地进行对抗训练。

那么既然这种生成器的损失函数存在这样的问题，就以为着我们无法随心所欲地让模型自己朝着收敛的目标前进，势必会对训练模型造成影响。那么如果采用改进后的第二种损失函数呢？

在论述第一个损失函数的时候我们已经证明了：

![](https://i.imgur.com/ZIYSgJI.png)

我们先对KL散度进行一些调整：

![](https://i.imgur.com/cyUf16J.png)

由这个KL散度的启发，可以得到新的损失函数表达式如下：

![](https://i.imgur.com/Gkuv6Xn.png)

由于目标是优化生成器，因此计算梯度的项只保留对生成器可微分的项即可。舍弃梯度为0的项之后得到：

![](https://i.imgur.com/HWSl6cC.png)

相比推到这里，也有很多人能看出这个式子本身存在的一些问题了吧。首先，这个损失函数的目的是要同时最小化生成分布和真实分布的KL散度，又要最大化两个分布的JS散度。但是我们知道这两个散度的物理意义是朝着同一个方向发展的（两者正相关）。所以这样的优化目的显然存在矛盾的地方，从而让我们的生成器训练不稳定。其次，KL散度本身的量化也存在一些问题，原因就是他不是一个对称的衡量单位。

### KL散度的问题
比较一下![](https://i.imgur.com/NXU4Tem.png)和![](https://i.imgur.com/k9Of88Y.png)可以发现：

- 当![](https://i.imgur.com/A3LmpuI.png)而![](https://i.imgur.com/znxr8sJ.png)时，![](https://i.imgur.com/V32hqvs.png)，对![](https://i.imgur.com/NXU4Tem.png)的贡献趋近于0

- 当![](https://i.imgur.com/CSPLSng.png)而![](https://i.imgur.com/TylgMGA.png)时，![](https://i.imgur.com/F5ZQHrU.png)，对![](https://i.imgur.com/k9Of88Y.png)的贡献趋近于无穷大。

这些差异意味着什么呢？我们可以这样理解，KL散度对于以上两种方式定义的不同，也就侧面反映了生成器在面对判别器两种不同情况时候给出的判断。首先第一种情况下，错误来自于 **“生成器没有办法生成真实的样本”** ，这个时候的惩罚比较小（KL散度趋于0）。而第二种错误对应的错误则来自于 **“生成器无法生成真实样本（当生成器的模型达到最佳状态的时候，判别器给出的分数仍然很低）”** ，这个时候的惩罚是非常大的（KL散度趋于无穷大）。第一种误差让生成模型缺乏多样性，因为模型只是为了尽可能模仿真实样本进行输出。而第二种误差则让生成模型缺乏准确性，因为模型一旦尝试一些不一样的方法便会受到大量的惩罚。这样的权衡之下，模型和人一样是具有惰性的，在一切条件相近的前提下，模型宁可选择那些惩罚小的苟且方式而放弃挺而走险尝试不同的东西。这样一来生成器模型往往会选择接受第一种错误，而让生成的句子千篇一律，这在特征角度被定义为塌陷模式（collapse mode）。

## 总结
经过了上述的介绍和数学推倒，我们可以看到GANs的模型所具备的一些问题和缺陷。其中包括了在最优判别器的时候，生成器往往会出现梯度消失的问题；以及在KL散度的物理意义上来说GANs的优化目标存在一定的矛盾性以及盲目性（collapse mode）。

## 解决方案
关于以上的问题具体可以参考Wasserstein GAN一文，介绍了如何利用新的距离衡量两个分布的关系，从而解决KL散度和JS散度的一些问题。

### Wasserstein GAN，那些年我们一起填完的坑
我们知道最开始生成器的损失函数面临KL散度和JS散度所带来的问题，而改进之后却又有着矛盾的优化策略。这样对一个模型而言是十分致命的，因此必须有人来填补这个坑，才能让GANs真正进入人们的实现成为一门科学。

Wasserstein GAN对GAN的改进是从替换KL散度进行的，他开创了一种全新的用来计算散度（divergence）的指标——**Earth Mover (EM) distance**。Wesserstein GAN的散度计算将两个分布看作是两堆土，而目的就是计算将第一堆土堆成第二堆的形状需要搬运的总距离。如下图所示：

![](https://i.imgur.com/BIthGKR.png)

左边的那个分布想要达到右边的状态，就必须移动制高点左边的那部分，而距离和量的考量就成为了当前的关键指标。

那么这样的评估方式又有什么好处呢？为什么说他能够解决KL散度和JS散度所带来的问题呢？首先我们先要明确一下问题的本质，从之前的讨论可以看出，KL散度和JS散度在当判别器训练到最佳状态的时候存在的问题就是两个分布的重合部分可忽略或者不存在。这个时候我们得到的生成器的损失函数（是一个常数![](https://i.imgur.com/FDmECAu.png)），除非能够让两个分布重合，这个时候才会突然变化成为最优解。这样一来我们的优化就缺少了一个所谓的“演化”过程，换句话说就是我们无法通过损失函数的改变来描述从好到不好的渐变过程，这样对于模型的训练是不利的。而利用Earth Mover distance我们就可以量化每一次更新之后的差异，从而还原这个重要的过程。具体情况如下图：

![](https://i.imgur.com/mhx2hDy.png)

Wasserstein散度能够通过通过不同的距离 **d** 来反应每一个时刻样本的变化情况。生成器每一次的output能否更加接近真实样本的那些“进步”都能被抓到并传回，这样就保证了对于离散数据的对抗能够持续下去。

# 传统GANs在NLP上遇到的问题（Motivation）
说完了传统的GANs模型本身的问题之后，我们需要进一步了解为什么GANs用在NLP的领域又是另外一片天地。曾经就有人问过，GANs能否被使用在文字生成领域，在Generative Adversarial Networks for Text一文（可见参考文献部分）中GANs的作者Ian Goodfellow就层明确给出了回答。让人心寒的是，作者给出的答案是否认的成分居多，并且还提出了一些理由：

![](https://i.imgur.com/1oVPCoq.png)

从文中的一些表达例如：“there is no way...”和“no one really knows...”等表达来看，当时的作者对于这一个问题保持着疑问和反对的态度是多么的强烈。

那么首先让我们来看一下这段回文中提到的一个问题，那就是“Discrete numbers（离散数字）”。保留作者的观点可以看出，GANs当初被设计的初衷是定义在真实数据的情况下，通过生成器的到的合成数据都是连续的分布。因此，**只有在数据连续的情况下，才有可能通过略微改变合成的数据，而如果是离散的数据，就无法通过改变这些数据来得到新的结果**。

这个表达听起来很难get到精髓是什么，因此作者给出了一个形象的例子。如果一张图片，其像素点是1.0，那么改变了这个值成为1.0001之后，就会成为其他的值（在计算机的表现方式中，不同的像素值代表了输出不同的颜色）。而如果输入的是一个文字“penguin”，那么接下来我们就无法用同样的方式进行输出的转换，因为**没有人知道“penguin+0.001”之后得到的新的文字会是什么**。因为对于自然语言而言，每一个文字都是离散的存在，包括“单词”，“音节”，“字母”等等。尽管有的研究也在尝试将离散的文字之间建立起连续的转换机制，例如：考虑化学物质的性质，假设我们得到了化学式A，B，C并且知道了这些物质的特性，我们就能得到三者的转换关系（D=A-B+C），同样的道理能不能用在文字上呢？我们是否可能设计出类似queen-woman+man=king这样的结果呢？尽管听起来make sense，然而当前的一些数据表示（Data representation）和嵌入方式（Word embedding）都无法完全解决这个问题（即让文字具有连续性和数学意义）。

## 困扰GANs在NLP上应用的三大强敌
即便是短短的发言，我们依然从中看出了GANs在文字生成领域乃至整个NLP应用层面存在着巨大的挑战。然后随着后续研究者们的不断探索，一个振奋人心的消息圣诞生了：那就是我们终于又发现了新的问题！！！下面就让我们来认识一下阻碍GANs在自然语言科学领域发展的“三巨头”：

- 原始的GANs设计初衷是为了应用在连续的空间数据，而在离散空间中并不能工作。那么什么是离散数据呢？文本就是一个典型的离散数据，所谓的离散，并不是逐字逐词叫做离散，而是分布的不连续特性。这样的解释是不是感觉又绕了回来了？没错，想要了解离散的概念，最简单的就是去理解什么是连续。在计算机内部的资料表示形式中，往往是通过浮点数进行的，而浮点数就是连续的数值。一些连续的分布就是指数据本身的意义就是通过数字表示的：例如图像的基本单位是像素点（pixel），并且**这些pixel本身就是一些数字**，换句话说数字就是pixel的全部。图像通过矩阵（二维）的形式将这些数字组合起来，就形成了一张图片。因此改变图片的方法无非是改变其中的数字即可。例如左上角的哪一个pixel太红了，就把红色的那个色阶的值减小一些，而输出的数值无论是多少都会有一个对应的颜色来与之对应。

而文字不同，一个句子是被定义为了一个词库的one-hot vector形式（不唯一，但是本质相同），通过类似table的形式索引字典里的每一个字。而引发问题的根本原因就是模型输出的结果都是一个大小和one-hot vector相同的概率分布。而这个分布若想要最终反应成为真实的字，就必须进过一个过程叫做“sampling”，而问题就是出在“sampling”上。我们往往通过softmax得到了一个分布，然后sample出其中最大的那一个维度的值对应的字作为输出。

![](https://i.imgur.com/Mj1HaTX.png)

从上图的例子来看，通常模型的优化方法是通过一些优化函数改变参数的分布，从而观察输出有没有好一点（loss小一些）。但是判别器若是得到sampling之后的结果呢？我们经过参数调整得到的结果是优化后的直接表现。例如：假如倒数第二个字对应最大的0.85并不是我们想要的那个输出，我们就会减小他的值假设成为了0.65，而第三项的0.12才是我们想要的那个字，因此我们扩大他的权重成为0.32，然而经过了sampling之后，呈现在判别器面前的仍然是倒数第二个字。这时候判别器给出的答案一样还是很糟糕，这样一来生成器就会失去训练的方向（可以理解为loss飘忽不定，有可能越优化loss越高）。

![](https://i.imgur.com/RCjbz2S.png)

也有人因此提出质疑： **既然sampling过程会造成这样的问题，那么不要sampling直接把softmax的结果丢进判别器不就好了？** 不得不说这个确实是一个可行的方法，但是却遭到了模型本身的拒绝。为什么说是“拒绝”呢？其实判别器的训练初衷是为了分辨生成样本和真实样本的差距，那么这个时候如果两个样本的表达方式存在差异（一个是one-hot vector，一个是softmax；一个是one-hot encoder，一个是probability distribution）。这个时候就会出现之前说过的两者的特征维度不重合的现象。在这里这个现象可以理解为模型在经过神经元转换后的latent space中往往存在许多的desert hole（AI技术讲座精选中的术语）。在training的时候，模型往往会因为两者的分布出现差异而故意现在那些不重叠的区域不肯跳出。从而导致模型判别的根基出现问题（模型学习到的判断方式可能就是看输出的分布是不是one-hot vector，如果不是直接over；而不是努力去比较softmax和真实的one-hot vector究竟有多像）。

如下图所示：
Comparison of continuous & discrete distributions. (x : θ, y : loss)

![](https://i.imgur.com/DSnxZxQ.png)

左边的连续分布我们总能找到一个loss下降的梯度，从而移动x轴（生成器的超参数）来达到更小的loss，这个过程被称为梯度下降优化（gradient descent）。而相比于离散的数据分布可以看到，独立点之间的优化是无法得到梯度的，因此我们往往会遇到上述提到的问题。

- 在生成文字的过程中，大多数模型都是基于循环神经网络（Recurrent neural network）模型设计的，这种时序性的模型在隐层表示（latent codes）计算过程中，error往往会随着序列不断累积直到最后。这种方式的另一个根本原因和最大似然估计（Maximum log-likelihood estimate）有关，也被称之为 **“Exposure bias”**。

![](https://i.imgur.com/ZKVx8U1.png)

在training过程中，我们每一个时刻输入的都会是真实资料对应的值，因此即使出现错误，下一个时刻的输入仍然是正确的。但是在testing的过程中，由于缺乏真实资料的辅助，如果上一个时刻的结果出现错误，我们就会用错误的记过进行下一个时刻的预测，造成错误的累加。

- 再生成文本序列的过程中，传统的GANs是对整个句子或是更大的序列单位进行评估的。这一点也是非常直观的一个反应，因为我们无法在一个人说完整句话之前就断章取义去推测这句话的好坏。但是在时序生成的过程中，因为loss在句子生成过程中的每一个字产生的时间点都会被计算，因此这个时候的error如何得到就成了一个新的问题。

![](https://i.imgur.com/0cO4Yth.png)

最直观的想法就是将整句的分数当做每一个部分单独的分数，但是这样的方式会让句子的成分承受相同的代价，不符合我们直观的感觉。因为造成句子不好的原因可能往往来自单独的一些部分，而不是全部。

# 传承的智慧（Methods）
见识到了GANs在文本生成领域的困难重重，是时候来一波强心剂了。尽管来自模型自身和外部环境的压力十分巨大，但是仍然有不少学者费尽心机探索学术的奥秘，挖掘最古老的智慧。

## Generating Text via Adversarial Training
- 论文链接：https://zhegan27.github.io/Papers/textGAN_nips2016_workshop.pdf
- Key Points：这篇论文是比较早期利用GANs进行文本生成任务的尝试，通过循环神经网络（RNN+LSTM）作为生成器，采用光滑近似（smooth approximation）的优化理论来逼近真实样本分布。
- Graph & Algorithm：

![](https://i.imgur.com/mIJ2pRu.png)

### Feature Matching
**思路**：文章提出的TextGAN模型利用了不同于以往的GANs所使用的优化函数，这种优化的思路基于特征匹配原理（feature matching）。

![](https://i.imgur.com/fkGHe16.png)

通过计算生成器输出和真实分布在隐层上特征的差异，从而比较两者的差异，并最终最小化这样的差异。

文章还采用了一个特殊的技巧，就是在判别器进行pre-training的过程中，利用真实的句子难免会让判别器的判断局限在你和那些表达。而如果加入一些十分类似却又略微不同的句子结构，势必会让模型的判断更为灵活。而作者采用了交换真实句子中两个词的位置从而得到新的句子，并利用这些新的样本来帮助判别器进行训练。从量化角度看，模型接触的信息量是相同的，能不能进行判断完全取决于模型对于文字内部关联的把握（如果利用卷积+同样的池化模式得到的结果会相近，采用序列生成模型可能会大不相同）。

论文实验部分提到了在对抗过程中想要让训练稳定进行，生成器的训练频率将会是判别器的5倍。原因是判别器采用了CNN的架构，而生成器采用LSTM，这两者的收敛难度存在一定的差异。

## SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- 论文链接：https://arxiv.org/pdf/1609.05473.pdf
- Key Points：本文利用强化学习的思路，用reward代替gradient来进行前馈训练，避免了GANs无法在离散数据传递梯度的问题。同时利用蒙特卡罗检索（Monte Carlo Search）的方法得到每一个时间点的误差信息。
- Graph & Algorithm：

![](https://i.imgur.com/X28Bdg4.png)

![](https://i.imgur.com/Bi1mLgJ.png)

### Policy Gradient
**思路**：如图所示，左边的部分是判别器（核心是CNN）的优化部分，利用真实资料和生成资料来训练判别器。右边的部分则是生成器（核心是LSTM）的优化部分，利用Monte Carlo的方式得到每一个时刻的reward，并借助policy gradient的强化学习方式进行模型优化更新。

强化学习更新的四个重要的因素包括：状态（state）、行为（action）、策略（policy）和奖励（reward）。在文字生成领域，state就相当于该时刻之前已经存在的序列（之前时刻输出的所有结果组成的sequence），action是当下需要选择生成的句子成分（当前解码后的字或词，也叫token），policy为GANs的生成器模型网络，reward则是GANs的判别器网络输出的概率值（也叫做scalar）。

### Monte Carlo Search
论文在得到部分序列（intermediate sequences）的奖励上面采用了蒙特卡罗（Monte Carlo）的方式，这样的策略通过多次平行抽样来共同决定当前的序列的可靠性。

![](https://i.imgur.com/nQsDr9Q.png)

当解码到 t 时刻的时候，模型会对后面的 T-t 个timestep采用蒙特卡罗检索出 N 条路径，这些路径经过判别器的结果会通过平均的方式作用在 t 时刻生成的那个token上。每一个序列都会经过几次深度检索得到不同的完整序列，然后这些序列的评分就综合决定了那个**固定前缀**的分数。作者给出了具体的算法如下：

![](https://i.imgur.com/DAvLkzr.png)

### Schedule Sampling
论文还参考了一种缓解exposure bias的方法。这种方法通过一个变化门槛值控制模型训练过程的每一个时刻是使用来自真实资料的值还是来自前一个时刻的预测结果，从而在训练的时候就让模型适应各种可能的错误发生。这样的经验在test的过程中就不容易因为一时的错误而让误差累积。

![](https://i.imgur.com/cv08KyT.png)

对于强化学习（reinforcement learning）的部分，文章采用了policy gradient的方法重新定义了生成器的目标函数：

![](https://i.imgur.com/4Jv1V4B.png)

求导结果如下：

![](https://i.imgur.com/r5qhNsB.png)

目标函数告诉我们生成器模型的目的是要尽可能提高生成序列得到的来自判别器的reward分数的期望，这个目标与最初GANs的初衷不矛盾。

对于判别器的部分，论文给出了和原先GANs相同的优化策略：

![](https://i.imgur.com/qR2aCQF.png)

## Adversarial Learning for Neural Dialogue Generation
- 论文链接：https://arxiv.org/pdf/1701.06547.pdf
- Key Points：这篇论文将对抗学习（adversarial learning）用在开放性对话生成（open-domain dialogue generation）领域。与SeqGAN相同的一点是，文章采用了强化学习的方式取代传统的梯度将GANs应用在离散数据中，借助奖励reward来指导生成器的训练。另外，本文还采用了teacher forcing的方式来辅助生成器的训练，能够让对抗学习的过程更加稳定。
- Graph & Algorithm：

![](https://i.imgur.com/p6nEpTF.png)

**思路**：因为是开放性对话生成，因此很自然地想到了采用seq2seq来作为生成器的核心模型；另外作者选择了hierarchical encoder而非CNN作为判别器，原因是循环神经网络能够更好地拟合序列的时序特征。但是这个理由需要建立在我们能够控制好判别器的训练幅度，因为之前讨论过过度训练或者训练不足都会让对抗过程变得不稳定。

针对如何得到每一个时刻的奖励，该论文的作者给出了两种不同的方法：Monte Carlo Search以及依靠自己训练的一个能够对部分生成序列进行reward计算的模型。前面那种方法和SeqGAN的做法相同就不在进行介绍，相比之下后者的模型能够通过使用完整的（fully）和部分的（partially）解码序列来训练一个分类模型。但是这种方法存在一个问题，那就是重复的序列会让模型overfitting在固定序列的生成模式。因为早期产生的部分序列会出现在后续的更长的序列中。怎么理解呢？假设我们有一个序列“This is a man.”，在模型训练第三个序列“This is a”的时候，模型已经学习过“This”和“This is”。也就是说this这个token的权重在训练后续的时间点时会被重复训练多次，这样以来模型对于早期的前缀序列会具有很深刻的印象，导致模型overfitting在一些固定的表达法而难以跳脱。为了克服这个问题，作者就选择了每一次更新的时候仅从所有的正序列（positive sequences）和负序列（negative sequences）中分别随机sample一个来训练判别器。实验结果表明这种方法比蒙特卡罗省时，但是却无法在收敛之后达到比蒙特卡罗更好的结果。

### Teacher Forcing
在序列生成对抗的过程中，GANs的生成器只能间接地通过判别器产生的reward来判断所产生的句子的好坏。而不是直接接触最终的真实句子（可以叫做gold-standard）获取信息。这样的方式可能存在一个隐藏的问题点，那就是一旦生成器在某个时刻产生的句子不好，判别器就会给出一个较低的reward，这个时候对于生成器而言最直观的感觉就是需要对生成序列的分布进行改变。然而因为无法接触到最终label的任何信息，因此生成器很容易迷失（只知道自己不好，但是却不知道应该如何改进）。为了解决这个问题，本文引入了teacher forcing的策略，在生成器更新的过程中，如果判别器给出的reward很小，模型就会将真实的资料作为输入指导生成器进行下一次的更新优化。就好比一个老师在学生遇到困难的时候能够帮他指引方向一样。这样一来生成器就有可能在上述的情况下仍能得到较好的结果。

## GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution
- 论文链接：https://arxiv.org/pdf/1611.04051.pdf
- Key Points：Gumbel Softmax Distribution

- Graph & Algorithm：

![](https://i.imgur.com/w04YQ7F.png)

**思路**：不同于之前的RL方法，该论文在处理离散数据的问题上采用了新的方法——Gumbel Softmax。离散数据通常是用one-hot vector表示，并可以通过sampling的方式从一些多项式（例如softmax函数）中采样得到。采样的过程：![](https://i.imgur.com/HNIeZqu.png)，其中的![](https://i.imgur.com/FMeYe1T.png)是服从Gumbel Distribution的分布函数。然而这样的函数在反向传递过程中是无法微分的，因此我们采用另一种表示方式：![](https://i.imgur.com/4aG1Sjc.png)，这样的方式通过调节温度系数![](https://i.imgur.com/5nuwM7C.png)的值我们可以让softmax的分布在形式上尽可能逼近one-hot vector的形式，从而打破KL散度识别的误区。

## Connecting Generative Adversarial Networks and Actor-Critic Methods
- 论文链接：https://arxiv.org/pdf/1610.01945.pdf
- Key Points：Summarize the correlation between GAN and Actor-Critic

- Graph & Algorithm：

![](https://i.imgur.com/We7q6hr.png)

**思路**：大多数的强化学习模型只是采用了单一的决策（policy）和价值（value）方法。而Actor-Critic则是结合了两者的一项应用于一身的模型。类似GANs的运作方式，Actor-Critic也是采用了两种方法结合的训练模式。其中actor用来决策输出的action，而critic用来对action进行价值评估。

论文提到了很多关于GANs与Actor-Critic的相同之处。actor相当于generator，都是用来为下一个时刻产生相应的输出（sentence for GANs & action for Actor-Critic）。critic相当于discriminator，都是用来评估所产生的输出所具备的价值水平。

## Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets
- 论文链接：https://arxiv.org/pdf/1703.04887.pdf
- Key Points：将SeqGAN用在机器翻译领域
- Graph & Algorithm：

![](https://i.imgur.com/YNKQOjk.png)

**思路**：论文提出了CSGAN-NMT模型，用对抗学习的方式训练模型解决机器翻译的问题。生成器用的是attention based的NMT，而判别器采用的则是CNN based的分类器。（对比RNN based发现，RNN的模型在句子分类上具有较高的准确率，这样导致发生先前提到的判别器过度训练导致的梯度消失问题）

文章的训练手法和SeqGAN十分类似，首先采用MLE的方式对生成器和判别器进行预训练。之后通过policy gradients+Monte Carlo的方式将两者结合起来做对抗学习。文章还采用了一种取代Schedule Sampling的方式来应对exposure bias的问题——Professor Forcing。

### Professor Forcing
不同于Schedule Sampling采用预设的门槛来决定什么时候采用teacher forcing的训练方式和free running。professor forcing的方法采用训练一个分类器（discriminator）来对门槛进行评估。有点类似RL中的DQN和Policy gradients的关系，具体示意图如下：

![](https://i.imgur.com/Ox01o2b.png)

这样的方法可以想象成NN-based Schedule Sampling。

## Improved Training of Wasserstein GANs
- 论文链接：http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf
- Key Points：让WGANs也能在NLP领域上发挥作用
- Graph & Algorithm：

![](https://i.imgur.com/PrcrDaQ.png)

**思路**：总结WGANs与传统GANs在实作上的差异包括：1、判别器最后一层的输出去掉sigmoid（或者softmax，相当于不做正规化）。2、生成器和判别器的loss不去log。（相当于利用value代替probability）3、每次更新判别器的参数之后把它们的绝对值截断到一个不超过c的范围（weight clipping）。4、不采用基于动量（momentum）的优化函数（如Adam），转而采用RMSProp或者SGD。

作者发现利用weight clipping的方式对判别器的输出分布进行Lipschitz限制，尽可能将结果逼近可靠的Wasserstein distance的方式是导致训练不稳定的一个关键因素。因为论文提出了通过梯度惩罚（gradient penalty）来代替之前采用的Lipschitz限制。

![](https://i.imgur.com/1QUo5yh.png)

可以从算法中看出损失函数除了原先的部分，还加入了一个梯度惩罚项。

## MaskGAN: Better Text Generation via Filling in the______
- 论文链接：https://arxiv.org/pdf/1801.07736.pdf
- Key Points：采用Mask的方式让GANs在序列生成的时候更加robust
- Graph & Algorithm：

![](https://i.imgur.com/rikhKho.png)

**思路**：在针对exporsure bias和部分序列reward的问题上，论文采用了fill-in-the-blank或in-filling的方式加以缓解。在训练过程中，一部分文本信息将会被去除（masked），而模型的目的是随后需要填充文本中缺失的那些部分，让他尽可能和真实资料相近。

在文本数据中有着更复杂的mode，比如词组、短语、长的谚语等，这些都是我们希望模型能够学习的。然而SeqGAN中判别器是对一整个完整的句子进行判别，可以想象对于GANs来说句子的mode更是稀疏，那么GANs学习到的句子diversity会远远不足。于是，作者想办法让GANs能够在词级别（sub-sequence）的生成上做判断。然而，当模型在sampling时，也就是完全free-running的模式下，我们并不能确定每个位置的词还和ground truth保持一致，这会给模型训练带来很大的干扰。于是作者采用了让模型做完型填空的形式（像一种改进版Schedule Sampling），在原句子中随机挖几个空（mask），然后让模型去填，之后让判别器去判断输出的是否是正确的词。

模型采用一种 curriculum learning 的方式，随着训练的进行，不断增加sequence的长度，使得模型从short-term dependencies开始，慢慢学习到long-term dependencies。
模型需要经过两轮预训练：
- 用传统的 maximum likelihood training pre-train一个语言模型，之后将这个语言模型的网络权值赋值给seq2seq模型的encoder和decoder模块;
- 用得到的seq2seq模型来做完型填空任务（ in-filling task），用传统的 maximum likelihood training来训练，得到的seq2seq模型网络权值赋值给MaskGAN的generator和discriminator。

# Attention-Based Reward Conditional Sequence Generative Adversarial Network(AR-CSGAN proposed by ourselves)

- Key Points：
    - **Conditional Sequence Generative Model**
    - **Schedule Sampling**
    - **Attention-Based Reward System**
    - **Policy Gradients Optimization**
- Graph & Algorithm：

- **Model overview**

![](https://i.imgur.com/rVQu9q5.png)

- **Reward attention system**

![](https://i.imgur.com/dEPeXUq.png)

- **Global teacher forcing**

![](https://i.imgur.com/dZC1l9f.png)

- **The whole system**

![](https://i.imgur.com/3kXUwmh.png)

- **Algorithm**

![](https://i.imgur.com/s06VLxG.png)

- **We propose an attention-based reward conditional SeqGAN Model to assign the reward from the discriminator.**
    - Generating diverse, meaningful and more extended sequences
    - Solving the problems of making GANs difficult to work in NLP
    - Automatically assign the rewards from discriminator
    - Stable and computationally efficient
- **Some special training strategies are presented, which help us to train our model stably.**
    - Sequence AutoEncoder
    - Teacher Forcing
    - Dynamic Learning Rate
    - Weight clipping

# 参考（reference）

- [DeepDrumpf Tweeter](https://www.csail.mit.edu/news/postdoc-develops-twitterbot-uses-ai-sound-donald-trump)
- [Generative Adversarial Networks for Text](https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/)
- [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
- [Role of RL in Text Generation by GAN](https://zhuanlan.zhihu.com/p/29168803)
- [AI技术讲座精选：GAN 在 NLP 中的尝试](https://www.toutiao.com/i6376751517893919234/)
- [《MASKGAN: BETTER TEXT GENERATION VIA FILLING IN THE __ __》阅读笔记](https://zhuanlan.zhihu.com/p/34657045)
- The encode-decode framework refer to Neural responding machine for short-text conversation (2015)
- ARJOVSKY, Martin; CHINTALA, Soumith; BOTTOU, Léon. Wasserstein gan. arXiv preprint arXiv:1701.07875, 2017.
- ZHANG, Yizhe; GAN, Zhe; CARIN, Lawrence. Generating text via adversarial training. In: NIPS workshop on Adversarial Training. 2016.
- YU, Lantao, et al. SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. In: AAAI. 2017. p. 2852-2858.
- LI, Jiwei, et al. Adversarial learning for neural dialogue generation. arXiv preprint arXiv:1701.06547, 2017.
- KUSNER, Matt J.; HERNÁNDEZ-LOBATO, José Miguel. Gans for sequences of discrete elements with the gumbel-softmax distribution. arXiv preprint arXiv:1611.04051, 2016.
- PFAU, David; VINYALS, Oriol. Connecting generative adversarial networks and actor-critic methods. arXiv preprint arXiv:1610.01945, 2016.
- YANG, Zhen, et al. Improving neural machine translation with conditional sequence generative adversarial nets. arXiv preprint arXiv:1703.04887, 2017.
- GULRAJANI, Ishaan, et al. Improved training of wasserstein gans. In: Advances in Neural Information Processing Systems. 2017. p. 5767-5777.
- FEDUS, William; GOODFELLOW, Ian; DAI, Andrew M. Maskgan: Better text generation via filling in the _. arXiv preprint arXiv:1801.07736, 2018.


