HEXO个人博客地址：[小简铺子](https://jianwenjun.xyz)

### 机器学习练手代码
描述：主要包括机器学习相关竞赛代码，论文项目复现代码，机器学习书籍实战代码

### 1ML
#### 1.1决策树相关算法
[决策树相关算法——ID3、C4.5的详细说明及实现](https://blog.csdn.net/u014732537/article/details/79667599) —— [代码地址](https://github.com/JianWenJun/MLDemo/blob/master/ML/DecisionTree/decision_tree.py)
>本篇博客记录的是使用python实现两个决策树相关的算法模型—— ID3、C4.5。其中训练模型使用的数据集是Adult。

[决策树相关算法——Bagging之基于CART的随机森林详细说明与实现](https://blog.csdn.net/u014732537/article/details/79667679)
>本篇博客主要记录的是基于CART决策树实现的随机森林算法，主要是从以下四个方面介绍: CART决策树的构建思想；集成学习中的Bagging思想；基于CART决策树的随机森林代码实现；随机森林不易过拟合的分析。(其中不易过拟合并不是说随机森林不会过拟合)

[决策树相关算法——Boosting之Adaboost&GBDT详细分析与实现](https://jianwenjun.xyz/2018/04/12/%E5%86%B3%E7%AD%96%E6%A0%91%E7%9B%B8%E5%85%B3%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94Boosting%E4%B9%8BAdaboost-GBDT%E8%AF%A6%E7%BB%86%E5%88%86%E6%9E%90%E4%B8%8E%E5%AE%9E%E7%8E%B0/)
>本篇博客主要记录的是集成学习中的Boosting提升算法的相关实现，主要分为以下四个部分，Boosting的提出，Boosting经典算法Adaboost的分析与实现，Adaboost算法的特例提升树的分析，梯度提升算法GBDT的提出原因及分析。

[决策树相关算法——XGBoost原理分析及实例实现(一)](https://jianwenjun.xyz/2018/04/26/%E5%86%B3%E7%AD%96%E6%A0%91%E7%9B%B8%E5%85%B3%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94XGBoost%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90%E5%8F%8A%E5%AE%9E%E4%BE%8B%E5%AE%9E%E7%8E%B0-%E4%B8%80/)  

[决策树相关算法——XGBoost原理分析及实例实现(二)](https://jianwenjun.xyz/2018/04/27/%E5%86%B3%E7%AD%96%E6%A0%91%E7%9B%B8%E5%85%B3%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94XGBoost%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90%E5%8F%8A%E5%AE%9E%E4%BE%8B%E5%AE%9E%E7%8E%B0-%E4%BA%8C/)  

[决策树相关算法——XGBoost原理分析及实例实现(三)](https://jianwenjun.xyz/2018/05/02/%E5%86%B3%E7%AD%96%E6%A0%91%E7%9B%B8%E5%85%B3%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94XGBoost%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90%E5%8F%8A%E5%AE%9E%E4%BE%8B%E5%AE%9E%E7%8E%B0-%E4%B8%89/)
>上述3篇博客主要记录的是XGBoost的代价函数的优化过程，XGBoost在构建决策树结构时，知道如何评定划分点的好坏的情况下，如何遍历查找出该树结构的切分点。最后，使用XGBoost对kaggle中的初级赛题Titanic: Machine Learning from Disaster进行预测的实例。

#### 1.2神经网络
[TensorFlow实现多层感知机及可视化训练过程中的数据记录](http://blog.csdn.net/u014732537/article/details/79412672) —— [代码地址](https://github.com/JianWenJun/MLDemo/blob/master/ML/TensorDemo/NN_tf.py)
>本篇博客主要有2个目的，第一，记录学习使用TensorFlow的操作流程；第二，将TensorFlow训练数据模型过程中的参数数据进行可视化记录。
### 2NLP
[卷积神经网络(TextCNN)在句子分类上的实现](http://blog.csdn.net/u014732537/article/details/79573174) —— [代码地址](https://github.com/JianWenJun/MLDemo/blob/master/NLP/Text_CNN/text_cnn_main.py)
> 本篇博客记录的是论文Convolutional Neural Networks for Sentence Classification中的实验实现过程，一篇介绍使用CNN对句子进行分类的论文。尽管网上有些代码已经实现了使用CNN进行句子分类(TextCNN),但是是基于Theano来实现的，本文将介绍使用TensorFlow来实现整个论文的实验过程，一方面熟悉使用TensorFlow API,另一方面加深自己对CNN在NLP上的应用的理解. 


##### 码代码不易，欢迎star~ ，谢谢~