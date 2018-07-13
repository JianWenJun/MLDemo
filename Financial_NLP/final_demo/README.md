
#### 1.项目结构
##### 1.1代码文件说明
>**./final_demo/util.py**：1.管理整个项目的文件存放路径 
2.存储和读取各种方法抽取的特征，并组合成DataFrame输出。
**./final_demo/data_prepare.py**：1.文本的清理工作 2.词向量训练工作</br>
**./final_demo/extract_feature.py**：1.各种方法进行特征抽取，并进行保存</br>
**./final_demo/train_model.py**：1.深度模型的构建工作</br>
**./final_demo/main.py**：1.最后整个项目的运行，读取各个方法抽取的特征并使用分类模型进行分类预测</br>

##### 1.2代码运行期间生成文件的目录说明(根目录为./atec/data)
>1./atec/data :根目录,存放该项目下的相关文件夹，和比赛最初的.csv文件(原始数据文件)</br>
>2../atec/data/aux :存放最后提交测试平台的重压文件，包括分词字典，词向量矩阵，停用词，拼写纠错，疑问词。</br>
>3./atec/data/feature :特征存放文件，存放各个方法进行抽取后的特征，一个抽取方法包括3个文件，特征值文件和特征列名文件。</br>
4./atec/data/preprocessed: 原始数据文本预处理文件目录</br>
5../atec/data/tmp: 存放用于提取特征的深度模型</br>
6../atec/data/trained: 存放最后需要提交测试平台的模型文件</br>
#### 2.项目使用步骤
>1.整个项目的初始化工作(设置整个项目的根目录，决定是否需要创建(第一次为create_dir为True,之后为False))</br>
2.文本的预处理工作</br>
3.特征抽取</br>
4.整个项目的运行(构建最终的分类模型，交叉验证的方式)</br>

```python
#最终的运行方式
python util.py
python data_prepare.py
python extract_feature.py
python main.py
```
#### 3赛题思路
[蚂蚁金融NLP竞赛——文本语义相似度赛题总结](https://jianwenjun.xyz/2018/07/13/%E8%9A%82%E8%9A%81%E9%87%91%E8%9E%8DNLP%E7%AB%9E%E8%B5%9B%E2%80%94%E2%80%94%E6%96%87%E6%9C%AC%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%B5%9B%E9%A2%98%E6%80%BB%E7%BB%93/)
