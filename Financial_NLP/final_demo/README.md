
#### 项目结构
##### 代码文件说明
>**./final_demo/util.py**1.管理整个项目的文件存放路径 2.存储和读取各种方法抽取的特征，并组合成DataFrame输出
**./final_demo/data_prepare.py**
1.文本的清理工作 2.词向量训练工作
**./final_demo/extract_feature.py**1.各种方法进行特征抽取，并进行保存
**./final_demo/train_model.py**1.深度模型的构建工作
**./final_demo/main.py**1.最后整个项目的运行，读取各个方法抽取的特征并使用分类模型进行分类预测

##### 代码运行期间生成文件的目录说明(根目录为./atec/data)
>1./atec/data :根目录,存放该项目下的相关文件夹，和比赛最初的.csv文件(原始数据文件)
>2../atec/data/aux :存放最后提交测试平台的重压文件，包括分词字典，词向量矩阵，停用词，拼写纠错，疑问词
>3./atec/data/feature :特征存放文件，存放各个方法进行抽取后的特征，一个抽取方法包括3个文件，特征值文件和特征列名文件
4./atec/data/preprocessed: 原始数据文本预处理文件目录
5../atec/data/tmp: 存放用于提取特征的深度模型
6../atec/data/trained: 存放最后需要提交测试平台的模型文件
#### 项目使用步骤
>1.整个项目的初始化工作(设置整个项目的根目录，决定是否需要创建(第一次为create_dir为True,之后为False))
2.文本的预处理工作
3.特征抽取
4.整个项目的运行(构建最终的分类模型，交叉验证的方式)

```python
# 最终的运行方式
python util.py
python data_prepare.py
python extract_feature.py
python main.py
```
