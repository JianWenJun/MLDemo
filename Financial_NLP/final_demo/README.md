
#### 项目结构
##### 代码文件说明

**./final_demo/util.py**
1.管理整个项目的文件存放路径 2.存储和读取各种方法抽取的特征，并组合成DataFrame输出

**./final_demo/data_prepare.py**
1.文本的清理工作 2.词向量训练工作

**./final_demo/extract_feature.py**
1.各种方法进行特征抽取，并进行保存

**./final_demo/train_model.py**
1.深度模型的构建工作

**./final_demo/main.py**
1.最后整个项目的运行，读取各个方法抽取的特征并使用分类模型进行分类预测

##### 生成文件的目录说明
./atec/data :根目录,存放该项目下的相关文件夹，和比赛最初的.csv文件(原始数据文件)
./atec/data/aux :存放最后提交测试平台的重压文件，包括分词字典，词向量矩阵，停用词，拼写纠错，疑问词
./atec/data/feature :特征存放文件，存放各个方法进行抽取后的特征，一个抽取方法包括3个文件，特征值文件和特征列名文件
./atec/data/preprocessed: 原始数据文本预处理文件目录
./atec/data/tmp: 存放用于提取特征的深度模型
./atec/data/trained: 存放最后需要提交测试平台的模型文件

#### 项目使用
##### 1.整个项目的初始化工作
**./final_demo/util.py** 设置整个项目的根目录，决定是否需要创建(第一次为create_dir为True,之后为False).
util.py初始化整个项目的基础类
project = Project.init('/Users/jian/PythonPrMl/Financial_NLP/atec',create_dir=False)

##### 2.文本的预处理工作(哪步工作已经做过后，可以对相应的步骤进行注释)
**./final_demo/data_prepare.py**
# step 1 # 预处理文本
    jieba.load_userdict(project.aux_dir + dict_path)
    data_local_df = pd.read_csv(project.data_dir + train_all, sep='\t', header=None,
                                names=["index", "s1", "s2", "label"])
    data_test_df = pd.read_csv(project.data_dir + test_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
    data_all_df = pd.concat([data_local_df, data_test_df])


    preprocessing(data_local_df,'train_0.6_seg')
    preprocessing(data_test_df,'test_0.4_seg')
    preprocessing(data_all_df,'data_all_seg')
    # 保存label
    project.save(project.features_dir + 'y_0.6_train.pickle', data_local_df['label'].tolist())

# step 2 # 利用训练集做语料库训练词向量

    pre_train_w2v()

#  step 3 # 保存训练集中的词汇对应的词向量矩阵，总共有3种方式
    process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=1,isStore_ids=True)
    process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=2,isStore_ids=False)
    process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=3,isStore_ids=False)

##### 3.特征抽取
**./final_demo/extract_feature.py**
每种抽取方法对应一个函数，抽取特征的步骤：
step1 定义抽取特征的方法名
step2 载入需要抽取特征的数据
step3 定义抽取特征的方法(如果是深度学习需要调用**./final_demo/train_model.py**文件中构建模型的方法)
step4 逐行读取数据，并进行特征的抽取，保存到相应的np.array数组中
step5 特征的保存
调用方法：
project.save_features(feature_train, feature_test, col_names, feature_name)
参数说明：训练集的抽取出的特征feature_train，测试集的抽取特征feature_test，抽取特征的方法的多列特征的列名col_names，抽取特征的方式名feature_name

##### 4.整个项目的运行
**./final_demo/main.py**
3个步骤：
step1 选出你的抽取特征的方法名组合成list

step2 加载多个抽取方法抽取出的特征数据和训练集的label数据
调用方法：
project.load_feature_lists(feature_names_list)
参数说明：feature_names_list为你的抽取特征的方法名组合成list
返回参数说明：你的训练集经过多种抽取方法组合成的多个列，你的测试集经过多种抽取方法组合成的多个列，每种抽取方法对应的列的index。
注意：一种抽取方法可能对应着多种特征列

step3 构建最终的分类模型，交叉验证的方式

