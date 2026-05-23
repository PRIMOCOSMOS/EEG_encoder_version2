### SEED-VII EEG预处理方案

---

```
原始 EEG (62通道, 200Hz)
    ↓ 带通滤波 (0.1-70Hz)以及工频陷波，原数据集已完成，不再需要重复了 
    ↓ ICA 去除眼动、肌电伪迹
    ↓ 基线校正，平均参考（CAR）
    ↓ 分段 (4秒窗口, 50%重叠作为一定的数据增强的手段，只取每一个情绪刺激的居中60%，注意避免数据泄漏)
    ↓ 标准化（优先考虑按通道 z-score）
    ↓ 标签处理：连续标签归一化到 [0,1] / 类别标签提取

```
对于分段问题，如果按窗口数直接训练，长视频会主导训练。可以每个对每个 clip 采样固定数量窗口，居中的原则不变，滑动窗口的原则也不变，
但是要通过均衡窗口数量来实现样本的均衡化。为避免数据泄露（尤其是数据预处理泄露），**在各自的数据里切4s窗口，不要整个切完了放一起划分，那样就会乱。**记住，先切分，后处理。


目标：EEG_preprocessed文件夹(zip压缩包下)

有1-20.mat文件（共20个），对应20个被试各自的Session中的各80各Clip，已降采样至200Hz, 62个通道

内部含有相应的字段：此处给一个案例。

```
  Name       Size                  Bytes  Class     Attributes

  1         62x12200             6051200  double              
  10        62x51800            25692800  double              
  11        62x23600            11705600  double              
  12        62x14600             7241600  double              
  13        62x31600            15673600  double              
  14        62x43800            21724800  double              
  15        62x29400            14582400  double              
  16        62x65800            32636800  double              
  17        62x48200            23907200  double              
  18        62x35800            17756800  double              
  19        62x47400            23510400  double              
  2         62x44400            22022400  double              
  20        62x35000            17360000  double              
  21        62x46000            22816000  double              
  22        62x47800            23708800  double              
  23        62x20200            10019200  double              
  24        62x13800             6844800  double              
  25        62x41200            20435200  double              
  26        62x50000            24800000  double              
  27        62x14800             7340800  double              
  28        62x59600            29561600  double              
  29        62x37800            18748800  double              
  3         62x37800            18748800  double              
  30        62x19800             9820800  double              
  31        62x62800            31148800  double              
  32        62x58400            28966400  double              
  33        62x19400             9622400  double              
  34        62x47600            23609600  double              
  35        62x45000            22320000  double              
  36        62x20600            10217600  double              
  37        62x37600            18649600  double              
  38        62x34800            17260800  double              
  39        62x26600            13193600  double              
  4         62x45800            22716800  double              
  40        62x17800             8828800  double              
  41        62x26800            13292800  double              
  42        62x32800            16268800  double              
  43        62x29800            14780800  double              
  44        62x13800             6844800  double              
  45        62x41600            20633600  double              
  46        62x15200             7539200  double              
  47        62x19400             9622400  double              
  48        62x48000            23808000  double              
  49        62x25800            12796800  double              
  5         62x53200            26387200  double              
  50        62x14400             7142400  double              
  51        62x10800             5356800  double              
  52        62x50200            24899200  double              
  53        62x28400            14086400  double              
  54        62x57200            28371200  double              
  55        62x35000            17360000  double              
  56        62x12400             6150400  double              
  57        62x41600            20633600  double              
  58        62x24200            12003200  double              
  59        62x70000            34720000  double              
  6         62x43800            21724800  double              
  60        62x47800            23708800  double              
  61        62x18200             9027200  double              
  62        62x48800            24204800  double              
  63        62x55400            27478400  double              
  64        62x14200             7043200  double              
  65        62x32600            16169600  double              
  66        62x18000             8928000  double              
  67        62x45400            22518400  double              
  68        62x33200            16467200  double              
  69        62x38800            19244800  double              
  7         62x52800            26188800  double              
  70        62x29600            14681600  double              
  71        62x34200            16963200  double              
  72        62x23800            11804800  double              
  73        62x53600            26585600  double              
  74        62x47600            23609600  double              
  75        62x32600            16169600  double              
  76        62x23800            11804800  double              
  77        62x17400             8630400  double              
  78        62x38200            18947200  double              
  79        62x24400            12102400  double              
  8         62x30000            14880000  double              
  80        62x26000            12896000  double              
  9         62x47200            23411200  double              

```

Session情绪标签：
```
{
  "experiment_design": {
    "emotion_labels": {
      "H": "Happy",
      "U": "Surprise",
      "N": "Neutral",
      "D": "Disgust",
      "F": "Fear",
      "S": "Sad",
      "A": "Anger"
    },
    
    "trial_structure": [
      {"step": 1, "duration": "3 s", "task": "Hints"},
      {"step": 2, "duration": "~3 min", "task": "Video"},
      {"step": 3, "duration": "3 s", "task": "Hints"},
      {"step": 4, "duration": "~10 s", "task": "Feedback"}
    ],
    
    "session_rules": {
      "total_trials": 20,
      "end_of_session_task": "Continuous labeling",
      "selection_rule": "Select 5 emotions out of 7 and arrange them into specific order",
      "time_interval_between_sessions": "> 24 h",
      "folds_per_session": 4,
      "trials_per_fold": 5
    },

    "session_sequences": {
      "Session_1": {
        "Fold_1": ["H", "N", "D", "S", "A"],
        "Fold_2": ["A", "S", "D", "N", "H"],
        "Fold_3": ["H", "N", "D", "S", "A"],
        "Fold_4": ["A", "S", "D", "N", "H"]
      },
      "Session_2": {
        "Fold_1": ["A", "S", "F", "N", "U"],
        "Fold_2": ["U", "N", "F", "S", "A"],
        "Fold_3": ["A", "S", "F", "N", "U"],
        "Fold_4": ["U", "N", "F", "S", "A"]
      },
      "Session_3": {
        "Fold_1": ["H", "U", "D", "F", "A"],
        "Fold_2": ["A", "F", "D", "U", "H"],
        "Fold_3": ["H", "U", "D", "F", "A"],
        "Fold_4": ["A", "F", "D", "U", "H"]
      },
      "Session_4": {
        "Fold_1": ["D", "S", "F", "U", "H"],
        "Fold_2": ["H", "U", "F", "S", "D"],
        "Fold_3": ["D", "S", "F", "U", "H"],
        "Fold_4": ["H", "U", "F", "S", "D"]
      }
    }
  }
}
```


### 模型设计

将任务重新定义为混合任务:**情绪类别分类；情绪强度感知**，模型双头输出。不建议盲目扩大参数量，参照 https://github.com/PRIMOCOSMOS/EEG_encoder 上的实现（这是一个基于SEED-IV的EEG-Conformer设计）即可，参数量大约是0.7-0.8M。保持这个参数规模即可。

损失函数设计：综合 分类损失（交叉熵 + 标签平滑）、强度回归损失、强度排序损失：

$$
\mathcal{L}_{weighted} = s_i \cdot [\alpha \mathcal{L}_{cls}^{(i)} + \beta \mathcal{L}_{reg}^{(i)}] + \gamma \mathcal{L}_{rank}^{(i,j)}
$$

权重策略：

基于 SEED-VII 的技术设计，利用连续标签过滤高情绪唤起数据被证明是有效的——建议设置强度阈值 $ \tau $（如 $ \tau = 0.5 $），将低强度样本在训练初期降权或剔除，避免弱情绪噪声样本污染编码器（但这对于中性情绪的学习效果可能会有影响）。具体地，可以将连续标签直接作为每个训练样本的动态损失权重：

$$
L_{weighted} = s_i \cdot [\alpha L_{cls}^{(i)} + \beta L_{reg}^{(i)}] + \gamma L_{rank}^{(i)}
$$

训练策略：先基于Adam 优化器（适当学习率）实现初步分类任务，预训练一定Epoch（只开启分类损失），再联合训练：

- 开启全部三项损失，使用损失权重调度：初期 $ \alpha = 1.0, \beta = 0.5, \gamma = 0.3 $，逐步提升 $ \gamma $ 至 0.8；
- 引入基于连续标签的样本权重动态调整；
- 余弦退火学习率，最小 $ 1 \times 10^{-5} $。


**强度排序损失可以先不应用，先利用分类损失和强度回归损失来进行深度学习，设计一个退化的方案**

```



输入: [Batch, 1, 62, T]
    ↓
[卷积模块] 时间卷积 + 空间深度卷积 + 池化
    ↓
[Transformer模块] 多头自注意力 (提取全局依赖)
    ↓ 提取此处的Embedding作为情绪编码器输出
    ├── [分类头] Linear → 7维 Softmax
    └── [强度回归头] Linear → 1维 Sigmoid

```

### 代码设计原则

支持周期断点保存，断点续训；定时停止并保存结果（日志，一次最多10个小时），以免进程被摧毁导致数据丢失；由于数据集在云端是以一个160G左右的zip形式存在的，要采取流式训练，
不要直接一口气全解压出来，造成存储爆炸。只是有针对性地把需要的数据提取出来训练即可。

要求分为模型设计、模型训练与继续训练、数据预处理、编码器运行编码的推理脚本，这些脚本都要以.py文件的形式一同上传，再以
Jupyter notebook的形式使用命令调用它们。**因此，你要先写Python代码完成每一个模块的功能，最后写Jupyter Notebook.**

更加要注意的是，我为了上传方便，把原来的压缩包分拆成了32个分卷，每个分卷5.37GB，你要先按序拼接回去，重新复原出我上文提到的
160G左右的zip文件；注意，ModelScope上的一个实例只有100G的持久化存储，你要在合并中合理优化逻辑，防止存储爆炸。另外，合成完的Zip文件也要上传回我的数据集https://www.modelscope.cn/datasets/DEREKVERSE/SEED-VII/,利用ModelScope的Python SDK API.



---
