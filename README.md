# Data-driven Approach for Quality Evaluation on Knowledge Sharing Platform

## Introduction
Knowledge sharing plays a significant role in knowledge acquisition for ordinary people,
however, due to the insufficient evaluation schemes, people suffer from lots of low-level
knowledge services, which overwhelm the platform as well.

In our paper, we propose a method to automatically predict a Zhihu Live's
score through multi-modal learning (by learning from both database records 
and comments corpus within a single model). 
Our proposed ```HybridNet``` achieves the best performance among baseline 
models.

| Model | MAE | RMSE |   
| :---: | :---: | :---: | 
| SVR (RBF Kernel) | 0.2252 | 0.327|
| SVR (Linear Kernel) | 0.2257 | 0.3267|
| SVR (Poly Kernel) | 0.2255 | 0.3268|
| KNN | 0.2401 | 0.3275|
| Linear Regression | 0.2354 | 0.3224|
| MLP | 0.2397 | 0.3276|
| MTNet | 0.2255 | 0.3244 |


For deeper analysis, please read my article at [Zhihu](https://zhuanlan.zhihu.com/p/30514792).
More details will be illustrated in our paper: 

```Data-driven Approach for Quality Evaluation on Knowledge Sharing Platform.```

