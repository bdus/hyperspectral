# building log

* preprocess
* myimport
* train try
在搭完了架子之后开始测试mt效果

首先没有架子训练 初始化msra 
	最高 0.894988 indian_conv_msra0.894988.params
	
然后在架子下训练 初始化msra
	最高 0.87  很奇怪 最后稳定在0.79 一个局部最优解上了

* train_try_simple1.py	
架子下训练 初始化net.initialize(mx.init.Xavier(magnitude=2.24))
	基本稳定在0.81

* train_try_simple2.py
架子下训练 初始化indian_simple20.870362 
indian_simple_ema 
	基本稳定在0.83左右

* train_try_simple3.py
架子下训练 初始化indian_simple20.870362 
indian_simple_ema_uset
交替训练 train_data 和 val_data 交替epoch
ema逐渐加上 没有噪声
收敛很快 有一点震荡
	稳定在了0.85+上 还行

