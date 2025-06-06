尽量统一专业术语的使用，可参考下表：

| 章节名称                     | 单词                       | 采用翻译             | 其他             | 含义                         | 备注 |
|--------------------------|--------------------------|------------------|----------------|----------------------------|------|
|                          | the normalizing constant | 标准化常量           | 归一化常数         | 先验概率                       |      |
|                          | argument                 | 实参/实际参数         |                | 调用的参数                      |      |
|                          | parameters               | 形参/形式参数         |                | 函数中的参数                     |      |
|                          | function signature       | 函数签名            |                | 函数的参数和返回类型说明               |      |
|                          | call                     | 调用              |                | 调用子程序                       |      |
|                          | Integration              | 积分              |                | 泛指                           |      |
|                          | integral                 | 积分              |                | 特指                           |      |
|                          | integrator               | 积分器             |                | 用于数值积分的算法或函数               |      |
|                          | evaluate                 | 求解              |                | 求（方程式、公式、函数）的数值         |      |
|                          | norm                     | 范数              |                | 常见如 L2 范数、L1 范数等              |      |
|                          | machine epsilon          | 机械极小值           |                | 浮点数精度的最小差值                 | 舍入误差 |
|                          | quadrature               | 求积              | 正交；求积；弦       | 一类数值积分方法                   |      |
|                          | numerator                | 式子（意译）          | 分子             | 积分中出现的表达式                  | 上下文中是定积分式子 |
|                          | thin QR decomposition    | 薄 QR 分解         |                | QR 分解的一种稀疏形式                |      |
|                          | item-response model      | 项目-响应模型         |                | 测验理论模型                     |      |
|                          | logistic regression      | Logistic 回归     |                | 广义线性模型中用于分类问题的形式          | 与“逻辑回归”为同义，保留英文形式 |
|                          | probit regression        | Probit 回归       |                | 与 Logistic 相似，但使用正态分布         |      |
|                          | improper prior           | 非正规先验           |                | 非归一化的先验分布                  | improper prior 的标准译法 |
|                          | improper flat prior      | 非正规的均匀先验       |                | improper flat prior            | 特指没有归一化常数的均匀分布型先验 |
|                          | informative prior        | 有信息的先验         |                | 表达主观知识的先验分布               |      |
|                          | noninformative prior     | 无信息的先验         |                | 避免主观偏见的先验分布               |      |
|                          | weakly informative prior | 弱信息先验           |                | 有轻微正则化作用的先验                | 常用于防止过拟合 |
|                          | meta analysis            | 元分析             |                | 汇总多个研究的统计技术                |      |
|                          | cluster                  | 聚类              |                | 聚合分析单位的集合（名词）             | 不翻译为“簇” |
|                          | multimodality            | 多峰性             |                | 指分布具有多个峰值                  | 避免译为“多模态性” |
| 回归模型                   | multilevel generalized linear models | 多层次广义线性模型       |                | 广义线性模型的层级拓展                | 第2章标题 |
| 2.2 线性回归               | predictor                | 自变量             |                | 回归模型中的输入变量                 |      |
| 2.2 线性回归               | outcome                  | 因变量             |                | 回归模型中的输出变量                 |      |
| 2.2 线性回归               | sampling                 | 采样              | 抽样             | 数据抽取过程                     | 建议统一为“抽样”，更贴合统计学习惯 |
| 2.2 线性回归               | overloaded               | 重载              |                | 指模型或术语的多义性                  | 如参数或函数用途多样 |
| 2.2 线性回归               | improper priors          | 非正规先验           |                | improper priors 同义                | 采用统一术语风格 |
| 2.10 逻辑回归和概率回归        | link function            | 链接函数            | 联系函数          | 将线性预测映射为概率的函数              | 参考周志华《机器学习》建议使用“联系函数” |
| 2.10 逻辑回归和概率回归        | Logit Parameterization   | 对数几率参数化         | 逻辑参数化         | logit(p/(1−p)) 的形式              | 建议使用“对数几率参数化”更精确 |
| 2.18 分层logistic回归       | pooling                  | 汇集              |                | 多组之间共享或合并估计值               | 分层模型中的信息整合机制 |
| 2.26 分层模型的多元先验       | group-level coefficients | 组级系数            |                | 高层（组间）模型参数                 |      |
| 2.26 分层模型的多元先验       | scale vector             | 尺度向量            |                | 控制先验协方差或方差大小的向量            |      |
| 6.10 缺失的多变量数据         | dummy values             | 虚拟值             |                | 占位而非实际数据                    | 不同于虚拟变量 dummy variable |
| 10.有限混合                | modes                    | 峰值              | 模态             | 分布中的局部最大点                   | “模态”也可用，建议最终统一 |
| 12.2 单位向量              | compact                  | 紧致的             |                | 数学术语，指集合在拓扑意义上紧性            | 参考“紧集”翻译自维基百科 |
| 12.2 单位向量              | improper                 | 不正确的            | 合理的；适合的       | improper distribution 的意译           | 建议统一为“非正规”以避免歧义 |
| 12.4 圆、球体和超球体         | a set of points          | 一组点             |                | $S^2$ 中 $\mathbb{R}^3$ 的点集合        | 可翻译为“$\mathbb{R}^3$ 中的一组点”更贴切 |
| 回归模型（IRT相关）           | discrimination parameter | 区分参数            |                | 用于描述项目区分度的参数                | 出自 IRT 模型中关于共线性问题的讨论 |
