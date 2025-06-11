尽量统一专业术语的使用，可参考下表：

| 章节名称                               | 单词                                 | 采用翻译           | 其他翻译           | 含义                                     | 备注   |
|:---------------------------------------|:-------------------------------------|:-------------------|:---------------|:-----------------------------------------|:-------|
|                                        | thin QR decomposition                | 薄 QR 分解         |                |                                          |        |
|                                        | item-response model                  | 项目-响应模型      |                |                                          |        |
|                                        | logistic regression                  | Logistic 回归      |                |                                          |        |
|                                        | probit regression                    | Probit 回归        |                |                                          |        |
|                                        | improper flat prior                  | 非正规的均匀先验   |                |                                          |        |
|                                        | informative prior                    | 有信息的先验       |                |                                          |        |
|                                        | noninformative prior                 | 无信息的先验       |                |                                          |        |
|                                        | weakly informative prior             | 弱信息先验         |                |                                          |        |
|                                        | meta analysis                        | 元分析             |                |                                          |        |
|                                        | cluster                              | 聚类               |                |              | 作为名词使用时，不翻译为“簇”       |
|                                        | multimodality                        | 多峰性             |                |                       |不翻译为“多模态性”         |
| 1 回归模型                             | multilevel generalized linear models | 多层次广义线性模型 |                |                                          |        |
| 1.1 线性回归                           | predictor                            | 自变量             |                |                                          |        |
| 1.1 线性回归                           | outcome                              | 因变量             |                |                                          |        |
| 1.1 线性回归                           | sampling                             | 采样               |                |  |        |（有疑问，感觉改成“抽样”会不会更好一点）
| 1.1 线性回归                           | overloaded                           | 重载               |                |                                          |        |
| 1.1 线性回归                           | improper prior                       | 非正规先验         |                |                                          |        |
| 1.5 Logistic 回归和 Probit 回归        | link function                        | 链接函数           |                |                |  （是否可译为“联系函数”？）      |
| 1.5 Logistic 回归和 Probit 回归        | Logit Parameterization               |                    |                |           |  “逻辑参数化”或“对数几率参数化”       |
| 1.9 分层回归                           | pooling                              | 汇集               |                |                                          |        |
| 1.13分层模型的多元先验                 | group-level coefficients             | 组级系数           |                |                                          |        |
| 1.13分层模型的多元先验                 | scale vector                         | 尺度向量           |                |                                          |        |
| 3 缺失数据和部分已知参数               | dummy values                         | 虚拟值             |                |                                          |        |
| 5 有限混合模型                         | modes                                | 模态               |                |                   |也可翻为“峰值”，暂保留原译         |
| 11 方向、旋转和超球面                  | compact                              | 紧致的             |                | 如数学中的紧集                           |        |
| 11 方向、旋转和超球面                  | improper                             | 不正确的           |                |                       |也可译为“非正规”“不合理”等         |
| 11.2 圆、球体和超球体                  | a set of points                      | 一组点             |                | $\mathbb{R}^3$ 中的点集                  |        |
| 11.2 圆、球体和超球体                  | discrimination parameter             | 区分参数           |                | IRT 中的区分度参数                       |        |
|13  常微分方程                       |forward sensitivity                    |前向灵敏度           |                |ODE 解对参数的前向敏感度               |        |
|13  常微分方程                       |adjoint sensitivity                   |伴随灵敏度            |                |反向计算梯度的 ODE 方法                |        |
|13  常微分方程                       |stiffness / stiff system               |刚性系统               |                |包含快速变化与慢速变化的微分系统        |        |
|13  常微分方程                       | system function                     | 系统函数             |                  |描述 ODE 系统结构的函数                       |       |
|13  常微分方程                       | time derivative                  | 时间导数                 |                |对时间的导数                      |       |
|13  常微分方程                       | initial condition                | 初始条件                 |                 |ODE 解的起始值设定                    |       |
|13  常微分方程                       | measurement error           | 测量误差            |                            |观测值中非系统性偏差         |         |
|13  常微分方程                       | generated quantities block  | 生成数量块           |                           | Stan 模型中生成推断结果的代码段 |        |
|13  常微分方程                       | initial value problem (IVP) | 初值问题            |                           | 指定初始条件下求解微分方程的问题   | 可用于“ODE 初值问题” |
|13  常微分方程                       | state variable              | 状态变量            |                            |表征系统状态的变量，如位置和速度等  |         |
|13  常微分方程                       | multivariate Student-t      | 多元 Student-t 分布 |                            |多维具有重尾和相关性的概率分布    |               |
|13  常微分方程                       | stiff ODE                       | 刚性 ODE |                            | 含有多个时间尺度、导致求解困难的微分方程 |      |
|13  常微分方程                       | warmup                          | 预热     |                            | MCMC 初始化阶段，用于调整采样参数  |      |
|13  常微分方程                       | time-scale                      | 时间尺度   |                            | 描述系统中变量变化快慢的相对速度     |    |
|13  常微分方程                       | control parameters        | 控制参数  |                            | 用于调节求解精度和步数等行为的 ODE 参数 |               |
|13  常微分方程                       | relative tolerance (RTOL) | 相对容差  |                            | 控制误差相对于解的大小            |               |
|13  常微分方程                       | absolute tolerance (ATOL) | 绝对容差  |                            | 控制误差的最小阈值（无论解是否接近零）    |               |
|13  常微分方程                       | discontinuity             | 间断点   |                            | 状态函数中不连续或跃变的位置         |                  |
|13  常微分方程                       | ill-defined problem       | 不适定问题 |                            | 数学上解不存在或不唯一的微分方程       |               |
|13  常微分方程                       | adjoint ODE solver    | 伴随 ODE 求解器 |                            | 通过后向积分计算梯度的 ODE 求解方法     |      |
|13  常微分方程                       | checkpointing         | 检查点机制      |                            | 在前向积分中保存状态以便后向阶段插值使用     |                 |
|13  常微分方程                       | Hermite interpolation | Hermite 插值 |                            | 一种插值方法，匹配函数及导数           |         |
|13  常微分方程                       | matrix exponential    | 矩阵指数       |                            | 定义为幂级数展开的矩阵函数，用于线性 ODE 解 |           |
|13  常微分方程                       | linear ODE system     | 线性 ODE 系统  |                            | 状态导数是状态本身线性函数的微分方程组      |                 |
|13  常微分方程                       | quadrature tolerance  | 积分容差       |                            | 在伴随求解器中用于积分计算精度控制        |                 |
| 14 Computing One Dimensional Integrals | integrator                           | 积分器             |                |                                          |        |
| 14 Computing One Dimensional Integrals | evaluate                             | 求解               |                | 求方程、公式、函数的数值                 |        |
| 14 Computing One Dimensional Integrals | norm                                 | 范数               |                |                                          |        |
| 14 Computing One Dimensional Integrals | machine epsilon                      | 机械极小值         |                | 舍入误差                                 |        |
| 14 Computing One Dimensional Integrals | quadrature                           | 求积               | 正交；求积；弦 |                                          |        |
| 14 Computing One Dimensional Integrals | numerator                            | 式子（意译）       | 分子           | 上下文为定积分式子                       |        |
| 14 Computing One Dimensional Integrals | the normalizing constant             | 标准化常量         | 归一化常数     | 先验概率                                 |        |
| 14 Computing One Dimensional Integrals | argument                             | 实参/实际参数      |                | 调用的参数                               |        |
| 14 Computing One Dimensional Integrals | parameters                           | 形参/形式参数      |                | 函数中的参数                             |        |
| 14 Computing One Dimensional Integrals | function signature                   | 函数签名           |                | 函数的参数和返回类型说明                 |        |
| 14 Computing One Dimensional Integrals | call                                 | 调用               |                | 调用子程序                               |        |
| 14 Computing One Dimensional Integrals | Integration                          | 积分               |                | 泛指                                     |        |
| 14 Computing One Dimensional Integrals | integral                             | 积分               |                | 特指                                     |        |
|15  复数                       | imaginary literals                        | 虚数字面量       |                            | 构造复数时的虚部字面量表达      |        |
|15  复数                       | complex vector                            | 复向量         |                            | Stan 中复数元素的向量类型    |        |
|15  复数                       | Cholesky factor                           | Cholesky 因子 |                            | 协方差矩阵的 Cholesky 分解 |        |
|15  复数                       | multivariate normal                       | 多元正态分布      |                            | 用于建模相关的随机变量        |        |
|15  复数                       | promotion                                 | 类型提升        |                            | 实数/整数自动转为复数        |        |
|18 浮点数算术                       | subnormal number           | 次正规数      |                            |               |                   |
|18 浮点数算术                       | signed zero                | 符号零      |                            |               |                   |
|18 浮点数算术                       | not-a-number (NaN)         | 非数字（NaN） |                            |               |                   |
|18 浮点数算术                       | positive/negative infinity | 正/负无穷    |                            |               |                   |
|18 浮点数算术                       | scientific notation        | 科学计数法    |                            |               |                   |
|18 浮点数算术                       | significand figures        | 有效数字       |    尾数                        |               |                   |
|18 浮点数算术                       | exponent (in IEEE 754)     | 指数       |                            |               |                   |
|18 浮点数算术                       | arithmetic precision      | 算术精度       |                           | IEEE 754 中尾数所决定的有效位数          |                 |
|18 浮点数算术                       | machine precision         | 机器精度       |                           | 即 double 精度中能区分 1 和 1+ε 的最小 ε |                 |
|18 浮点数算术                       | rounding error            | 舍入误差       |                           | 数值计算中常见误差类型之一                 |                 |
|18 浮点数算术                       | catastrophic cancellation | 抵消灾难       |                           | 常用于描述相近数值相减时的精度问题             |                 |
|18 浮点数算术                       | overflow                  | 上溢         |     溢出                      | 计算结果超出可表示最大浮点数                |                 |
|18 浮点数算术                       | underflow                 | 下溢         |                           | 计算结果小于最小非零浮点数                 |                 |
|18 浮点数算术                       | log scale                 | 对数尺度       |                           | 常用于避免下溢                       |                 |
|18 浮点数算术                       | Welford’s algorithm       | Welford 算法 |                           | 用于方差估计的在线算法                   |                 |
|18 浮点数算术                       | CCDF                      | 互补累积分布函数   |                           | \$1 - F(x)\$，用于稳定计算尾概率        |                 |
|20  多重索引和范围索引                       | Multiple Indexing                      | 多重索引   |                           |         |                 |
|20  多重索引和范围索引                       | Range Indexing                     | 范围索引   |                           |         |                 |
|20  多重索引和范围索引                       | container                      | 容器   |                           |  （即数组、向量和矩阵）       |                 |
|20  多重索引和范围索引                       | Lower/upper bound                     | 下界/上界   |    下限/上限                       |         |                 |
|20  多重索引和范围索引                       | prefixes                     | 前若干个元素   | 开头部分                      | 表示向量或数组的前若干个元素        |    意译             |
|20  多重索引和范围索引                       | suffixes                     | 后若干个元素   | 末尾部分                      | 表示向量或数组的后若干个元素        |    意译             |

