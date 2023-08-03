---
layout: post
title:  "数值最优化笔记（1）：从统一的角度认识梯度下降、牛顿法与拟牛顿法"
subtitle: "Note of Numerical Optimization: A Unified View of Gradient Descent, Newton Method & Quasi-Newton Method"
tag: "math", "note"
---

数值最优化方法一般可以分为线搜索（line search）与信赖域（trust region）两类，在无约束优化的线搜索方法中，最主流的方法有：梯度下降、牛顿法、拟牛顿法与共轭梯度法。在本文中，我们将努力回答以下问题：

+ 如何从统一的角度理解这四种看似不同的方法？
+ 前三种方法的收敛性，以及到底是什么决定了它们在靠近解时的收敛性？



#### 1. 一个统一的角度

我们知道，对于线搜索方法，优化的每一步最重要的东西有两个：优化的方向和步长。我们假设优化问题的形式为$$\min_{x} f(x)$$, 从统一的角度来看，梯度下降法、牛顿法、拟牛顿法与共轭梯度法在优化的第$k$步选取的优化方向可以被概括为以下形式：
$$
p_k = -B_k^{-1}\nabla f_k + \beta_k p_{k-1}
$$
在梯度下降、牛顿法与拟牛顿法中，$\beta = 0$。

+ 对于梯度下降：$$ B = I$$

  

+ 对于牛顿法：$$B = \nabla^2 f$$。这是因为牛顿法的出发点是：假设当前所在点$x_k$接近局部最小点（local minimum）$$x^*$$, 那么我们可以用一个二次模型（三维中是一个开口向上的碗）来近似$$x_k$$附近$$f(x)$$的地形（optimization landscape）：

  $$
  m_k(p) := f(x_k+p) \approx f(x_k) + \nabla f(x_k)^{\top}p + \frac{1}		{2}p^{\top}\nabla^2 f(x_k)p
  $$

  通过求解$$\min_{p} m_k(p)$$， 可以得到最优的$$ p = - [\nabla^2 f]^{-1}\nabla f$$。

  

+ 对于拟牛顿法，$$ B $$的选取策略如下：拟牛顿法的思考出发点是我们没有真正Hessian阵的信息，只能根据优化过程中得到的一系列梯度去估计Hessian的相关信息。假设我们在第$$k-1 (k \geq 1)$$步所在的位置为$$x_{k-1}$$，$$x_k = x_{k-1} + \alpha p_{k-1}$$, 其中$$\alpha$$ 为步长。 我们在$x_k$处建立一个二次模型近似$$x_k$$附近$$f(x)$$的地形：
  $$
  m_k(p) := f(x_k + p) \approx f(x_k) + \nabla f(x_k)^{\top} p + \frac{1}{2}p^{\top} B_k p
  $$
  其中$$B_k$$是我们在$x_k$处对$f(x)$ Hessian的估计。

  为了确定$B_k$, 我们希望提出的这个模型$$m_k(p)$$能够满足以下性质：

  1. $$m_k(0) = f(x_k)$$ （天然满足）

  2. $$\nabla m_{k}(0) = \nabla f(x_k)$$ （天然满足）

  3. $$\nabla m_k(-\alpha p_{k-1}) = \nabla f(x_{k-1})$$

     展开：$$\nabla f(x_k) + B_{k} (-\alpha p_{k-1}) = \nabla f(x_{k-1}) $$

     利用$-\alpha p_{k-1} = - (x_{k-1} - x_{k})$, 有 $$B_k (x_k - x_{k-1}) = \nabla f(x_k) - \nabla f(x_{k-1})$$

     我们记$s_{k-1} = x_k - x_{k-1}, \ y_{k-1} = \nabla f_{k} - \nabla f_{k-1}$

     最后得：$$B_k s_{k-1} = y_{k-1}$$ （当然如果取的下标是 $$k 与 k+1$$， 教科书上就是$$B_{k+1}s_k = y_k$$）

     这个公式被称为secant equation. 

  4. $$B$$最好是对称正定的。

  将上面这4个条件整理成下面这个优化问题：
  $$
  \min_B \ || B - B_{k-1} || \\
  s.t. B= B^{\top}, \ Bs_{k-1} = y_{k-1}
  $$
  选取不同的$$||\cdot ||$$会给我们带来不同的$$B$$的结果，从而得到不同的拟牛顿法。

  实际中，为了避免对$B$求逆，定义$$H = B^{-1}$$, BFGS使用的是
  $$
  \min_H \| H - H_{k-1} \| \\
  s.t. H = H^{\top}, \ Hy_{k-1} = s_{k-1}
  $$
  这里仍然有几个有趣的问题值得深入思考：

  1. 为什么我们没有要求$$m_k(p)$$满足其他条件如$$m_k(-\alpha p_{k-1}) = f(x_{k-1})?$$

  2. 如果想引入更多的历史信息如$$x_{k-2}, x_{k-3}...$$，在上面的优化问题中要怎么加入？

     

+ 对于共轭梯度法：$$B = I$$, $$\beta$$的选取要使得$$p_k$$与$$p_{k-1}$$共轭，即$$p_i^TAp_j = 0, \ \forall i\neq j, \  p_i,p_j \in \{p_0, ..., p_k\} $$, 我们会在另一篇笔记里单独介绍与讨论共轭梯度法的独特性质（例如有一个有趣的点是：共轭梯度法对于clustered eigenvalues比uniformly distributed eigenvalues有更快的优化速度）与其在优化问题中的应用。

最后我们通过一个一维上的toy example理解梯度下降法、牛顿法、拟牛顿法对优化方向的选取：
$$
\min_x f(x)
$$




#### 2. 收敛性分析



#### 3. 源代码

```python
```

