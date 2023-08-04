---
layout: post
title:  "(Chinese) 数值最优化笔记(1)：从统一的角度认识梯度下降、牛顿法与拟牛顿法"
subtitle: "Note of Numerical Optimization: A Unified View of Gradient Descent, Newton Method & Quasi-Newton Method"
tag: "math" 
---

数值最优化方法一般可以分为线搜索（line search）与信赖域（trust region）两类，在无约束优化的线搜索方法中，最主流的方法有：梯度下降、牛顿法、拟牛顿法与共轭梯度法。在本文中，我们将努力回答以下问题：

+ 如何从统一的角度理解这四种看似不同的方法？
+ 前三种方法的收敛性，以及到底是什么决定了它们在靠近解时的收敛性？



#### 1. 一个统一的角度

我们知道，对于线搜索方法，优化的每一步最重要的东西有两个：优化的方向和步长。我们假设优化问题的形式为$$\min_{x} f(x)$$, 从统一的角度来看，梯度下降法、牛顿法、拟牛顿法与共轭梯度法在优化的第$k$步选取的优化方向可以被概括为以下形式：

$$
p_k = -B_k^{-1}\nabla f_k + \beta_k p_{k-1}
$$

在梯度下降、牛顿法与拟牛顿法中，$$\beta = 0$$。

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

  3. $$\nabla m_k(-\alpha p_{k-1}) = \nabla f(x_{k-1})$$, 

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

  选取不同的$$\|\cdot \|$$会给我们带来不同的$$B$$的结果，从而得到不同的拟牛顿法。

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

+ 梯度下降：选取负梯度方向

+ 牛顿法：选一个抛物线，要求过$x_k$，切线与$x_k$处切线一致，并且这个抛物线的碗底所在的横坐标为$$x_k - [\nabla^2 f]^{-1}\nabla f_k$$

+ 拟牛顿法：选一个抛物线，要求过$x_k$，$x_k$处切线与$\nabla f(x_k)$一致，并且$x_{k-1}$处切线也要与$$\nabla f(x_{k-1})$$一致。

  

#### 2. 梯度下降、牛顿法与拟牛顿法的收敛性分析

##### 2.1 一般理论

对于这三种方法收敛性分析，我们都假设当前点$x^k$在局部最优值点$x^*$附近的一个邻域内。

先说结论：牛顿法（二阶收敛速度）> 拟牛顿法（超线性收敛速度）> 梯度下降法（线性收敛速度）

1. 牛顿法 （Theorem 3.5）

+ 前提假设：假设$f$二阶可导，$$\nabla^2 f$$ Lipschitz连续，下标k和\*分别表示在 $x_k, x^*  $ 处

+ 证明：

    $$
    \begin{align}
    x_k + p_k^N - x^* &= x_k - x^* - \nabla^2 f_k^{-1} \nabla f_k \\
    &= \nabla^2 f_k^{-1}(\nabla^2 f_k (x_k - x^*) - \nabla f_k ) \\
    &= \nabla^2 f_k^{-1}(\nabla^2 f_k (x_k - x^*) - (\nabla f_k - \nabla f_* ))
    \end{align}
    $$

    想把上面的式子变成关于$(x_k - x^*)$的：

    利用泰勒公式
    $$
    \nabla f_k - \nabla f_* = \int_{0}^1 \nabla^2 f(x^* + t(x_k -x^*))(x_k - x^*) \text{d}t
    $$
    然后两边取模，利用Lipschitz连续

    <img src="/home/user/.config/Typora/typora-user-images/image-20230804100214709.png" alt="image-20230804100214709" />

    可以通过让$x_k$离$x^*$比较近把上面的$\frac{1}{2}$也去掉，那么：
    $$
    \begin{align}
    \|x_k + p^N_k - x^* \| &\leq L\|\nabla^2  f(x^*)^{-1} \| \| x_k - x^*\|^2 \\ & = \widetilde{L}  \| x_k - x^*\|^2 \\
    \end{align}
    $$
    即
    $$
    \frac{\|x_{k+1} - x^* \|}{\|x_k - x^* \|^2} = c 
    $$
    所以为二阶收敛。



2. 拟牛顿法（Theorem 3.6）

+ 前提假设：
  $$
  \lim_{k\rightarrow \infty} \frac{\|(B_k - \nabla^2 f(x^*))p_k \|}{\| p_k\|} = 0 \tag{1}
  $$
  即$$\lim_{k \rightarrow \infty} p_k = -B_k^{-1} \nabla f_k = -[\nabla^2 f_k]^{-1} \nabla f_k = p_k^N$$, 即拟牛顿法给出的搜索方向要在极限意义上逼近牛顿法的搜索方向

+ 证明：
  $$
  \begin{align}
  p_k - p_k^N &= p_k + [\nabla^2 f_k]^{-1} \nabla f_k \\
  &= \nabla^2 f_k^{-1} (\nabla^2 f_k p_k + \nabla f_k) \\
  &= \nabla^2 f_k^{-1}(\nabla^2 f_k p_k - B_k p_k) \\
  &= \nabla^2 f_k^{-1}(\nabla^2 f_k - B_k) p_k \\
  &= O(\|(\nabla^2 f_k - B_k)p_k \|) \\
  &= o(\|p_k \|) （根据前提）
  \end{align}
  $$
  所以：
  $$
  \begin{align} 
  \|x_k + p_k - x^* \| &\leq \|x_k + p_k^N - x^* + (p_k - p_k^N) \| \\
  &= \| x_k + p_k^N - x^* \| + \| p_k - p_k^N\| \\
  \end{align}
  $$
  利用牛顿法的结论：
  $$
  \| x_k + p_k^N - x^* \| = O(\|x_k -x^* \|^2)
  $$

  代入之前的式子：
  $$
  \begin{align} 
  \|x_k + p_k - x^* \| &\leq  O(\| x_k - x^* \|^2) + \| p_k - p_k^N\| \\
  &= O(\| x_k - x^* \|^2) + o(\|p_k \|)
  \end{align}
  $$
  
  又有$$\|p_k \| = O(\|x_k - x^* \|)$$
  
  所以：
  $$
  \|x_{k+1} - x^* \| = \|x_k + p_k - x^*\| \leq o(\|x_k - x^* \|)
  $$
  同时：
  $$
  \begin{align}
  \| x_k + p_k - x^*\| &\geq \|x_k + p_k^N - x^* \| - \| p_k - p_k^N \| \\
  &= O(\|x_k - x^* \|^2) - o(\|p_k \|)
  
  \end{align}
  $$
  即有超线性收敛速度。
  
  **事实上，公式(1)是拟牛顿法具有超线性收敛速度的充要条件。**
  
  

但是问题来了，是否所有的拟牛顿法都能够满足$$\lim_{k\rightarrow \infty} \frac{\|(B_k - \nabla^2 f(x^*))p_k \|}{\| p_k\|} = 0$$呢？以及这些方法中极限的收敛速度也会影响它们的优化速度。下面我们以经典的BFGS方法来进行分析：

##### 2.2 BFGS的收敛性分析



#### 3. 总结

回到我们最初的问题：

+ 如何从统一的角度理解这四种看似不同的方法？

  一个统一的公式：$$p_k = -B_k^{-1}\nabla f_k + \beta_k p_{k-1}$$

  区别在于是否估计Hessian

+ 前三种方法的收敛性，以及到底是什么决定了它们在靠近解时的收敛性？

  与牛顿方向的接近程度决定了它们在靠近解的收敛性

  

