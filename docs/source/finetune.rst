Finetune
=============

.. _finetune:


Pointwise
--------------

arcface
- 分层学习率
- batch size影响大
- arcface_margin动态调整, margin大小影响较大
- arc_weight初始化
- 含状态训练的损失函数不适合每个epoch训练时也过一遍评价指标
