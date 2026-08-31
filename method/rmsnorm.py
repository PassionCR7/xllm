import torch
import torch.nn as nn

#继承nn.Module 类
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim #维度
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim)) #公式的可学习参数γ，构建一个初始化全1、维度dim的参数
#_norm计算
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps) 
#实现forward
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)*x