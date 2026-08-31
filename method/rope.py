def precompute_freqs_cis(
        dim:int,
        end:int=int(32*1024),
        rope_base=1e6,
        rope_scaling:Optional[dict]=None,
):
    #初始化RoPE频率
    freqs,attn_factor=(1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0)

    if rope_scaling is not None:
        # 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
            )
        #推理长度大于训练长度就使用缩放
        if end>orig_max:
           #波长b到i的映射
           inv_dim=lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
           #划分高低维
           # low为不需要缩放的高频部分
           # high为需要缩放的低频部分
           low,high=(max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))
           #计算缩放因子
           #low不需要缩放，ramp为0，需要缩放的high，ramp为1，两者之间先行过度
           ramp=torch.clamp(
               (torch.arange(dim//2,device=freqs.device).float()-low)
               / max(hirgh-low, 0.001),
               0,
               1,
           )
           #ramp=0,高频系数是0，保持不变；ramp=1，低频系数是1/factor，对频率进行线性插值缩放。
           freqs=freqs*(1-ramp+ramp/factor)
        #根据end，生成位置索引t
        t=torch.arange(end,device=freqs.device).float()
        #计算外积，将t和频率部分相乘，得到每个位置的旋转角度
        freqs=torch.outer(t,freqs).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin
#编写RoPE
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x.shape[-1]取最后一个维度的重点
        #(x[...,x.shape[-1]//2:]取出x的后半部分
        return torch.cat(
            (-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1
        )
    # x_rotated=x*cos+rotate_half(x)*sin
    q_embed=(q*cos.unsqueeze(unsqueeze_dim)+rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed=(k*cos.unsqueeze(unsqueeze_dim)+rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed,k_embed