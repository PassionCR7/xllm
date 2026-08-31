def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim=x.shape
    if n_rep==1:
        return x

    return (
        x[:,:,:,None,:]
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
        .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args:XLLMConfig):
        super().__init__()
        self.num_key_value_heads=(
            args.num_attention_heads 
            if args.num_key_value_heads is None 
            else args.num_key_value_heads
            )

        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_attention_heads must be divisible by n_key_value_heads"

        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=args.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #投影，计算q，k，v
        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        #把输入拆分成多个头，用view
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        #对q和k使用RoPE
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)
        #对k，v使用repeat（注意kv cache）
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None

        xq,xk,xv=(
            xq.transpose(1,2),
            #[bsz,n_local_heads,seq_len,head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2),
        )
        #进行attention注意力分数计算
        if (
            self.flash  
            and (seq_len>1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask==1))
            ):
            attn_mask=(
                None
                if attention_mask is None
                else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
            )
            output=F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                )
        else:
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            #只在最新输入的seq_len长度部分施加causal mask，兼容kv cache
            scores=[:,:,:,-seq_len:]+=torch.triu(
                torch.full((seq_len,seq_len),float("-inf"),device=scores.device),
                diagonal=1,
                )
            # scores=scores+torch.triu(
            #     torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
            #     diagonal=1,
            # ).unsqueeze(0).unsqueeze(0)
        #最后拼接头，输出投影，返回
            if attention_mask is not None:
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1.0-extended_attention_mask)*-1e9
                scores=scores+extended_attention_mask

            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv
        #[bsz,n_local_heads,seq_len,head_dim]
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.resid_dropout(self.o_proj(output))
        return output,past_kv