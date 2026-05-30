import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from loss.hl_gauss import HLGaussLossFromSupport

class MultiViewPatternMemory(nn.Module):
    """
    [创新版] 双重调制的记忆模块
    Innovation: Output both 'Bias' (Additive) and 'Scale' (Multiplicative)
    """
    def __init__(self, input_len, output_dim, d_model=64, n_memory=64):
        super().__init__()
        self.n_memory = n_memory
        self.output_dim = output_dim
        
        # --- 1. 特征提取 (保持不变) ---
        self.time_encoder = nn.Linear(input_len, d_model)
        self.diff_encoder = nn.Linear(input_len, d_model)
        fft_len = input_len // 2 + 1
        self.freq_encoder = nn.Linear(fft_len, d_model)

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(), 
            nn.Linear(d_model, d_model)
        )
        
        # --- 2. 记忆库 ---
        self.memory_keys = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        
        # [创新点] Memory Values 维度翻倍: 前一半是 Bias，后一半是 Scale
        self.memory_values = nn.Parameter(torch.randn(n_memory, output_dim * 2) * 0.02)
        
        # --- 3. 门控 ---
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def get_prior_features(self, x):
        eps = 1e-6
        diff = x[:, 1:] - x[:, :-1]
        diff = F.pad(diff, (1, 0), mode='replicate') 
        diff = diff / (diff.std(dim=-1, keepdim=True) + eps)
        x_fft = torch.fft.rfft(x, dim=-1)
        freq_amp = torch.abs(x_fft)
        freq_amp = F.normalize(freq_amp, dim=-1, eps=eps)
        return diff, freq_amp

    def forward(self, x):
        # x: [B, L]
        
        # 1. 特征融合
        diff_view, freq_view = self.get_prior_features(x)
        emb_time = self.time_encoder(x)
        emb_diff = self.diff_encoder(diff_view)
        emb_freq = self.freq_encoder(freq_view)
        
        combined_feat = torch.cat([emb_time, emb_diff, emb_freq], dim=-1)
        query = self.fusion_layer(combined_feat) # [B, d]
        
        # 2. 检索
        query_norm = F.normalize(query, dim=1)
        keys_norm = F.normalize(self.memory_keys, dim=1)
        similarity = torch.matmul(query_norm, keys_norm.t())
        
        # Top-k Sparse Attention (保留最强的3个模式)
        top_k = 3
        topk_vals, topk_indices = torch.topk(similarity, top_k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
        attn_weights = F.softmax(mask * 2.0, dim=-1)
        
        # [B, Out*2]
        retrieved_content = torch.matmul(attn_weights, self.memory_values)
        
        # 3. 分离 Bias 和 Scale
        # Split: [B, Out], [B, Out]
        bias, raw_scale = torch.split(retrieved_content, self.output_dim, dim=-1)
        
        # 4. 处理 Scale
        # Tanh 将 scale 限制在 [-1, 1] 之间
        # 最终 scale 范围是 [0, 2]，即最大放大2倍，最小缩小到0
        scale = torch.tanh(raw_scale) + 1.0 
        
        gate = self.gate_net(query)
        
        # Gate 同时控制 Bias 和 Scale 的介入程度
        # 如果 Gate=0 (无记忆)，则 bias=0, scale=1.0 (无影响)
        final_bias = bias * gate
        final_scale = 1.0 + (scale - 1.0) * gate
        
        return final_bias, final_scale

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.sigma = configs.sigma

        self.decompsition = series_decomp(25)

        self.num_bins = configs.num_bins
        self.support = torch.linspace(-1, 1, self.num_bins + 1)
        bin_w = (self.support[1] - self.support[0])
        self.support = self.support.clone()
        self.support[0]  -= 0.5 * bin_w
        self.support[-1] += 0.5 * bin_w
        self.prob_transformer = HLGaussLossFromSupport(self.support, sigma=self.sigma)

        out_dim = self.pred_len * self.num_bins

        # ============================================================
        # 创新: 调制型记忆模块
        # ============================================================
        self.peak_memory = MultiViewPatternMemory(
            input_len=self.seq_len, 
            output_dim=out_dim, 
            d_model=64, 
            n_memory=64
        )

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, out_dim))
                self.Linear_Trend.append(nn.Linear(self.seq_len, out_dim))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, out_dim)
            self.Linear_Trend = nn.Linear(self.seq_len, out_dim)

    def forward(self, x):
        # x: [B, L, C]
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        B, C, L = seasonal_init.shape
        out_dim = self.pred_len * self.num_bins

        if self.individual:
            seasonal_output = torch.zeros([B, C, out_dim], dtype=seasonal_init.dtype).to(x.device)
            trend_output = torch.zeros([B, C, out_dim], dtype=trend_init.dtype).to(x.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 1. 基础线性预测 (Logits)
        linear_logits = seasonal_output + trend_output

        # 2. 记忆检索与调制
        x_input_reshaped = x.permute(0, 2, 1).reshape(B * C, L) 
        
        # [创新点] 获取 Bias (偏置) 和 Scale (增益)
        memory_bias, memory_scale = self.peak_memory(x_input_reshaped)
        
        memory_bias = memory_bias.view(B, C, out_dim)
        memory_scale = memory_scale.view(B, C, out_dim)

        # 3. 创新融合逻辑: 调制 (Modulation)
        # 原始: Final = Linear + Bias
        # 现在: Final = Linear * Scale + Bias
        # 这允许记忆模块“放大”线性预测的置信度，这对于高值预测至关重要
        x_final = linear_logits * memory_scale + memory_bias
        
        # 4. 输出
        x_final = x_final.view(B, C, self.pred_len, self.num_bins).permute(0, 2, 1, 3)
        prob = x_final.softmax(dim=-1)
        
        return prob