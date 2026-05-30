import logging
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from loss.OrdinalCrossEntropyLoss import OrderedCrossEntropyLoss
from loss.hl_gauss import HLGaussLossFromSupport
from models import DLinear, Linear, NLinear
from utilss.tools import EarlyStopping, visual, adjust_learning_rate
from utilss.metrics import metric

import numpy as np
import pandas as pd
import torch
from torch import optim
import os
import time
import warnings

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def Mseloss(outputs, batch_y):
    return torch.mean((outputs - batch_y) ** 2)

def probs_to_quantile(probs, support, q=0.99, eps=1e-12):
    """
    probs:  [..., K]   (K = num_bins)
    support:[K+1]      bin edges
    return: [...,]     quantile value
    """
    # make sure device / dtype align
    support = support.to(device=probs.device, dtype=probs.dtype)

    # safety normalize (in case probs not perfectly sum to 1)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)

    cdf = probs.cumsum(dim=-1)  # [..., K]
    mask = cdf >= q

    # first index where cdf >= q
    idx = mask.to(torch.int64).argmax(dim=-1)  # [...]

    # handle rare case: all False (numerical issues) -> use last bin
    K = probs.size(-1)
    all_false = ~mask.any(dim=-1)
    idx = torch.where(all_false, torch.full_like(idx, K - 1), idx)  # [...]

    # bin edges for that idx
    left = support[idx]         # [...]
    right = support[idx + 1]    # [...]

    # cdf at idx and idx-1 (along last dim!)
    cdf_here = torch.gather(cdf, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [...]
    idxm1 = (idx - 1).clamp(min=0)
    cdf_prev = torch.gather(cdf, dim=-1, index=idxm1.unsqueeze(-1)).squeeze(-1)  # [...]
    cdf_left = torch.where(idx > 0, cdf_prev, torch.zeros_like(cdf_prev))  # [...]

    # linear interpolation inside bin
    denom = (cdf_here - cdf_left).clamp(min=eps)
    t = ((q - cdf_left) / denom).clamp(0.0, 1.0)

    return left + t * (right - left)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def add_gaussian_noise_with_snr(batch_x, snr_db, seed=None):
    x_np = batch_x.detach().cpu().numpy()
    signal_power = np.mean(x_np ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0, scale=np.sqrt(noise_power), size=x_np.shape)
    noisy_x = x_np + noise
    return torch.tensor(noisy_x, dtype=batch_x.dtype, device=batch_x.device)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.num_bins = args.num_bins
        #self.support = torch.linspace(-1, 1, self.num_bins + 1).to(self.device)
        self.support = torch.linspace(-1, 1, self.num_bins + 1).to(self.device)

        bin_w = (self.support[1] - self.support[0])
        self.support = self.support.clone()
        self.support[0]  -= 0.5 * bin_w
        self.support[-1] += 0.5 * bin_w
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.sigma = args.sigma

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        count_parameters(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return OrderedCrossEntropyLoss()

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        criterion = self._select_criterion()

        hlg_loss = HLGaussLossFromSupport(self.support, sigma=self.sigma).to(self.device)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                eps = 1e-8

                min_val = batch_y.min(axis=1, keepdims=True).values
                max_val = batch_y.max(axis=1, keepdims=True).values
                batch_y_norm = 2 * (batch_y - min_val) / (max_val - min_val + eps) - 1
                target_probs = hlg_loss.transform_to_probs(batch_y_norm)

                min_val = batch_x.min(axis=1, keepdims=True).values
                max_val = batch_x.max(axis=1, keepdims=True).values
                batch_x_norm = 2 * (batch_x - min_val) / (max_val - min_val + eps) - 1

                outputs = self.model(batch_x_norm)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target_probs = target_probs[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, target_probs)
                total_loss.append(loss.unsqueeze(0))

        return torch.cat(total_loss).mean().item()

    def train(self, setting, scheduler=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        hlg_loss = HLGaussLossFromSupport(self.support, sigma=self.sigma).to(self.device)

        print(">>> Start Training")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                eps = 1e-8
                min_val = batch_y.min(axis=1, keepdims=True).values
                max_val = batch_y.max(axis=1, keepdims=True).values
                batch_y_norm = 2 * (batch_y - min_val) / (max_val - min_val + eps) - 1
                target_probs = hlg_loss.transform_to_probs(batch_y_norm)

                min_val = batch_x.min(axis=1, keepdims=True).values
                max_val = batch_x.max(axis=1, keepdims=True).values
                batch_x_norm = 2 * (batch_x - min_val) / (max_val - min_val + eps) - 1

                outputs = self.model(batch_x_norm)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target_probs = target_probs[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, target_probs)
                train_loss.append(loss.unsqueeze(0))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if len(train_loss) > 0:
                train_loss = torch.cat(train_loss).mean().item()
            else:
                train_loss = 0.0

            if not self.args.train_only:
                vali_loss = self.vali(vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        # 1. 加载数据
        test_data, test_loader = self._get_data(flag='test')

        # 2. 如果需要，加载模型权重
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        
        # 结果保存路径
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        print(f"Start inference... Results will be saved to {folder_path}")

        # 3. 推理循环
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                eps = 1e-8

                # --- 归一化处理 (与训练保持一致) ---
                min_val_x = batch_x.min(axis=1, keepdims=True).values
                max_val_x = batch_x.max(axis=1, keepdims=True).values
                batch_x_norm = 2 * (batch_x - min_val_x) / (max_val_x - min_val_x + eps) - 1

                min_val_y = batch_y.min(axis=1, keepdims=True).values
                max_val_y = batch_y.max(axis=1, keepdims=True).values
                
                # --- 模型前向传播 ---
                outputs = self.model(batch_x_norm)
                
                # 处理特征维度 (如果是MS任务，通常取最后一维)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                # --- 核心：将预测分布还原为连续值 ---
                # 假设 hlg_loss 已经定义在类中或外部
                hlg_loss = HLGaussLossFromSupport(self.support, sigma=self.sigma).to(self.device)
                pred_continuous = hlg_loss.transform_from_probs(outputs)

                # --- 反归一化 ---
                pred_continuous = (pred_continuous + 1) * (max_val_y - min_val_y + eps) * 0.5 + min_val_y
                true = batch_y

                # 收集结果
                preds.append(pred_continuous.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                inputx.append(batch_x.detach().cpu().numpy())

                # --- 可视化 (每40个batch画一张图) ---

        # 4. 拼接所有 Batch 的结果
        # 假设 shape 为 [N, T] 或 [N, T, C]
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 确保形状一致，如果有多变量，这里可能需要 reshape 为 [N, -1] 来计算整体指标
        # 这里假设 metric 函数能处理多维数组，或者我们展平处理
        
        print(f"\nTotal Samples: {trues.shape[0]}")

        # ==========================================
        #  核心逻辑：按真实值大小分层评估 (Top %)
        # ==========================================
        
        # 1. 计算每个样本的"强度"。这里使用时间窗口内的最大值作为该样本的强度得分。
        #    如果数据是 [N, T, C]，我们取 max(T, C)
        N = trues.shape[0]
        sample_scores = trues.reshape(N, -1).max(axis=1) 
        
        # 2. 获取从大到小的排序索引
        sort_indices = np.argsort(sample_scores)[::-1] # 降序排列
        
        fractions = [0.25, 0.50, 0.75, 1.00]
        results_dict = {}

        print("\n" + "="*60)
        print(f"{'Data Subset':<20} | {'MSE':<10} | {'MAE':<10} | {'RMSE':<10} | {'MAPE':<10}")
        print("-" * 60)

        for frac in fractions:
            # 截取前 frac 比例的样本索引
            top_n = max(1, int(N * frac))
            current_indices = sort_indices[:top_n]
            
            # 提取对应的预测值和真实值
            p_sub = preds[current_indices]
            t_sub = trues[current_indices]
            
            # 计算指标
            mae, mse, rmse, mape, mspe, rse = metric(p_sub, t_sub)
            results_dict[frac] = (mse, mae, rmse, mape)

            label = f"Top {int(frac*100)}% (High Val)"
            print(f"{label:<20} | {mse:<10.4f} | {mae:<10.4f} | {rmse:<10.4f} | {mape:<10.4f}")

        print("="*60 + "\n")

        # ==========================================
        #  保存结果到 result.txt
        # ==========================================
        result_path = "result.txt"
        with open(result_path, 'a') as f:
            f.write(f"Setting: {setting}\n")
            
            # 写入 Overall (即 Top 100%)
            mse_all, mae_all, rmse_all, mape_all = results_dict[1.00]
            f.write(f"OVERALL | mse:{mse_all:.4f}, mae:{mae_all:.4f}, rmse:{rmse_all:.4f}, mape:{mape_all:.4f}\n")
            
            # 写入分层结果
            for frac in [0.25, 0.50, 0.75]:
                mse, mae, rmse, mape = results_dict[frac]
                f.write(f"TOP {int(frac*100)}%  | mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}\n")
            
            f.write("-" * 30 + "\n")

        return