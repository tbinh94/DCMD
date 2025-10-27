import argparse
import os
from math import prod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models.stsae.stsae import STSAE
from sklearn.metrics import roc_curve, roc_auc_score
from torch.optim import Adam, AdamW
from tqdm import tqdm

from utils.diffusion_utils import Diffusion
from utils.eval_utils import (compute_fig_matrix, compute_var_matrix, filter_vectors_by_cond,
                              get_avenue_mask, get_hr_ubnormal_mask, pad_scores, score_process)
from utils.model_utils import processing_data, my_kl_loss
from models.transformer import MotionTransformer
from utils.tools import get_dct_matrix, generate_pad, padding_traj



class DCMD(pl.LightningModule):
    
    losses = {'l1':nn.L1Loss, 'smooth_l1':nn.SmoothL1Loss, 'mse':nn.MSELoss}
    conditioning_strategies = {'inject':'inject'}
    def __init__(self, args: argparse.Namespace, steps_per_epoch: int = None) -> None:
        """
        This class implements DCMD model.
        
        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model
        """

        super(DCMD, self).__init__()

        ## Log the hyperparameters of the model
        self.save_hyperparameters(args)

        # cho optimizer
        self.hparams.steps_per_epoch = steps_per_epoch

        ## Set the internal variables of the model
        # Data parameters
        self.n_frames = args.seg_len
        self.num_coords = args.num_coords
        self.n_joints = self._infer_number_of_joint(args)

        ## Model parameters
        # Main network
        self.dropout = args.dropout
        self.conditioning_strategy = self.conditioning_strategies[args.conditioning_strategy]
        # Conditioning network
        self.cond_h_dim = args.h_dim
        self.cond_latent_dim = args.latent_dim
        self.cond_channels = args.channels
        self.cond_dropout = args.dropout

        ## Training and inference parameters
        self.learning_rate = args.opt_lr
        self.loss_fn = self.losses[args.loss_fn](reduction='none')
        self.noise_steps = args.noise_steps
        self.aggregation_strategy = args.aggregation_strategy
        self.n_generated_samples = args.n_generated_samples
        self.model_return_value = args.model_return_value
        self.gt_path = args.gt_path
        self.split = args.split
        self.use_hr = args.use_hr
        self.ckpt_dir = args.ckpt_dir
        self.save_tensors = args.save_tensors
        self.num_transforms = args.num_transform
        self.anomaly_score_pad_size = args.pad_size
        self.anomaly_score_filter_kernel_size = args.filter_kernel_size
        self.anomaly_score_frames_shift = args.frames_shift
        self.dataset_name = args.dataset_choice

        # New parameters
        self.n_his = args.n_his
        self.padding = args.padding
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.latent_dims = args.latent_dims
        self.automatic_optimization = False
        self.loss_1_series_weight = args.loss_1_series_weight
        self.loss_1_prior_weight = args.loss_1_prior_weight
        self.loss_2_series_weight = args.loss_2_series_weight
        self.loss_2_prior_weight = args.loss_2_prior_weight
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.n_his, self.n_frames-self.n_his)

        ## Set the noise scheduler for the diffusion process
        self._set_diffusion_variables()
        
        ## Build the model
        self.build_model()
        
    
    def build_model(self) -> None:
        """
        Build the model according to the specified hyperparameters.
        """

        # Prediction Model
        pre_model = MotionTransformer(
            input_feats=2 * self.n_joints,
            num_frames=self.n_frames,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            latent_dim=self.latent_dims,
            dropout=self.dropout,
            device=self.device,
            inject_condition=(self.conditioning_strategy == 'inject')
        )

        # Reconstruction Model
        rec_model = STSAE(
            c_in=self.num_coords,
            h_dim=self.cond_h_dim,
            latent_dim=self.cond_latent_dim,
            n_frames=self.n_his,
            dropout=self.cond_dropout,
            n_joints=self.n_joints,
            layer_channels=self.cond_channels,
            device=self.device)

        self.pre_model, self.rec_model = pre_model, rec_model

    # Hàm Loss Thông minh hơn
    def compute_motion_loss(self, predicted_x0, true_x0):
        """
        Tính toán loss kết hợp bao gồm pose, velocity và acceleration.
        x_pred và x_true có shape (B, T, D_pose)
        """
        # 1. Pose Loss (giống loss L1/L2 cũ)
        loss_pose = torch.mean(self.loss_fn(predicted_x0, true_x0))

        # Chuyển đổi shape để tính toán động học
        # (B, T, C*V) -> (B, T, V, C)
        b, t, _ = predicted_x0.shape
        v = self.n_joints
        c = self.num_coords
        
        predicted_x0_reshaped = predicted_x0.view(b, t, v, c)
        true_x0_reshaped = true_x0.view(b, t, v, c)

        # 2. Velocity Loss
        velocity_pred = predicted_x0_reshaped[:, 1:, :, :] - predicted_x0_reshaped[:, :-1, :, :]
        velocity_true = true_x0_reshaped[:, 1:, :, :] - true_x0_reshaped[:, :-1, :, :]
        loss_velocity = torch.mean(self.loss_fn(velocity_pred, velocity_true))

        # 3. Acceleration Loss
        accel_pred = velocity_pred[:, 1:, :, :] - velocity_pred[:, :-1, :, :]
        accel_true = velocity_true[:, 1:, :, :] - velocity_true[:, :-1, :, :]
        loss_accel = torch.mean(self.loss_fn(accel_pred, accel_true))
        
        # 4. Lấy trọng số từ hparams (cần thêm vào file config)
        w_vel = self.hparams.w_vel if hasattr(self.hparams, 'w_vel') else 0.0
        w_accel = self.hparams.w_accel if hasattr(self.hparams, 'w_accel') else 0.0

        total_loss = loss_pose + w_vel * loss_velocity + w_accel * loss_accel
        
        # Log các thành phần loss để theo dõi
        self.log('loss_pose', loss_pose, on_step=True, on_epoch=True, sync_dist=True)
        self.log('loss_velocity', loss_velocity, on_step=True, on_epoch=True, sync_dist=True)
        self.log('loss_accel', loss_accel, on_step=True, on_epoch=True, sync_dist=True)
        
        return total_loss
        

    def forward(self, input_data:List[torch.Tensor], aggr_strategy:str=None, return_:str=None) -> List[torch.Tensor]:
        """
        Forward pass of the model.
        """

        ## Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, meta_out = self._unpack_data(input_data)
        B = tensor_data.shape[0]

        ## Select frames to reconstruct and to predict
        history_data = tensor_data[:, :, :self.n_his, :]
        x_0 = padding_traj(history_data, self.padding, self.idx_pad, self.zero_index)

        generated_xs = []
        # Generate m future predictions
        for _ in range(self.n_generated_samples):

            ## Reconstruction —— AE model
            condition_embedding, rec_his_data = self.rec_model(history_data)

            ## Prediction —— diffusion model
            ## DCT transformation
            dct_m, idct_m = get_dct_matrix(self.n_frames)
            dct_m_all = dct_m.float().to(self.device)
            idct_m_all = idct_m.float().to(self.device)
            # (B, C, T, V) -> (B, T, V, C)
            x = x_0.permute(0, 2, 3, 1).contiguous()
            # (B, T, V, C) -> (B, T, C*V)
            x = x.reshape([x.shape[0], self.n_frames, -1])
            y = torch.matmul(dct_m_all, x)  # [B, T, C*V]

            ## Generate gaussian noise of the same shape as the y
            y_d = torch.randn_like(y, device=self.device)

            ## (t ∈ T, T-1, ..., 1)
            for i in reversed(range(1, self.noise_steps)):

                ### Prediction (Two branches)
                ## Set the time step
                t = torch.full(size=(B,), fill_value=i, dtype=torch.long, device=self.device)
                t_prev = torch.full(size=(B,), fill_value=i, dtype=torch.long, device=self.device)
                t_prev[0] = 0

                ## Generate gaussian noise of the same shape as the predicted noise
                noise_pre = torch.randn_like(y_d, device=self.device) if i > 1 else torch.zeros_like(y_d, device=self.device)

                ## First branch
                # Predict the noise
                predicted_noise_pre, _, _, _ = self.pre_model(y_d, t, condition_data=condition_embedding)
                # Get the alpha and beta values and expand them to the shape of the predicted noise
                alpha_pre = self._alpha[t][:, None, None]
                alpha_hat_pre = self._alpha_hat[t][:, None, None]
                beta_pre = self._beta[t][:, None, None]
                # Recover the predicted sequence
                y_d = (1 / torch.sqrt(alpha_pre)) * (y_d - ((1 - alpha_pre) / (torch.sqrt(1 - alpha_hat_pre))) * predicted_noise_pre) \
                    + torch.sqrt(beta_pre) * noise_pre
                ## Second branch
                alpha_hat_prev = self._alpha_hat[t_prev][:, None, None]
                # Add noise
                y_n = (torch.sqrt(alpha_hat_prev) * y) + (torch.sqrt(1 - alpha_hat_prev) * noise_pre)
                ## Mask completion
                # Get M values
                mask = torch.zeros_like(x, device=self.device) # [batch, T, C*V]
                for m in range(0, self.n_his):
                    mask[:, m, :] = 1
                # iDCT transformation
                y_d_idct = torch.matmul(idct_m_all, y_d)
                y_n_idct = torch.matmul(idct_m_all, y_n)
                # mask-mul
                m_mul_y_n = torch.mul(mask, y_n_idct)
                m_mul_y_d = torch.mul((1-mask), y_d_idct)
                # together
                m_y = m_mul_y_d + m_mul_y_n
                # DCT again
                y_d = torch.matmul(dct_m_all, m_y)

            # iDCT
            pre_future_data = torch.matmul(idct_m_all, y_d)
            # (B, T, C*V) -> (B, T, V, C)
            pre_future_data = pre_future_data.reshape(pre_future_data.shape[0], pre_future_data.shape[1], -1, 2)
            # (B, T, V, C) -> (B, C, T, V)
            pre_future_data = pre_future_data.permute(0, 3, 1, 2).contiguous()
            # select future sequences
            pre_future_data = pre_future_data[:,:,self.n_his:,:]

            ## Reconstruction + Prediction
            xs = torch.cat((rec_his_data, pre_future_data), dim=2)

            generated_xs.append(xs)

        selected_x, loss_of_selected_x = self._aggregation_strategy(generated_xs, tensor_data, aggr_strategy)
        
        return self._pack_out_data(selected_x, loss_of_selected_x, [tensor_data] + meta_out, return_=return_)
        
        
    def training_step(self, batch:List[torch.Tensor], batch_idx:int) -> torch.float32:
        """
        Training step of the model.
        """

        ## Get the optimizer returned in configuration_optimizers()
        opt = self.optimizers()

        ## Unpack data: tensor_data is the input data
        tensor_data, _ = self._unpack_data(batch)

        ## Select frames to reconstruct and to predict
        history_data = tensor_data[:,:,:self.n_his,:] # Used for rec (first n_his)
        x_0 = tensor_data # Used for pre（all）

        ## Reconstruction
        # Encode the history data
        condition_embedding, rec_his_data = self.rec_model(history_data)
        # Compute the rec_loss
        rec_loss = torch.mean(self.loss_fn(rec_his_data, history_data))
        self.log('rec_loss',rec_loss)

        ## Prediction
        # DCT transformation
        dct_m, _ = get_dct_matrix(self.n_frames)
        dct_m_all = dct_m.float().to(self.device)
        # (B, C, T, V) -> (B, T, V, C)
        x = x_0.permute(0, 2, 3, 1).contiguous()
        # (B, T, V, C) -> (B, T, C*V)
        x = x.reshape([x.shape[0], self.n_frames, -1]) # [batch, T, C*V]
        y_0 = torch.matmul(dct_m_all, x)

        # Sample the time steps and corrupt the data
        t = self.noise_scheduler.sample_timesteps(y_0.shape[0]).to(self.device)
        y_t, pre_noise = self.noise_scheduler.noise_motion(y_0, t) # (B, T, C*(V-1))

        # Predict the noise
        pre_predicted_noise, series, prior, _ = self.pre_model(y_t, t, condition_data=condition_embedding)



        # TÍNH PRE_LOSS BẰNG HÀM LOSS ĐỘNG HỌC ( thêm sau khi có kết quả 0.86 - alpha 0.3 - 20epoch)
        # a. Khôi phục y0_pred từ nhiễu dự đoán
        alpha_hat_t = self._alpha_hat[t][:, None, None]
        predicted_y0 = (y_t - torch.sqrt(1.0 - alpha_hat_t) * pre_predicted_noise) / torch.sqrt(alpha_hat_t)
        # b. Tính loss trên dữ liệu đã khôi phục
        pre_loss = self.compute_motion_loss(predicted_y0, y_0) # Gọi hàm helper mới
        self.log('pre_loss', pre_loss)




        # Compute the pre_loss
        # Calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            # Pdetach, S <-> Maximize
            series_loss += \
                (torch.mean(my_kl_loss(
                    series[u],
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.n_frames)).detach()))
                + torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.n_frames)).detach(),
                    series[u])))
            # P, Sdetach <-> Minimize
            prior_loss += \
                (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.n_frames)),
                    series[u].detach()))
                + torch.mean(my_kl_loss(
                    series[u].detach(),
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.n_frames)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        # Tạm không dùng pre_loss cũ mà dùng hàm loss thông minh hơn ( DCMD improvement 2)
        #pre_loss = torch.mean(self.loss_fn(pre_predicted_noise, pre_noise))
        #self.log('pre_loss', pre_loss)

        ## Compute loss1 & loss2
        loss1 = rec_loss + pre_loss \
                - self.loss_1_series_weight * series_loss \
                + self.loss_1_prior_weight * prior_loss
        self.log('loss1', loss1)
        loss2 = rec_loss + pre_loss \
                + self.loss_2_prior_weight * prior_loss \
                + self.loss_2_series_weight * series_loss
        self.log('loss2', loss2)

        ## Minimax strategy
        self.manual_backward(loss1, retain_graph=True)
        self.manual_backward(loss2)
        opt.step()
        opt.zero_grad()


    def test_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        Test step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """

        self._test_output_list.append(self.forward(batch))
        return
    
    
    def on_test_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """

        super().on_test_epoch_start()
        self._test_output_list = []
        return
    
    
    def on_test_epoch_end(self) -> float:
        """
        Test epoch end of the model.

        Returns:
            float: test auc score
        """

        out, gt_data, trans, meta, frames = processing_data(self._test_output_list)
        del self._test_output_list
        if self.save_tensors:
            tensors = {'prediction':out, 'gt_data':gt_data, 
                       'trans':trans, 'metadata':meta, 'frames':frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy, n_gen=self.n_generated_samples)
        auc_score = self.post_processing(out, gt_data, trans, meta, frames)
        self.log('AUC', auc_score)
        return auc_score
    
    
    def validation_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        Validation step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        
        # 1. Thực hiện forward pass để có output
        output_data = self.forward(batch)
        
        # 2. Trích xuất và tính loss trung bình cho batch này
        # Dựa vào _pack_out_data, loss_of_selected_x là phần tử đầu tiên 
        # nếu model_return_value là 'all' hoặc 'loss'
        loss_tensor = output_data[0] # Đây là tensor loss có shape (batch_size,)
        val_loss = loss_tensor.mean()

        # 3. Log giá trị val_loss. 
        # prog_bar=True sẽ hiển thị nó trên thanh tiến trình của epoch.
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # 4. Lưu lại toàn bộ output để on_validation_epoch_end xử lý
        self._validation_output_list.append(output_data)
        return
    
    
    def on_validation_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """
        
        super().on_validation_epoch_start()
        self._validation_output_list = []
        return


    def on_validation_epoch_end(self) -> float:
        """
        Validation epoch end of the model.

        Returns:
            float: validation auc score
        """
        
        out, gt_data, trans, meta, frames = processing_data(self._validation_output_list)
        del self._validation_output_list
        if self.save_tensors:
            tensors = {'prediction':out, 'gt_data':gt_data, 
                       'trans':trans, 'metadata':meta, 'frames':frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy, n_gen=self.n_generated_samples)
        auc_score =  self.post_processing(out, gt_data, trans, meta, frames)
        self.log('AUC', auc_score, sync_dist=True)
        return auc_score

    
    def configure_optimizers(self) -> Dict:
        """
        Configure the optimizers and the learning rate schedulers.

        Returns:
            Dict: dictionary containing the optimizers, the learning rate schedulers and the metric to monitor
        """
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'AUC'}
        """

        """
        # Lấy siêu tham số từ một nơi duy nhất: self.hparams
        lr = self.hparams.opt_lr
        epochs = self.hparams.n_epochs
        steps = self.hparams.steps_per_epoch

        # Đảm bảo các giá trị cần thiết tồn tại
        if lr is None or epochs is None or steps is None:
            raise ValueError("opt_lr, n_epochs, and steps_per_epoch must be present in hparams.")

        
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps
        )
        """

        lr = self.hparams.opt_lr
        epochs = self.hparams.n_epochs
        
        # Sử dụng AdamW thay vì Adam để cải thiện hiệu suất        
        optimizer = AdamW(self.parameters(), lr=lr)
        
        # --- TINH CHỈNH SCHEDULER ĐỂ HỢP TÁC VỚI SWA ---
        # Đặt T_max để scheduler hoàn thành ngay trước khi SWA bắt đầu (khoảng 80% tổng thời gian)
        swa_start_epoch = int(epochs * 0.8)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=swa_start_epoch,  # THAY ĐỔI Ở ĐÂY: từ total_epochs thành swa_start_epoch
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # <--- dùng step nếu là OneCycleLR
                "frequency": 1,
                "monitor": "val_loss" # Giữ lại monitor để ModelCheckpoint hoạt động tốt
            },
        }
        


    def post_processing(self, out:np.ndarray, gt_data:np.ndarray, trans:np.ndarray, meta:np.ndarray, frames:np.ndarray) -> float:
        """
        Post processing of the model.

        Args:
            out (np.ndarray): output of the model
            gt_data (np.ndarray): ground truth data
            trans (np.ndarray): transformation index
            meta (np.ndarray): metadata
            frames (np.ndarray): frame indexes of the data

        Returns:
            float: auc score
        """
        
        all_gts = [file_name for file_name in os.listdir(self.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]
        hr_ubnormal_masked_clips = get_hr_ubnormal_mask(self.split) if (self.use_hr and (self.dataset_name == 'UBnormal')) else {}
        hr_avenue_masked_clips = get_avenue_mask() if self.dataset_name == 'HR-Avenue' else {}

        num_transform = self.num_transforms
        model_scores_transf = {}
        dataset_gt_transf = {}


        for transformation in tqdm(range(num_transform)):
            # iterating over each transformation T

            dataset_gt = []
            model_scores = []
            cond_transform = (trans == transformation)

            out_transform, gt_data_transform, meta_transform, frames_transform = filter_vectors_by_cond([out, gt_data, meta, frames], cond_transform)

            for idx in range(len(all_gts)):
                # iterating over each clip C with transformation T

                scene_idx, clip_idx = scene_clips[idx]

                gt = np.load(os.path.join(self.gt_path, all_gts[idx]))
                n_frames = gt.shape[0]
                
                cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)
                out_scene_clip, gt_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([out_transform, gt_data_transform, 
                                                                                                           meta_transform, frames_transform], 
                                                                                                           cond_scene_clip)
                # Nếu không tìm thấy dữ liệu cho clip này (vì limit_test_batches), hãy bỏ qua
                if meta_scene_clip.shape[0] == 0:
                    continue

                figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
                error_per_person = []
                error_per_person_max_loss = []

                for fig in figs_ids:
                    # iterating over each actor A in each clip C with transformation T

                    cond_fig = (meta_scene_clip[:, 2] == fig)
                    # 1. Lấy ra cả out_fig, gt_fig và frames_fig
                    out_fig, gt_fig, frames_fig = filter_vectors_by_cond([out_scene_clip, gt_scene_clip, frames_scene_clip], cond_fig)

                    # 2. Tạo ma trận pose cho cả dữ liệu dự đoán và dữ liệu thật
                    pose_matrix_out = compute_fig_matrix(out_fig, frames_fig, n_frames)
                    pose_matrix_gt = compute_fig_matrix(gt_fig, frames_fig, n_frames)

                    # 3. Tính toán loss thực sự. 
                    # Chúng ta tính sai số Euclidean giữa pose dự đoán và pose thật cho mỗi frame
                    # Shape kết quả: (số_lần_lấy_mẫu, số_frame)
                    frame_wise_error = np.linalg.norm(pose_matrix_out - pose_matrix_gt, axis=-1)

                    # 4. Lấy loss lớn nhất trên các mẫu cho mỗi frame (giống logic cũ)
                    fig_reconstruction_loss = np.nanmax(frame_wise_error, axis=0)
                    
                    if self.anomaly_score_pad_size != -1:
                        fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.anomaly_score_pad_size)                 
                    
                    error_per_person.append(fig_reconstruction_loss)
                    # Dòng này cũng cần sửa vì fig_reconstruction_loss là một array, không phải số
                    error_per_person_max_loss.append(np.max(fig_reconstruction_loss))

                clip_score = np.stack(error_per_person, axis=0)
                clip_score_log = np.log1p(clip_score)
                clip_score = np.mean(clip_score, axis=0) + (np.amax(clip_score_log, axis=0)-np.amin(clip_score_log, axis=0))

                # removing the non-HR frames for UBnormal dataset
                if (scene_idx, clip_idx) in hr_ubnormal_masked_clips:
                    clip_score = clip_score[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                    gt = gt[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                
                # removing the non-HR frames for Avenue dataset
                if clip_idx in hr_avenue_masked_clips:
                    clip_score = clip_score[np.array(hr_avenue_masked_clips[clip_idx])==1]
                    gt = gt[np.array(hr_avenue_masked_clips[clip_idx])==1]

                # Abnormal score per frame
                clip_score = score_process(clip_score, self.anomaly_score_frames_shift, self.anomaly_score_filter_kernel_size)
                model_scores.append(clip_score)

                dataset_gt.append(gt)
                    
             # Nếu không có clip nào được xử lý cho transformation này, bỏ qua
            if not model_scores:
                continue

            model_scores = np.concatenate(model_scores, axis=0)
            dataset_gt = np.concatenate(dataset_gt, axis=0)

            model_scores_transf[transformation] = model_scores
            dataset_gt_transf[transformation] = dataset_gt

        # aggregating the anomaly scores for all transformations
        # aggregating the anomaly scores for all transformations
        pds = np.mean(np.stack(list(model_scores_transf.values()),0),0)
        gt = dataset_gt_transf[0]

        # computing the AUC
        try:
            auc = roc_auc_score(gt, pds)
        except ValueError as e:
            # Kiểm tra xem có đúng là lỗi "only one class" không
            if "Only one class present in y_true" in str(e):
                print("\nCảnh báo: Chỉ tìm thấy một lớp trong dữ liệu ground truth (dự kiến khi chạy limit_test_batches). Không thể tính AUC. Tạm thời trả về 0.5.")
                # Trả về 0.5 là một quy ước phổ biến, vì đó là điểm AUC của một bộ phân loại ngẫu nhiên.
                auc = 0.5 
            else:
                # Nếu là một ValueError khác, hãy ném lại lỗi để chúng ta biết
                raise e

        return auc
    
    
    def test_on_saved_tensors(self, split_name:str) -> float:
        """
        Skip the prediction step and test the model on the saved tensors.

        Args:
            split_name (str): split name (val, test)

        Returns:
            float: auc score
        """
        
        tensors = self._load_tensors(split_name, self.aggregation_strategy, self.n_generated_samples)
        auc_score = self.post_processing(tensors['prediction'], tensors['gt_data'], tensors['trans'],
                                         tensors['metadata'], tensors['frames'])
        print(f'AUC score: {auc_score:.6f}')
        return auc_score
        
    
    
    ## Helper functions
    
    def _aggregation_strategy(self, generated_xs:List[torch.Tensor], input_sequence:torch.Tensor, aggr_strategy:str) -> Tuple[torch.Tensor]:
        """
        Aggregates the generated samples and returns the selected one and its reconstruction error.
        Strategies:
            - all: returns all the generated samples
            - random: returns a random sample
            - best: returns the sample with the lowest reconstruction loss
            - worst: returns the sample with the highest reconstruction loss
            - mean: returns the mean of the losses of the generated samples
            - median: returns the median of the losses of the generated samples
            - mean_pose: returns the mean of the generated samples
            - median_pose: returns the median of the generated samples

        Args:
            generated_xs (List[torch.Tensor]): list of generated samples
            input_sequence (torch.Tensor): ground truth sequence
            aggr_strategy (str): aggregation strategy

        Raises:
            ValueError: if the aggregation strategy is not valid

        Returns:
            Tuple[torch.Tensor]: selected sample and its reconstruction error
        """

        aggr_strategy = self.aggregation_strategy if aggr_strategy is None else aggr_strategy 
        if aggr_strategy == 'random':
            return generated_xs[np.random.randint(len(generated_xs))]
        
        B, repr_shape = input_sequence.shape[0], input_sequence.shape[1:]
        compute_loss = lambda x: torch.mean(self.loss_fn(x, input_sequence).reshape(-1, prod(repr_shape)), dim=-1)
        losses = [compute_loss(x) for x in generated_xs]

        if aggr_strategy == 'all':
            dims_idxs = list(range(2, len(repr_shape)+2))
            dims_idxs = [1,0] + dims_idxs
            selected_x = torch.stack(generated_xs).permute(*dims_idxs)
            loss_of_selected_x = torch.stack(losses).permute(1,0)
        elif aggr_strategy == 'mean':
            selected_x = None
            loss_of_selected_x = torch.mean(torch.stack(losses), dim=0)
        elif aggr_strategy == 'mean_pose':
            selected_x = torch.mean(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'median':
            loss_of_selected_x, _ = torch.median(torch.stack(losses), dim=0)
            selected_x = None
        elif aggr_strategy == 'median_pose':
            selected_x, _ = torch.median(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'best' or aggr_strategy == 'worst':
            strategy = (lambda x,y: x < y) if aggr_strategy == 'best' else (lambda x,y: x > y)
            loss_of_selected_x = torch.full((B,), fill_value=(1e10 if aggr_strategy == 'best' else -1.), device=self.device)
            selected_x = torch.zeros((B, *repr_shape)).to(self.device)

            for i in range(len(generated_xs)):
                mask = strategy(losses[i], loss_of_selected_x)
                loss_of_selected_x[mask] = losses[i][mask]
                selected_x[mask] = generated_xs[i][mask]
        elif 'quantile' in aggr_strategy:
            q = float(aggr_strategy.split(':')[-1])
            loss_of_selected_x = torch.quantile(torch.stack(losses), q, dim=0)
            selected_x = None
        else:
            raise ValueError(f'Unknown aggregation strategy {aggr_strategy}')
        
        return selected_x, loss_of_selected_x
    

    def _infer_number_of_joint(self, args:argparse.Namespace) -> int:
        """
        Infer the number of joints based on the dataset parameters.

        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model

        Returns:
            int: number of joints
        """
        
        if args.headless:
            joints_to_consider = 14
        elif args.kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17
        return joints_to_consider
    
    
    def _load_tensors(self, split_name:str, aggr_strategy:str, n_gen:int) -> Dict[str, torch.Tensor]:
        """
        Loads the tensors from the experiment directory.

        Args:
            split_name (str): name of the split (train, val, test)
            aggr_strategy (str): aggregation strategy
            n_gen (int): number of generated samples

        Returns:
            Dict[str, torch.Tensor]: dictionary containing the tensors. The keys are inferred from the file names.
        """
        
        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        tensor_files = os.listdir(path)
        tensors = {}
        for t_file in tensor_files:
            t_name = t_file.split('.')[0]
            tensors[t_name] = torch.load(os.path.join(path, t_file))
        return tensors
    
    
    def _pack_out_data(self, selected_x:torch.Tensor, loss_of_selected_x:torch.Tensor, additional_out:List[torch.Tensor], return_:str) -> List[torch.Tensor]:
        """
        Packs the output data according to the return_ argument.

        Args:
            selected_x (torch.Tensor): generated samples selected among the others according to the aggregation strategy
            loss_of_selected_x (torch.Tensor): loss of the selected samples
            additional_out (List[torch.Tensor]): additional output data (ground truth, meta-data, etc.)
            return_ (str): return strategy. Can be 'pose', 'loss', 'all'

        Raises:
            ValueError: if return_ is None and self.model_return_value is None

        Returns:
            List[torch.Tensor]: output data
        """
        
        if return_ is None:
            if self.model_return_value is None:
                raise ValueError('Either return_ or self.model_return_value must be set')
            else:
                return_ = self.model_return_value

        if return_ == 'poses':
            out = [selected_x]
        elif return_ == 'loss':
            out = [loss_of_selected_x]
        elif return_ == 'all':
            out = [loss_of_selected_x, selected_x]
            
        return out + additional_out


    def _save_tensors(self, tensors:Dict[str, torch.Tensor], split_name:str, aggr_strategy:str, n_gen:int) -> None:
        """
        Saves the tensors in the experiment directory.

        Args:
            tensors (Dict[str, torch.Tensor]): tensors to save
            split_name (str): name of the split (val, test)
            aggr_strategy (str): aggregation strategy
            n_gen (int): number of generated samples
        """
        
        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        for t_name, tensor in tensors.items():
            torch.save(tensor, os.path.join(path, t_name+'.pt'))
    

    def _set_diffusion_variables(self) -> None:
        """
        Sets the diffusion variables.
        """
        
        self.noise_scheduler = Diffusion(noise_steps=self.noise_steps, n_joints=self.n_joints,
                                         device=self.device, time=self.n_frames)
        self._beta_ = self.noise_scheduler.schedule_noise()
        self._alpha_ = (1. - self._beta_)
        self._alpha_hat_ = torch.cumprod(self._alpha_, dim=0)

    def _unpack_data(self, x:torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Unpacks the data.

        Args:
            x (torch.Tensor): list containing the input data, the transformation index, the metadata and the actual frames.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: input data, list containing the transformation index, the metadata and the actual frames.
        """
        tensor_data = x[0].to(self.device)
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        meta_out = [transformation_idx, metadata, actual_frames]
        return tensor_data, meta_out


    @property
    def _beta(self) -> torch.Tensor:
        return self._beta_.to(self.device)
    
    
    @property
    def _alpha(self) -> torch.Tensor:
        return self._alpha_.to(self.device)
    
    
    @property
    def _alpha_hat(self) -> torch.Tensor:
        return self._alpha_hat_.to(self.device)
