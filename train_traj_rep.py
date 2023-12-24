import os
import json
import time
from collections import defaultdict
from typing import Tuple, Union

import gym
import gym3
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributed.rpc import RRef
from gym_minigrid.wrappers import ImgObsWrapper

from minirl.algos.ppo.agent import PPOWorker
from minirl.algos.ppo.policy import PPODiscretePolicy
from minirl.buffer import Buffer
from minirl.envs.gym3_wrapper import ObsTransposeWrapper
from minirl.utils import explained_variance

import logger
import minigrid_env
from network import StateEmbeddingNet, ForwardDynamicNet, InverseDynamicNet, ForwardDynamicUncertaintyNet, \
    SR_rep, VQEncoder, VQDecoder, BackDynamicNet, DetectorHead
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from network import FiLM, ResidualBlock
from wrapper import ModifiedEpisodeStatsWrapper

from vq_embedding import VQEmbedding
from vectorquantizerEMA import VectorQuantizerEMA
from codebook import EuclideanCodebook

def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = ImgObsWrapper(env)
    return env


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = ModifiedEpisodeStatsWrapper(env)
    return env


class ICMPPODiscretePolicy(PPODiscretePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = 128

        self.state_emb_net = StateEmbeddingNet(
            input_shape=kwargs["extractor_kwargs"]["input_shape"], embedding_size=self.embedding_size
        )
        
        self.bisim = kwargs["bisim"]
        self.lap = kwargs["lap"]
        self.lap2 = kwargs["lap2"]
        self.contrastive_batch_wise = kwargs["contrastive_batch_wise"]
        self.bisim_delta = kwargs["bisim_delta"]
        self.uncert = kwargs["uncert"]
        self.permute = kwargs["permute"]
        self.uniform = kwargs["uniform"]
        self.sr = kwargs["sr"]
        self.hinge = kwargs["hinge"]
        self.vq_vae = kwargs["vq_vae"]
        self.backward = kwargs["backward"]
        self.film = kwargs["film"]
        self.critical = kwargs["critical"]

        self.sparse_rew = kwargs["sparse_rew"]
        self.icm_rew = kwargs["icm_rew"]
        self.pred_rew = kwargs["pred_rew"]

        self.mec = kwargs["mec"]
        self.lra_p = kwargs["lra_p"]
        self.contrast_future = kwargs["contrast_future"]

        self.bt = kwargs["bt"]

        if self.bisim:
            self.coef_bisim = kwargs["coef_bisim"]
        if self.lap:
            self.coef_lap = kwargs["coef_lap"]
        if self.lap2:
            self.coef_lap2 = kwargs["coef_lap2"]
        if self.bt:
            self.bt = kwargs["coef_bt"]
        if self.contrastive_batch_wise:
            self.coef_contrastive = kwargs["coef_contrastive"]
        if self.bisim_delta:
            self.coef_bisim_delta = kwargs["coef_bisim_delta"]
            self.lat_sim = kwargs["lat_sim"]


        if self.uncert:
            self.forward_uncert_pred_net = ForwardDynamicUncertaintyNet(num_actions=kwargs["n_actions"])
            self.uncertainty_budget = 0.05
        else:
            self.forward_pred_net = ForwardDynamicNet(num_actions=kwargs["n_actions"])
            if self.backward:
                self.coef_back = kwargs["coef_back"]
                self.backward_pred_net = BackDynamicNet(num_actions=kwargs["n_actions"], embedding_dim = self.embedding_size)

        if self.hinge:
            self.coef_hinge = kwargs["coef_hinge"]

        self.tran_sim = kwargs["tran_sim"]

        self.inverse_pred_net = InverseDynamicNet(num_actions=kwargs["n_actions"], embedding_size=self.embedding_size)

        if self.film:
            self.coef_traj = kwargs["coef_traj"]
            self.FiLM_net = FiLM(num_actions=kwargs["n_actions"])
            self.traj_rep_type = 'score' # choices: [concat, mean, score]
            self.coef_traj = kwargs["coef_traj"]
            # self.embedding_size * self.n_steps (timesteps)
            self.traj_linear = nn.Linear(self.embedding_size * 128, 128)
            # similar to attention
            self.traj_linear_to1 = nn.Linear(self.embedding_size, 1)

        if self.critical:
            # critical state detector
            self.detector_head = DetectorHead()
            self.trajectory_encoder = TransformerEncoder(TransformerEncoderLayer(self.embedding_size, 8), num_layers=2) 

            self.return_predictor = nn.Linear(self.embedding_size, 1)
            self.coef_critical = kwargs["coef_critical"]

        if self.sparse_rew:
            self.coef_sparse_rew = kwargs["coef_sparse_rew"]
        if self.icm_rew:
            self.coef_icm_rew = kwargs["coef_icm_rew"]
        if self.pred_rew:
            self.coef_pred_rew = kwargs["coef_pred_rew"]

        if self.mec:
            self.coef_mec = kwargs["coef_mec"]
            self.order = 4

        if self.lra_p:
            self.coef_lra_p = kwargs["coef_lra_p"]
        
        if self.contrast_future:
            self.coef_cont_future = kwargs["coef_cont_future"]


    def mahalanobis_dist(self, tensor1, tensor2, epsilon=1e-8):
        covariance_matrix = th.cov(tensor1.T)
        if th.allclose(covariance_matrix, th.zeros_like(covariance_matrix)):
            covariance_matrix += epsilon * th.eye(covariance_matrix.size(0))
        else:
            covariance_matrix += epsilon * th.diag(th.diagonal(covariance_matrix))
        inv_covariance = th.inverse(covariance_matrix)
        diff = tensor1 - tensor2
        dist = th.sqrt(th.einsum('ij,jk,ik->i', diff, inv_covariance, diff.T))
        return dist

    def loss(self, *args, next_obs, **kwargs):
        pg_loss, vf_loss, entropy, extra_out = super().loss(*args, **kwargs)
        # mini_batchsize 8, timestep 128 -> 2048
        obs = th.as_tensor(kwargs["obs"]).to(self.device).float() # (1, 2048, 3, 7, 7)
        next_obs = th.as_tensor(next_obs).to(self.device).float()
        firsts = th.as_tensor(kwargs["firsts"]).to(self.device)
        next_firsts = th.as_tensor(kwargs["next_firsts"]).to(self.device)
        actions = th.as_tensor(kwargs["actions"]).to(self.device) # size([1, 2048])
        # (1) use sparse reward
        sparse_r = th.as_tensor(kwargs["rewards"]).to(self.device).float() # size([1, 2048]
        # (2) √ use intrinsic reward of ICM
        icm_r = th.as_tensor(kwargs["icm_rew"]).to(self.device).float() # size([1, 2048]

        smoothl1 = th.nn.SmoothL1Loss(reduction="none")
        mse = th.nn.MSELoss(reduction="none")  
        cosine = th.nn.CosineSimilarity(dim=-1)
        softmax = nn.Softmax(dim=0)

        future_obs = th.as_tensor(kwargs["future_obs"]).to(self.device).float() 
        future_first = th.as_tensor(kwargs["future_first"]).to(self.device)

        # 1.1 extract tuple-level representation, (s,a,s')
        state_emb, _ = self.state_emb_net.extract_features(obs, firsts) # size([1, 2048, 128])
        next_state_emb, _ = self.state_emb_net.extract_features(next_obs, next_firsts)
        pred_actions = self.inverse_pred_net(state_emb, next_state_emb) # size([1, 2048, 7])
  
        if self.uncert:
            pred_next_state_emb, pred_next_state_std = self.forward_uncert_pred_net(state_emb, actions)
            # mse_loss = F.mse_loss(pred_next_state_emb, next_state_emb, reduction="none")
            mse_loss = th.norm(pred_next_state_emb - next_state_emb, dim=-1, p=2).unsqueeze(-1)  # size(1, 2048)
            mse_loss = mse_loss.repeat(1, 1, pred_next_state_emb.size(-1))
            forward_loss = th.mean(th.exp(-pred_next_state_std) * mse_loss + pred_next_state_std * self.uncertainty_budget)
        else:
            # only forward
            pred_next_state_emb = self.forward_pred_net(state_emb, actions)
            forward_loss = th.norm(pred_next_state_emb - next_state_emb, dim=2, p=2).mean()

        if self.backward:
            pred_next_state_emb = self.backward_pred_net(next_state_emb, actions) # size([1, 2048, 128])
            backward_loss = th.norm(pred_next_state_emb - state_emb, dim=2, p=2).mean()
            forward_loss += backward_loss * self.coef_back  # 1.21
        
        if self.critical:
            # sparse reward
            traj_rew = th.as_tensor(kwargs["traj_rewards"]).to(self.device).float() # size([128, 16])
            # icm reward
            icm_traj_rew = th.as_tensor(kwargs["icm_traj_rew"]).to(self.device).float() # size([128, 16])

            traj_obs = th.as_tensor(kwargs["traj_obs"]).to(self.device).float() # size([128, 16, 3, 7, 7])
            traj_actions = th.as_tensor(kwargs["traj_actions"]).to(self.device) # size([128, 16])
            traj_firsts = th.as_tensor(kwargs["traj_firsts"]).to(self.device)
            traj_obs_emb, _ = self.state_emb_net.extract_features(traj_obs, traj_firsts) # size([128, 16, 128]) T, B, dim

            # Critial state detector
            x_out, mask, _ = self.detector_head(traj_obs_emb, lengths=traj_obs_emb.size(0)) # size([128, 16, 128]), size([128, 16, 1])
            l1_norm = th.mean(mask)
            mask = mask.squeeze(-1)
            loss_l1 = th.linalg.norm(mask, ord=1) * 5e-3

            t, bs = mask.size()

            mask = mask.view(t, bs, 1, 1, 1)
            masked_traj = traj_obs * mask
            masked_traj_obs_emb, _ = self.state_emb_net.extract_features(masked_traj, traj_firsts) # size([128, 16, 128]) T, B, dim

            # Take representation as the embeddings of last timestep
            x_hn = self.trajectory_encoder(masked_traj_obs_emb)[-1,:,:]  # torch.Size([16, 128])

            predicted_ret = self.return_predictor(x_hn).squeeze(-1) # [16]
            loss_classify = th.norm(predicted_ret - traj_rew.sum(0), dim=-1, p=2).mean() * 0.1

            reverse_mask = 1 - mask

            reverse_masked_traj = traj_obs * reverse_mask
            reverse_traj_obs_emb, _ = self.state_emb_net.extract_features(reverse_masked_traj, traj_firsts)

            reverse_traj_rep = self.trajectory_encoder(reverse_traj_obs_emb)[-1,:,:] 
            reverse_predicted_ret = self.return_predictor(reverse_traj_rep).squeeze(-1) # size([128, 16, 1]) -> size([128, 16])
            loss_classify_reverse = th.norm(reverse_predicted_ret - th.zeros_like(reverse_predicted_ret), dim=-1, p=2).mean() * 5e-5

            # orthogonal loss
            mask = mask.view(t, bs)
            ort = th.abs(mask.mm(mask.t()))
            ort = (th.ones_like(ort) - th.eye(ort.size(0)).to(self.device)) * ort
            ort_loss = th.mean(th.triu(ort, diagonal=1))
            ort_loss = ort_loss * 5e-4

            forward_loss += loss_classify + loss_classify_reverse # + ort_loss


        # 1.2 extract trajectory-level representation u = (FiLM(phi(s1),a1), ..., FiLM(phi(sT),aT))     
        if self.film:
            # sparse reward
            traj_rew = th.as_tensor(kwargs["traj_rewards"]).to(self.device).float() # size([128, 16])
            # icm reward
            icm_traj_rew = th.as_tensor(kwargs["icm_traj_rew"]).to(self.device).float() # size([128, 16])

            traj_perm = th.randperm(x_hn.size(0))
            x_perm = x_hn[traj_perm, :] # size([16, 128])
            # (1) √ l2 norm
            traj_delta = th.linalg.norm(x_hn-x_perm)
            # (2) mse distance
            # traj_delta = mse(x_hn, x_perm).mul(-1).exp() # size([16, 128])
            # (3) cosine distance
            # traj_delta = cosine(u, u_perm) # size([16])
            # traj_delta = traj_delta.mean(-1).unsqueeze(0) # size([1, 16])

            # 1.2.2
            # (1) √ compute transition distribution similarity for predicted next_state_emb based on forward model
            # TODO:
            # traj_next_obs_emb = self.forward_pred_net(traj_obs_emb, traj_actions) # [128, 16, 128]?
            traj_next_obs_emb = self.forward_pred_net(masked_traj_obs_emb, traj_actions) # [128, 16, 128]?

            # a. trajectory transformer
            next_x_hn = self.trajectory_encoder(traj_next_obs_emb)[-1,:,:]

            # b. bi-lstm
            # _, _, next_x_hn = self.detector_head(traj_next_obs_emb, lengths=traj_next_obs_emb.size(0)) # size([128, 16, 128]), size([128, 16, 1])
            # next_x_hn = next_x_hn.view(next_x_hn.size(1), -1) # size(16, 128)

            next_x_perm = next_x_hn[traj_perm, :]

            # (1) √ l2 norm
            next_state_delta = th.linalg.norm(next_x_hn-next_x_perm)
            # (2) mse distance
            # next_state_delta = mse(next_x_hn, next_x_perm).mul(-1).exp()
            # (3) cosine distance
            # next_state_delta = cosine(s, s_perm)
            # next_state_delta = next_state_delta.mean(-1).unsqueeze(0) # size([1, 16, 128]) or [1, 16]

            if self.sparse_rew:
                traj_rew = traj_rew.T # [128, 16] -> [16, 128]
                sparse_r_delta = th.abs(traj_rew - traj_rew[traj_perm,:]).sum() # [16, 128]
                # (1) √ mse distance
                next_state_delta = next_state_delta + sparse_r_delta * self.coef_sparse_rew # 10
                # (2) cosine distance
                # next_state_delta = next_state_delta + sparse_r_delta.mean(-1) * self.coef_sparse_rew
            elif self.icm_rew:
                icm_traj_rew = icm_traj_rew.T # [128, 16] -> [16, 128]
                icm_r_delta = th.abs(icm_traj_rew - icm_traj_rew[traj_perm,:]).sum()
                # (1) √ mse distance
                next_state_delta = next_state_delta + icm_r_delta * self.coef_icm_rew
                # (2) cosine distance
                # next_state_delta = next_state_delta + icm_r_delta.mean(-1) * self.coef_icm_rew
            elif self.pred_rew:
                # predicted_ret = predicted_ret.T
                # pred_r_delta = th.abs(predicted_ret - predicted_ret[traj_perm,:])
                pred_r_delta = th.abs(predicted_ret - predicted_ret[traj_perm]).sum()

                next_state_delta = next_state_delta + pred_r_delta * self.coef_pred_rew

            else:
                next_state_delta = next_state_delta
            # (1) l2 norm distance, 2 scalar values, coef traj 1
            forward_loss += F.mse_loss(traj_delta, next_state_delta) * self.coef_traj
            # (2) √ mse distance, coef traj 10
            # forward_loss += th.norm(traj_delta - next_delta, dim=-1, p=2).mean() * self.coef_traj
            # (3) cosine distance
            # forward_loss += th.norm(traj_delta - next_delta, dim=-1, p=2) * self.coef_traj

        inverse_loss = F.nll_loss(
            F.log_softmax(th.flatten(pred_actions, 0, 1), dim=-1),
            target=th.flatten(actions, 0, 1),
        )
        

        # 2.1. compute transition distribution similarity
        if self.tran_sim == 'smooth_l1':
            transition_dist = smoothl1(pred_next_state_emb, pred_next_state_emb[:, perm, :])
        elif self.tran_sim == 'mse':
            transition_dist = mse(pred_next_state_emb, pred_next_state_emb[:, perm, :]).mean(-1).mul(-1).exp()
        elif self.tran_sim == 'cosine':
            transition_dist = cosine(pred_next_state_emb, pred_next_state_emb[:, perm, :]) # size(1, 2048)

        elif self.tran_sim == 'mahalanobis':
            transition_dist = self.mahalanobis_dist(pred_next_state_emb[0], pred_next_state_emb[0, perm, :]).unsqueeze(0)
        # if self.vq_vae:
        #     transition_dist += cosine(selected_codes, selected_codes[:, perm, :])

        if self.hinge:
            #  E_(s_t, s_k)max(0,epsilon-||f(s_t)-f(s_k)||^2
            epsilon = 0.1
            distance = th.norm(x_hn - x_perm, p=2, dim=1) ** 2
            # Apply the hinge loss function
            hinge_loss = F.relu(epsilon - distance)            
            hinge_loss = th.mean(hinge_loss)
            forward_loss += hinge_loss * self.coef_hinge  # 0.0029

        if self.lra_p:
            # version1:
            # p = th.einsum("sd, td -> st", next_state_emb[0], pred_next_state_emb[0])
            # i = th.eye(*p.size(), device=self.device)
            # off_diag = ~i.bool()
            # batch_wise_loss = p[off_diag].pow(2).mean() - 2 * p.diag().mean() * 0.1

            # version2:
            p = th.einsum("sd, td -> st", next_state_emb, state_emb_perm)
            i = th.eye(*p.size(), device=self.device)
            off_diag = ~i.bool()
            batch_wise_loss = p[off_diag].pow(2).mean() - 2 * p.diag().mean() * 0.1      
                   
            forward_loss += batch_wise_loss * self.coef_lra_p

        if self.lap:
            # orthonormality loss
            cov = th.matmul(x_hn, x_hn.T) # size(bs, bs) batch-wise
            I = th.eye(*cov.size()).to(self.device)
            off_diag = ~I.bool()
            orth_loss_diag = - 2 * cov.diag().mean()
            orth_loss_off_diag = cov[off_diag].pow(2).mean()
            # orth_loss = orth_loss_diag * 0.01 + orth_loss_off_diag
            orth_loss = orth_loss_diag * 0.1 + orth_loss_off_diag
            # orth_loss = orth_loss_diag + orth_loss_off_diag
            
            forward_loss += orth_loss * self.coef_lap  # 0.2295
        
        if self.bt:
            cov = x_hn.T @ x_hn
            on_diag = th.diagonal(cov).add_(-1).pow_(2).mean()
            I = th.eye(*cov.size()).to(self.device)
            off = ~I.bool()
            off_diag = cov[off].pow_(2).mean() 
            bt_loss = on_diag + off_diag
            forward_loss += bt_loss * self.coef_bt

        # svd_values = th.linalg.svd(state_emb[0])[1]

        if self.lap2:
            # ||phi(s_t) - phi(s_t+1)||^2 + orthonormality loss
            # (1) TODO: coefficient
            forward_loss += (x_hn - next_x_hn).pow(2).mean()
            # (2) orthonormality loss
            cov = th.matmul(x_hn, x_hn.T) # size(bs, bs) batch-wise
            I = th.eye(*cov.size()).to(self.device)
            off_diag = ~I.bool()
            orth_loss_diag = - 2 * cov.diag().mean()
            orth_loss_off_diag = cov[off_diag].pow(2).mean()
            # orth_loss = orth_loss_diag * 0.01 + orth_loss_off_diag
            orth_loss = orth_loss_diag * 0.1 + orth_loss_off_diag
            forward_loss += orth_loss * self.coef_lap2  
    

        if self.mec:
            p = x_hn
            z = x_hn

            c= p @ z.T  # size(bs, bs)
            c = c/100
            power_matrix = c
            sum_matrix = th.zeros_like(power_matrix)

            for k in range(1, self.order+1):
                if k > 1:
                    power_matrix = th.matmul(power_matrix, c)
                if (k + 1) % 2 == 0:
                    sum_matrix += power_matrix / k
                else: 
                    sum_matrix -= power_matrix / k

            trace = th.trace(sum_matrix)/100
            forward_loss += -1 * trace * self.coef_mec

            
        if self.contrastive_batch_wise: # batch_wise
            p = th.einsum("sd, td -> st", x_hn, next_x_hn)
            i = th.eye(*p.size(), device=self.device)
            off_diag = ~i.bool()
            batch_wise_loss = p[off_diag].pow(2).mean() - 2 * p.diag().mean() * 0.1
            forward_loss += batch_wise_loss * self.coef_contrastive # 0.0892

        if self.contrast_future:
            # contrastive learning for future_obs as positive samples
            future_emb, _ = self.state_emb_net.extract_features(future_obs, future_first)    
            # 1. paper way 
            # logits = th.einsum("sd, td -> st", state_emb[0], future_emb[0]) # batch * batch
            # I = th.eye(*logits.size()).to(self.device)
            # off_diag = ~I.bool()
            # logits_off_diag = logits[off_diag].reshape(logits.size(0), logits.size(0)-1)
            # loss = -logits.diag() + th.logsumexp(logits_off_diag, dim=-1) * 0.1
            # forward_loss += loss.mean() * self.coef_cont_future

            # 2. traditional way, as in SimCLR
            state_emb = F.normalize(state_emb, 0)
            future_emb = F.normalize(future_emb, 0)
            p = th.einsum("sd, td -> st", state_emb, future_emb)
            i = th.eye(*p.size(), device=self.device)
            off_diag = ~i.bool()
            future_loss = p[off_diag].pow(2).mean() - 2 * p.diag().mean() * 0.1
            forward_loss += future_loss * self.coef_cont_future 

        return pg_loss, vf_loss, entropy, forward_loss, inverse_loss, extra_out
        


def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train(config):
    # Setup logger
    env_name = config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["id"]
    task_name = "-".join(env_name.split("-")[1:-1])
    run_dir = os.path.join(
        config["run_cfg"]["log_dir"],
        task_name,
        f"run_{config['run_cfg']['run_id']}",
    )
    logger.configure(dir=run_dir, format_strs=["stdout", "csv"]) #, "wandb"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)
    # Modify worker to add ICM intrinsic rewards
    class ICMPPOWorker(PPOWorker):
        def __init__(self, *args, forward_loss_coef, inverse_loss_coef, **kwargs):
            super().__init__(*args, **kwargs)
            self.ep_cnts = [dict() for _ in range(self.env.num)]
            self.forward_loss_coef = forward_loss_coef
            self.inverse_loss_coef = inverse_loss_coef
            self.uncert = kwargs['policy_kwargs']['uncert']
            self.permute = kwargs['policy_kwargs']['permute']
            self.vq_vae = kwargs['policy_kwargs']['vq_vae']
            self.coef_vq = kwargs['policy_kwargs']['coef_vq']
            self.film = kwargs['policy_kwargs']['film']
            self.sparse_rew = kwargs['policy_kwargs']['sparse_rew']
            self.icm_rew = kwargs['policy_kwargs']['icm_rew']
            self.pred_rew = kwargs['policy_kwargs']['pred_rew']
            self.std = kwargs['policy_kwargs']['std']
            self.critical = kwargs['policy_kwargs']['critical']

        def collect_batch(self) -> Tuple[dict, np.ndarray, np.ndarray]:
            """
            Additionally, collect next obs
            """
            # Update episodic unique states set before collecting experience
            reward, obs, first = self.env.observe()
            state_keys = [tuple(x) for x in obs.reshape(obs.shape[0], -1).tolist()]
            for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                if first[env_idx]:
                    ep_cnt.clear()
                ep_cnt[key] = 1 + ep_cnt.get(key, 0)

            # Rollout
            batch = defaultdict(list)
            for _ in range(self.n_steps):
                reward, obs, first = self.env.observe()
                action, value, logpacs = self.policy.step(
                    obs[None, ...], first[None, ...]
                )
                batch["obs"].append(obs)
                batch["first"].append(first)
                batch["action"].append(action.squeeze(0))
                batch["value"].append(value.squeeze(0))
                batch["logpac"].append(logpacs.squeeze(0))
                self.env.act(action.squeeze(0))
                reward, next_obs, next_first = self.env.observe()
                batch["next_first"].append(next_first)
                # Calculate ICM intrinsic reward
                
                with th.no_grad():
                    state_emb, _ = self.policy.state_emb_net.extract_features(
                        th.as_tensor(obs[None, ...]).float().to(self.device),
                        th.as_tensor(first[None, ...]).float().to(self.device),
                    )
                    next_state_emb, _ = self.policy.state_emb_net.extract_features(
                        th.as_tensor(next_obs[None, ...]).float().to(self.device),
                        th.as_tensor(next_first[None, ...]).float().to(self.device),
                    )
                    if self.vq_vae:
                        _, _, pred_next_state_emb, _ = self.policy.compute_vq_loss3(state_emb, next_state_emb, th.as_tensor(action).to(self.device))
                    else:
                        if self.uncert:
                            pred_next_state_emb, pred_next_state_std = self.policy.forward_uncert_pred_net(state_emb, th.as_tensor(action).to(self.device))
                        else:
                            pred_next_state_emb = self.policy.forward_pred_net(
                                state_emb, th.as_tensor(action).to(self.device)
                            )
                    

                # if self.uncert:
                #     # mse = F.mse_loss(pred_next_state_emb, next_state_emb, reduction="none")
                #     mse = th.norm(next_state_emb - pred_next_state_emb, dim=2, p=2).unsqueeze(-1).repeat(1, 1, next_state_emb.size(-1))
                #     icm_rew = mse - th.exp(pred_next_state_std) * 0.001
                #     # icm_rew = th.clamp(icm_rew, min=-1, max=1)
                #     icm_rew = th.mean(icm_rew, dim=-1).cpu().numpy().squeeze(0)
                # else:
                icm_rew = th.norm(next_state_emb - pred_next_state_emb, dim=2, p=2)
                if self.std:
                    # use the std of state_emb or pred_next_state_emb
                    icm_rew += th.std(state_emb[0], -1).unsqueeze(0) * 0.01
                icm_rew = icm_rew.cpu().numpy().squeeze(0)
                # Record episodic visitation count and calculate curiosity
                ep_curiosity = np.zeros(shape=(self.env.num,), dtype=np.float32)
                state_keys = [
                    tuple(x) for x in next_obs.reshape(next_obs.shape[0], -1).tolist()
                ]
                for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                    if next_first[env_idx]:
                        ep_cnt.clear()
                        ep_cnt[key] = 1
                    else:
                        ep_cnt[key] = 1 + ep_cnt.get(key, 0)
                        if config["ep_curiosity"] == "visit":
                            ep_curiosity[env_idx] = ep_cnt[key] == 1
                        elif config["ep_curiosity"] == "count":
                            ep_curiosity[env_idx] = 1 / np.sqrt(ep_cnt[key])
                # Add into batch
                batch["icm_rew"].append(icm_rew)
                batch["ep_curiosity"].append(ep_curiosity)
                batch["reward"].append(reward)
                batch["next_obs"].append(next_obs)

            # collect future obs
            future_idx = th.randint(0, self.n_steps, size=(len(batch["obs"]), self.n_steps))
            # convert batch["obs"] into tensor, then use gather, then convert to numpy array
            obs_tensor = th.as_tensor(batch["obs"])
            future_obs = obs_tensor.gather(1, future_idx.view(obs_tensor.size(0), self.n_steps, 1, 1, 1).expand(-1, -1, *obs_tensor.shape[2:]))
            batch["future_obs"] = np.asarray(future_obs, dtype=obs.dtype)

            future_first_tensor = th.as_tensor(batch["first"])
            future_first = future_first_tensor.gather(1, future_idx) 
            batch["future_first"] = np.asarray(future_first, dtype=np.bool)

            batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
            batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
            batch["next_obs"] = np.asarray(batch["next_obs"], dtype=next_obs.dtype)
            batch["first"] = np.asarray(batch["first"], dtype=np.bool)
            batch["next_first"] = np.asarray(batch["next_first"], dtype=np.bool)
            batch["ep_curiosity"] = np.asarray(batch["ep_curiosity"], dtype=np.float32)
            batch["icm_rew"] = np.asarray(batch["icm_rew"], dtype=np.float32)
            batch["action"] = np.asarray(batch["action"])
            batch["value"] = np.asarray(batch["value"], dtype=np.float32)
            batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)

            
            return batch, next_obs, next_first

        def process_batch(
            self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray
        ):
            """
            Add ICM intrinsic reward
            """
            intrinsic_rewards = batch["icm_rew"]
            if self.permute:
                to_shuffle = intrinsic_rewards[intrinsic_rewards != 0]
                np.random.shuffle(to_shuffle)
                intrinsic_rewards[intrinsic_rewards != 0] = to_shuffle
            if config["ep_curiosity"] in ("visit", "count"):
                intrinsic_rewards *= batch["ep_curiosity"]
            if config["intrinsic_only"]:
                batch["reward"] = intrinsic_rewards * config["intrinsic_reward_coef"]
            else:
                batch["reward"] += intrinsic_rewards * config["intrinsic_reward_coef"]
            super().process_batch(batch=batch, last_obs=last_obs, last_first=last_first)
            return batch

        def learn(self, scheduler_step: int, buffer: Union[Buffer, RRef]):
            # Retrieve data from buffer
            if isinstance(buffer, RRef):
                batch = buffer.rpc_sync().get_all()
            else:
                batch = buffer.get_all()
            # Build a dict to save training statistics
            stats_dict = defaultdict(list)

            # 1. tuple-level shuffle
            # Minibatch training
            B, T = batch["obs"].shape[:2] # 128, 128
            if self.policy.is_recurrent:
                batch_size = B
                indices = np.arange(B)
            else:
                batch_size = B * T
                indices = np.mgrid[0:B, 0:T].reshape(2, batch_size).T
            minibatch_size = batch_size // self.n_minibatches  # 16384 / 8 = 2048
            assert minibatch_size > 1

            # 2. trajectory-level shuffle
            if self.film or self.critical:
                traj_batchsize = B # 128
                traj_indices = np.arange(B)
                traj_minibatchsize = traj_batchsize // self.n_minibatches # 128 / 8 = 16
                assert traj_minibatchsize > 1

            # Get current clip range
            cur_clip_range = self.clip_range.value(step=scheduler_step)
            cur_vf_clip_range = self.vf_clip_range.value(step=scheduler_step)
            # Train for n_epochs
            for _ in range(self.n_epochs): # 4
                np.random.shuffle(indices)
                if self.film or self.critical:
                    np.random.shuffle(traj_indices)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size # 2048
                   
                    if self.policy.is_recurrent:
                        sub_indices = indices[start:end]
                        rnn_states = batch["rnn_states"][sub_indices].swapaxes(0, 1)
                    else:
                        sub_indices = indices[start:end]
                        sub_indices = tuple(sub_indices.T) + (None,)
                        rnn_states = None

                    self.optimizer.zero_grad()

                    if self.film or self.critical:
                        traj_start = start // T
                        traj_end = traj_start + traj_minibatchsize # 16
                        traj_subindices = traj_indices[traj_start: traj_end]
                        (
                            pg_loss,
                            vf_loss,
                            entropy,
                            forward_loss,
                            inverse_loss,
                            extra_out,
                        ) = self.policy.loss(
                            obs=batch["obs"][sub_indices].swapaxes(0, 1),
                            next_obs=batch["next_obs"][sub_indices].swapaxes(0, 1),
                            advs=batch["adv"][sub_indices].swapaxes(0, 1),
                            firsts=batch["first"][sub_indices].swapaxes(0, 1),
                            next_firsts=batch["next_first"][sub_indices].swapaxes(0, 1),
                            actions=batch["action"][sub_indices].swapaxes(0, 1),
                            old_values=batch["value"][sub_indices].swapaxes(0, 1),
                            old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                            rnn_states=rnn_states,
                            clip_range=cur_clip_range,
                            vf_clip_range=cur_vf_clip_range,
                            normalize_adv=self.normalize_adv,
                            traj_obs=batch["obs"][traj_subindices].swapaxes(0, 1),
                            traj_next_obs=batch["next_obs"][traj_subindices].swapaxes(0, 1),
                            traj_actions=batch["action"][traj_subindices].swapaxes(0, 1),
                            traj_firsts=batch["first"][traj_subindices].swapaxes(0, 1),
                            traj_next_firsts=batch["next_first"][traj_subindices].swapaxes(0, 1),
                            rewards=batch["reward"][sub_indices].swapaxes(0, 1),
                            icm_rew=batch["icm_rew"][sub_indices].swapaxes(0, 1),
                            traj_rewards=batch["reward"][traj_subindices].swapaxes(0, 1),
                            icm_traj_rew=batch["icm_rew"][traj_subindices].swapaxes(0, 1),
                            future_obs=batch["future_obs"][sub_indices].swapaxes(0, 1),
                            future_first=batch["future_first"][sub_indices].swapaxes(0, 1),
                        )
                    else:
                        (
                            pg_loss,
                            vf_loss,
                            entropy,
                            forward_loss,
                            inverse_loss,
                            extra_out,
                        ) = self.policy.loss(
                            obs=batch["obs"][sub_indices].swapaxes(0, 1),
                            next_obs=batch["next_obs"][sub_indices].swapaxes(0, 1),
                            advs=batch["adv"][sub_indices].swapaxes(0, 1),
                            firsts=batch["first"][sub_indices].swapaxes(0, 1),
                            next_firsts=batch["next_first"][sub_indices].swapaxes(0, 1),
                            actions=batch["action"][sub_indices].swapaxes(0, 1),
                            old_values=batch["value"][sub_indices].swapaxes(0, 1),
                            old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                            rnn_states=rnn_states,
                            clip_range=cur_clip_range,
                            vf_clip_range=cur_vf_clip_range,
                            normalize_adv=self.normalize_adv,
                            rewards=batch["reward"][sub_indices].swapaxes(0, 1),
                            icm_rew=batch["icm_rew"][sub_indices].swapaxes(0, 1),
                            future_obs=batch["future_obs"][sub_indices].swapaxes(0, 1),
                            future_first=batch["future_first"][sub_indices].swapaxes(0, 1),
                        )
                    total_loss = (
                        pg_loss
                        + self.vf_loss_coef * vf_loss
                        - self.entropy_coef * entropy
                        + self.forward_loss_coef * forward_loss
                        + self.inverse_loss_coef * inverse_loss
                    )
                    total_loss.backward()
                    self.pre_optim_step_hook()
                    self.optimizer.step()
                    # Saving statistics
                    stats_dict["policy_loss"].append(pg_loss.item())
                    stats_dict["value_loss"].append(vf_loss.item())
                    stats_dict["forward_loss"].append(forward_loss.item())
                    stats_dict["inverse_loss"].append(inverse_loss.item())
                    stats_dict["entropy"].append(entropy.item())
                    stats_dict["total_loss"].append(total_loss.item())
                    stats_dict["ep_percentage"].append(batch["ep_curiosity"][sub_indices].sum()/len(batch["ep_curiosity"][sub_indices]))
                    for key in extra_out:
                        stats_dict[key].append(extra_out[key].item())
            # Compute mean
            for key in stats_dict:
                stats_dict[key] = np.mean(stats_dict[key])
            # Compute explained variance
            stats_dict["explained_variance"] = explained_variance(
                y_pred=batch["value"], y_true=batch["value"] + batch["adv"]
            )
            return stats_dict

    # Create worker
    worker = ICMPPOWorker(**config["worker_kwargs"])

    # Create buffer
    buffer_size = worker.env.num * worker.n_steps
    buffer = Buffer(max_size=buffer_size, sequence_length=worker.n_steps)

    # Training
    n_iters = int(config["run_cfg"]["n_timesteps"] / worker.env.num / worker.n_steps)
    for i in range(n_iters):
        t_start = time.perf_counter()
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        # Logging
        ret = worker.env.callmethod("get_ep_stat_mean", "r")
        finish = worker.env.callmethod("get_ep_stat_mean", "finish")
        logger.logkv("time", time.perf_counter() - t_start)
        logger.logkv("iter", i + 1)
        logger.logkv("return", ret)
        logger.logkv("success", finish)
        for key, value in stats_dict.items():
            logger.logkv(key, value)
        logger.dumpkvs()

    # Save model
    th.save(worker.policy.state_dict(), os.path.join(run_dir, "policy.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--n_timesteps", type=int, default=int(4e7))
    parser.add_argument("--env_id", type=str, default="MiniGrid-KeyCorridorS4R3-v0")
    parser.add_argument("--n_envs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4) 
    parser.add_argument("--n_steps", type=float, default=128)
    parser.add_argument("--n_epochs", type=float, default=4)
    parser.add_argument("--n_minibatches", type=int, default=8)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--intrinsic_only", action="store_true")
    parser.add_argument("--intrinsic_reward_coef", type=float, default=1e-3)
    parser.add_argument(
        "--ep_curiosity", type=str, choices=("visit", "count", "none"), default="none"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bisim", action='store_true', default=False)
    parser.add_argument("--coef_bisim", type=float, default=1e2)
    parser.add_argument("--forward_loss_coef", type=float, default=10)
    parser.add_argument("--inverse_loss_coef", type=float, default=0.1)
    parser.add_argument("--dropout_int_rew", type=bool, default=False)

    parser.add_argument("--lap", action='store_true', default=False)
    parser.add_argument("--coef_lap", type=float, default=1e2)
    parser.add_argument("--contrastive_batch_wise", action='store_true', default=False)
    parser.add_argument("--coef_contrastive", type=float, default=1)
    parser.add_argument("--bisim_delta", action='store_true', default=False)
    parser.add_argument("--coef_bisim_delta", type=float, default=1)
    parser.add_argument("--uncert", action='store_true', default=False)
    parser.add_argument("--permute", action='store_true', default=False)
    parser.add_argument("--uniform", action='store_true', default=False)
    parser.add_argument("--coef_uniform", type=float, default=1)
    parser.add_argument("--sr", action='store_true', default=False)
    parser.add_argument("--hinge", action='store_true', default=False)
    parser.add_argument("--coef_hinge", type=float, default=1)
    parser.add_argument("--vq_vae", action='store_true', default=False)
    parser.add_argument("--coef_vq", type=float, default=1)
    
    parser.add_argument("--backward", action='store_true', default=False)
    parser.add_argument("--coef_back", type=float, default=1)

    parser.add_argument("--lat_sim", type=str, choices=("smoothl1", "mse", "cosine", "mse_delta"), default="none")
    parser.add_argument("--tran_sim", type=str, choices=("smoothl1", "mse", "cosine"), default="none")
    parser.add_argument("--sr_sim", type=str, choices=("smoothl1", "mse", "cosine"), default="none")
    
    # sprint3, trajectory-level
    parser.add_argument("--film", action='store_true', default=False)
    parser.add_argument("--coef_traj", type=float, default=1)
    parser.add_argument("--sparse_rew", action='store_true', default=False)
    parser.add_argument("--coef_sparse_rew", type=float, default=1)
    parser.add_argument("--icm_rew", action='store_true', default=False)
    parser.add_argument("--coef_icm_rew", type=float, default=1)
    parser.add_argument("--pred_rew", action='store_true', default=False)
    parser.add_argument("--coef_pred_rew", type=float, default=1)

    parser.add_argument("--lap2", action='store_true', default=False)
    parser.add_argument("--coef_lap2", type=float, default=1)
    # maximum entropy coding
    parser.add_argument("--mec", action='store_true', default=False)
    parser.add_argument("--coef_mec", type=float, default=1)

    parser.add_argument("--std", action='store_true', default=False)

    parser.add_argument("--lra_p", action='store_true', default=False)
    parser.add_argument("--coef_lra_p", type=float, default=1)

    # contrastive_future
    parser.add_argument("--contrast_future", action='store_true', default=False)
    parser.add_argument("--coef_cont_future", type=float, default=1)

    # critical state detector
    parser.add_argument("--critical", action='store_true', default=False)
    parser.add_argument("--coef_critical", type=float, default=1)

    parser.add_argument("--bt", action='store_true', default=False)
    parser.add_argument("--coef_bt", type=float, default=1)

    args = parser.parse_args()
    config = {
        # Run
        "run_cfg": {
            "run_id": args.run_id,
            "log_dir": f"./exps/icm_{args.ep_curiosity}/",
            "n_timesteps": args.n_timesteps,
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_gym3_env,
            "env_kwargs": {
                "env_fn": make_gym_env,
                "num": args.n_envs,
                "env_kwargs": {"id": args.env_id},
                "use_subproc": False,
            },
            "policy_fn": "__main__.ICMPPODiscretePolicy",
            "policy_kwargs": {
                "extractor_fn": "cnn",
                "extractor_kwargs": {
                    "input_shape": (3, 7, 7),
                    "conv_kwargs": (
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                    ),
                    "activation": nn.ELU,
                    "hiddens": (512,),
                    "final_activation": nn.ReLU,
                },
                "n_actions": 7,
                "bisim": args.bisim,
                "coef_bisim": args.coef_bisim,
                "lap": args.lap,
                "coef_lap": args.coef_lap,
                "contrastive_batch_wise": args.contrastive_batch_wise,
                "coef_contrastive": args.coef_contrastive,
                "bisim_delta": args.bisim_delta,
                "coef_bisim_delta": args.coef_bisim_delta,
                "uncert": args.uncert,
                "permute": args.permute,
                "uniform": args.uniform,
                "coef_uniform": args.coef_uniform,
                "sr": args.sr,
                "hinge": args.hinge,
                "coef_hinge": args.coef_hinge,
                "lat_sim": args.lat_sim,
                "tran_sim": args.tran_sim,
                "sr_sim": args.sr_sim,
                "vq_vae": args.vq_vae,
                "coef_vq": args.coef_vq,
                "backward": args.backward,
                "coef_back": args.coef_back,
                "film": args.film,
                "coef_traj": args.coef_traj,
                "sparse_rew": args.sparse_rew,
                "coef_sparse_rew": args.coef_sparse_rew,
                "icm_rew": args.icm_rew,
                "coef_icm_rew": args.coef_icm_rew,
                "lap2": args.lap2,
                "coef_lap2": args.coef_lap2,
                "mec": args.mec,
                "coef_mec": args.coef_mec,
                "std": args.std,
                "lra_p": args.lra_p,
                "coef_lra_p": args.coef_lra_p,
                "contrast_future": args.contrast_future,
                "coef_cont_future": args.coef_cont_future,
                "critical": args.critical,
                "coef_critical": args.coef_critical,
                "pred_rew": args.pred_rew,
                "coef_pred_rew": args.coef_pred_rew,
                "bt": args.bt,
                "coef_bt": args.coef_bt,
            },
            "forward_loss_coef": args.forward_loss_coef,
            "inverse_loss_coef": args.inverse_loss_coef,
            "optimizer_fn": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": args.lr},
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "n_minibatches": args.n_minibatches,
            "discount_gamma": args.discount_gamma,
            "gae_lambda": args.gae_lambda,
            "normalize_adv": True,
            "clip_range": args.clip_range,
            "entropy_coef": args.entropy_coef,
            "device": args.device,
        },
        # ICM
        "ep_curiosity": args.ep_curiosity,
        "intrinsic_only": args.intrinsic_only,
        "intrinsic_reward_coef": args.intrinsic_reward_coef,
        "dropout_int_rew": args.dropout_int_rew,
    }

    train(config)
