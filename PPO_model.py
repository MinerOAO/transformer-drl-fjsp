import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformer.transformer import EmbeddingNetwork, OMPairAttention, SimpleMLP
# from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []
        
        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []
        
        
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        
        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]
        

class TransformerScheduler(nn.Module):
    def __init__(self, model_paras):
        super(TransformerScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.embedding = EmbeddingNetwork(self.in_size_ma, self.in_size_ope, self.out_size_ma, self.out_size_ope)

        self.attention = OMPairAttention(self.actor_dim)
        actual_dim = 3 * self.actor_dim
        self.actor = SimpleMLP(2 * actual_dim, self.action_dim, 2 * 2 * actual_dim)
        self.critic = SimpleMLP(actual_dim, 1, 2 * actual_dim)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return (raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5), proc_time_norm

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes

        # Same process in evaluate(), don't forget to change
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]

        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]

        # ope_ma_adj = state.ope_ma_adj_batch[batch_idxes]
        # ope_pre_adj = state.ope_pre_adj_batch[batch_idxes]
        # ope_sub_adj = state.ope_sub_adj_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        norm_opes, norm_mas, norm_proc = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)

        # experiment - less features
        # indices = torch.tensor([0, 1, 2, 3, 4]) # 01234 0134
        # norm_opes = torch.index_select(norm_opes, dim=-1, index=indices)
        
        #----------------------------Important--------------------------------------------------------------------
        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        # Actually here are jobs
        # end_ope_biases_batch records indices of end operation of each job
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        # operation indices of each job [20, 10, 1] -> [20, 10, 8]
        jobs_gather = ope_step_batch[..., :, None][batch_idxes]
        # Input of actor MLP
        h_jobs_padding, h_actions = self.embedding(jobs_gather, norm_opes, norm_mas, norm_proc)

        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        # masked_actions = h_actions.clone()
        # masked_actions[~eligible] = 0

        mask = eligible.transpose(1, 2)
        # [batch_size, machine_num, job_num, feature_size]
        h_actions, graph = self.attention(h_actions)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions)
        scores[~mask] = float('-inf')
        scores = scores.flatten(1)
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        return action_probs, ope_step_batch, None

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, flag_sample=False):
        # batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        h_jobs_padding, h_actions = self.embedding(jobs_gather, raw_opes, raw_mas, proc_time)

        # [batch_size, machine_num, job_num, feature_size]
        h_actions, graph = self.attention(h_actions)

        # Stacking and pooling
        # h_mas_pooled = h_mas.mean(dim=-2)
        # h_opes_pooled = h_opes.mean(dim=-2)

        # h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        scores = self.actor(h_actions)
        mask = eligible.transpose(1, 2)
        scores[~mask] = float('-inf')
        scores = scores.flatten(1)
        action_probs = F.softmax(scores, dim=1)
        
        state_values = self.critic(graph)

        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.max_grad_norm = train_paras["max_grad_norm"]
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = TransformerScheduler(model_paras).to(self.device)
        for param in self.policy.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0,1).flatten(0,1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0,1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        # 解决batch_size个FJSSP实例后，根据每个实例每个状态的价值，计算折扣累积奖励
        for i in range(self.num_envs): # batch_size 20
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                # 使用每个实例每个状态的价值，计算折扣累积奖励. 由于gamma=1.0, No reward discount on time
                discounted_reward = reward + (self.gamma * discounted_reward) # Action-value Func = reward(t) + discount * V(t+1), so V(t+1) similar to r(t+1) + discount * V(t+2)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx])
                # 计算优势函数（优势 = 实际奖励 - 价值估计函数）
                advantages = rewards_envs[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach() # adv = Action-value - State-value, state-value by critic network
                # PPO Loss 1 (Policy loss): Clipped objective function for policy network(actor)
                # 计算策略比率(ratios)，即新策略概率与旧策略概率的比值
                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                # 计算未裁剪的目标(surr1)
                surr1 = ratios * advantages
                # 计算裁剪后的目标(surr2)，将比率限制在[1-ε, 1+ε]范围内
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # PPO Loss 2 (Value loss): Mseloss for state estimation network(critic)
                # PPO Loss 3 (Entropy term): Entropy for exploration
                # Maximize policy loss
                # Minimize value loss
                # Maximize entropy to improve exploration ability
                loss = - self.A_coeff * torch.min(surr1, surr2)\
                       + self.vf_coeff * self.MseLoss(state_values, rewards_envs[i*minibatch_size:(i+1)*minibatch_size])\
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # self.scheduler.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
