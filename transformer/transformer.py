import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, MultiheadAttention
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, bias=False):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size, bias=bias)
        )
    def forward(self, x):
        return self.fc(x)

class GateMechanism(nn.Module):
    def __init__(self, feature_size):
        super(GateMechanism, self).__init__()
        self.linear_w_r = nn.Linear(feature_size, feature_size, bias=False)
        self.linear_u_r = nn.Linear(feature_size, feature_size, bias=False)
        self.linear_w_z = nn.Linear(feature_size, feature_size, bias=False)  ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(feature_size, feature_size, bias=False)
        self.linear_w_g = nn.Linear(feature_size, feature_size, bias=False)
        self.linear_u_g = nn.Linear(feature_size, feature_size, bias=False)

    def forward(self, x, x_attn):

        z = torch.sigmoid(self.linear_w_z(x) + self.linear_u_z(x_attn))  # MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(x) + self.linear_u_r(x_attn))
        x_hat = torch.tanh(self.linear_w_g(x) + self.linear_u_g(r * x_attn))  # Note elementwise multiplication of r and x
        h = (1. - z) * x_attn + z * x_hat
        h = h * torch.sigmoid(x_attn)
        return h

class MachineEmbedd(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(MachineEmbedd, self).__init__()
        self.fc = SimpleMLP(in_size, out_size, 2 * out_size, bias)
    # ope_ma [batch, operaion_num, machine_num]
    def forward(self, mas):
        return self.fc(mas)
    
class OperationEmbed(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(OperationEmbed, self).__init__()
        self.fc = SimpleMLP(in_size, out_size, 2 * out_size, bias)
    def forward(self, ope):
        # incorporate with next operation
        return self.fc(ope)

class EmbeddingNetwork(nn.Module):

    def __init__(self,
                 in_size_ma,
                 in_size_ope,
                 out_size_ma,
                 out_size_ope):
        super(EmbeddingNetwork, self).__init__()
        # Machine node embedding
        self.machines_embedd = MachineEmbedd(in_size_ma, out_size_ma, bias=False)

        # Operation node embedding
        self.ope_embedd = OperationEmbed(in_size_ope, out_size_ope, bias=False)
        # self.sub_ope_embedd = OperationEmbed(in_size_ope, out_size_ope)
    # input:
    # state.ope_ma_adj_batch, state.batch_idxes, features
    # state.ope_pre_adj_batch, state.ope_sub_adj_batch
    # features: ([1, max(operation_num), 6], [1, machine_num, 3], [1, max(ope_num), max(mas_num)])
    def forward(self, jobs_gather, feature_ope, feature_mas, feature_arc):

        # use indices to filter out current operation features of each job
        # Select current ope_sub_adj
        # cur_ope_sub_adj = jobs_gather.expand(-1, -1, ope_sub_adj_batch.size(1))
        # cur_ope_sub_adj = ope_sub_adj_batch.gather(1, cur_ope_sub_adj)

        # # Mask of operations that have sub opes
        # mask = torch.any(cur_ope_sub_adj, dim=-1)
        # # Indices of features of sub opes
        # indices = torch.argmax(cur_ope_sub_adj.float(), dim=-1)
        # # corner case: index 0, argmax 0
        # indices = torch.where(mask, indices, torch.tensor(-1))

        # # Gather sub opes from original features
        # indices = indices.unsqueeze(-1).expand(-1, -1, feature_ope.size(-1))
        # mask = (indices != -1).float()
        # indices = indices.clamp(min=0)
        # feature_sub_ope = feature_ope.gather(1, indices)
        # # Set unrelated features to 0
        # feature_sub_ope = feature_sub_ope * mask

        # 
        ope_gather = jobs_gather.expand(-1, -1, feature_ope.size(-1))
        feature_ope = feature_ope.gather(1, ope_gather)

        arc_gather = jobs_gather.expand(-1, -1, feature_arc.size(-1))
        h_arc = feature_arc.gather(1, arc_gather)

        # [1, machine_num, pre-defined embedding dimension]
        # [1, operation_num, pre-defined embedding dimension]
        h_mas = self.machines_embedd(feature_mas)
        h_opes = self.ope_embedd(feature_ope)
        # h_sub_opes = self.ope_embedd(feature_sub_ope)

        h_arc = h_arc.unsqueeze(-1)
        h_jobs_padding = h_opes.unsqueeze(-2).expand(-1, -1, feature_arc.size(-1), -1)
        # h_sub_opes_padding = h_sub_opes.unsqueeze(-2).expand(-1, -1, features[2].size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand(-1, h_jobs_padding.size(1), -1, -1)

        h_actions = torch.cat((h_jobs_padding, h_arc, h_mas_padding), dim=-1)

        return h_jobs_padding, h_actions

class FeedForward(nn.Module):
    def __init__(self, input, hidden):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, input, bias=False)
        )
        self.norm1 = nn.LayerNorm(input)
        self.norm2 = nn.LayerNorm(input)
    def forward(self, attn, residual):
        x1 = self.norm1(residual + attn)
        ff_output = self.ffn(x1)
        x2 = self.norm2(x1 + ff_output)
        return x2
    
class GLUFeedForward(nn.Module):
    def __init__(self, input, hidden, acti=nn.SiLU):
        super(GLUFeedForward, self).__init__()
        self.linear1 = nn.Linear(input, hidden, bias=False)
        self.glu = acti()
        
        self.linear2 = nn.Linear(input, hidden, bias=False)

        self.linear_out = nn.Linear(hidden, input, bias=False)

        self.norm1 = nn.LayerNorm(input)
        self.norm2 = nn.LayerNorm(input)
        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, attn, residual):
        # First add norm
        #x1 = self.norm1(residual + self.dropout1(attn))

        x1 = self.norm1(residual + attn)

        xw = self.linear1(x1)
        xw = self.glu(xw)

        xv = self.linear2(x1)

        out = self.linear_out(xw * xv)
        # Second add norm
        # x2 = self.norm2(x1 + self.dropout2(out))
        x2 = self.norm2(x1 + out)
        return x2
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  
        self.d_k = d_model // heads  
        self.h = heads 

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)

        # self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model, bias=False) 

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch_size

        # Split
        # [batch_size, seq_length, num_head, feature_dim]
        # [batch_size, num_head, seq_length, feature_dim]
        q = self.q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # [batch_size, num_head, seq_length, feature_dim] * [batch_size, num_head, feature_dim, seq_length]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)

        # if mask is not None:
        #     mask = mask.unsqueeze(1) 
        #     scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数

        # [batch_size, num_head, seq_length, seq_length]
        scores = F.softmax(scores, dim=-1)
        # scores = self.dropout(scores)

        output = torch.matmul(scores, v) 

        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(concat)

class OMPairAttention(nn.Module):

    def __init__(self, actor_feature_size):
        
        super(OMPairAttention, self).__init__()

        self.TransEncoderOpe = TransformerEncoder(actor_feature_size, 5)
        self.TransEncoderMac = TransformerEncoder(actor_feature_size, 5)
        self.TransEncoder = TransformerEncoder(actor_feature_size, 5)

        self.gate1 = GateMechanism(3 * actor_feature_size)

    def forward(self, h_actions):
        mac_flatten_actions = h_actions.reshape(h_actions.size(0) * h_actions.size(1), h_actions.size(2), h_actions.size(-1))
        h_actions = h_actions.transpose(1, 2)
        ope_flatten_actions = h_actions.reshape(h_actions.size(0) * h_actions.size(1), h_actions.size(2), h_actions.size(-1))
        flatten_actions = h_actions.reshape(h_actions.size(0), h_actions.size(1) * h_actions.size(2), h_actions.size(-1))

        # 以[batch_size, ope_num, mas_num, feature_size]为基准?
        mac_attn_actions = self.TransEncoderMac(mac_flatten_actions).view_as(h_actions.transpose(1, 2)).transpose(1, 2)
        ope_attn_actions = self.TransEncoderOpe(ope_flatten_actions).view_as(h_actions)
        attn_actions = self.TransEncoder(flatten_actions).view_as(h_actions)

        result = torch.cat((attn_actions, ope_attn_actions, mac_attn_actions), dim=-1)

        # Pooling and gating
        mean_attn_actions = result.mean(dim=(1, 2))
        max_attn_actions = result.amax(dim=(1, 2))

        graph = self.gate1(max_attn_actions, mean_attn_actions)

        return torch.cat((result, graph.unsqueeze(-2).unsqueeze(-2).expand_as(result)), dim=-1), graph

class TransformerEncoder(nn.Module):

    def __init__(self, actor_feature_size, num_heads):
        
        super(TransformerEncoder, self).__init__()

        self.MHA1 = MultiHeadAttention(actor_feature_size, num_heads)
        self.glu_feed_forward = GLUFeedForward(actor_feature_size, 4 * actor_feature_size)

    def forward(self, flatten_actions):
        attention_1 = self.MHA1(flatten_actions, flatten_actions, flatten_actions)
        glu_result = self.glu_feed_forward(attention_1, flatten_actions)
        return glu_result

    