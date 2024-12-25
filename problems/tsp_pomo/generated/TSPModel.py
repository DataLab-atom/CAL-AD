
import torch
import torch.nn as nn
import torch.nn.functional as F
def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    heu_ij = - log(dis_ij) if j is the topK nearest neighbor of i, else - dis_ij
    """
    distance_matrix[distance_matrix == 0] = 1e5
    K = 100
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)
    heu = -distance_matrix.clone()
    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)
    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])
    return heu

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        distance_matrices = torch.cdist(reset_state.problems, reset_state.problems, p=2)

        self.attention_bias = torch.stack([
                heuristics(distance_matrices[i]) for i in range(distance_matrices.size(0))
            ], dim=0)
        assert not torch.isnan(self.attention_bias).any()
        assert not torch.isinf(self.attention_bias).any()
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            attention_bias_current_node = self.attention_bias[torch.arange(batch_size)[:, None], state.current_node, :]
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, attention_bias_current_node=attention_bias_current_node)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)

                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None 
        self.v = None 
        self.single_head_key = None 
        self.q_first = None 

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)

    def forward(self, encoded_last_node, ninf_mask, attention_bias_current_node):

        head_num = self.model_params['head_num']

        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)

        q = self.q_first + q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)

        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)
        
        score = score + attention_bias_current_node

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)

        return probs

def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)

    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):

        added = input1 + input2

        transposed = added.transpose(1, 2)

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):

        return self.W2(F.relu(self.W1(input1)))
