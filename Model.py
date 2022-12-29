import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from Config import config
from utils import getConnLabelMapping

_INF = -1e30
EPS = 1e-8

def multi_perspective_match(vector1, vector2, weight):
    assert vector1.size(0) == vector2.size(0)
    assert weight.size(1) == vector1.size(2)

    # (batch, seq_len, 1)
    similarity_single = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)

    # (1, 1, num_perspectives, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspectives, hidden_size)
    vector1 = weight * vector1.unsqueeze(2)
    vector2 = weight * vector2.unsqueeze(2)

    similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)

    return similarity_single, similarity_multi


def multi_perspective_match_pairwise(vector1, vector2, weight):
    num_perspectives = weight.size(0)

    # (1, num_perspectives, 1, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(2)

    # (batch, num_perspectives, seq_len*, hidden_size)
    vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
    vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspectives, -1, -1)

    # (batch, num_perspectives, seq_len*, 1)
    vector1_norm = vector1.norm(p=2, dim=3, keepdim=True)
    vector2_norm = vector2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspectives, seq_len1, seq_len2)
    mul_result = torch.matmul(vector1, vector2.transpose(2, 3))
    norm_value = vector1_norm * vector2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspectives)
    return (mul_result / norm_value.clamp(min=EPS)).permute(0, 2, 3, 1)


def masked_max(vector, mask, dim, keepdim=False):
    replaced_vector = vector.masked_fill(mask==0, _INF) if mask is not None else vector
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector, mask, dim, keepdim=False):
    replaced_vector = vector.masked_fill(mask==0, 0.0) if mask is not None else vector
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=EPS)


def masked_softmax(vector, mask, dim=-1):
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill(mask==0, _INF)
        result = F.softmax(masked_vector, dim=dim)
    return result


class BiMpmMatching(nn.Module):
    def __init__(self, hidden_dim, num_perspectives=16, share_weights_between_directions=True):
        super(BiMpmMatching, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_perspectives = num_perspectives

        def create_parameter():  # utility function to create and initialize a parameter
            param = nn.Parameter(torch.zeros(num_perspectives, hidden_dim))
            nn.init.kaiming_normal_(param)
            return param

        def share_or_create(weights_to_share):  # utility function to create or share the weights
            return weights_to_share if share_weights_between_directions else create_parameter()

        output_dim = 2  # used to calculate total output dimension, 2 is for cosine max and cosine min
        #  with_full_match:
        self.full_forward_match_weights = create_parameter()
        self.full_forward_match_weights_reversed = share_or_create(self.full_forward_match_weights)
        self.full_backward_match_weights = create_parameter()
        self.full_backward_match_weights_reversed = share_or_create(self.full_backward_match_weights)
        output_dim += (num_perspectives + 1) * 2

        # with_maxpool_match:
        self.maxpool_match_weights = create_parameter()
        output_dim += num_perspectives * 2

        # with_attentive_match:
        self.attentive_match_weights = create_parameter()
        self.attentive_match_weights_reversed = share_or_create(self.attentive_match_weights)
        output_dim += num_perspectives + 1

        # with_max_attentive_match:
        self.max_attentive_match_weights = create_parameter()
        self.max_attentive_match_weights_reversed = share_or_create(self.max_attentive_match_weights)
        output_dim += num_perspectives + 1

        self.output_dim = output_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, context_1, mask_1, context_2, mask_2):
        assert (not mask_2.requires_grad) and (not mask_1.requires_grad)
        assert context_1.size(-1) == context_2.size(-1) == self.hidden_dim

        # (batch,)
        len_1 = mask_1.sum(dim=1).long()
        len_2 = mask_2.sum(dim=1).long()

        # explicitly set masked weights to zero
        # (batch_size, seq_len*, hidden_dim)
        context_1 = context_1 * mask_1.unsqueeze(-1)
        context_2 = context_2 * mask_2.unsqueeze(-1)

        # array to keep the matching vectors for the two sentences
        matching_vector_1 = []
        matching_vector_2 = []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = F.cosine_similarity(context_1.unsqueeze(-2), context_2.unsqueeze(-3), dim=3)

        # (batch, seq_len*, 1)
        cosine_max_1 = masked_max(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_1 = masked_mean(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_max_2 = masked_max(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_2 = masked_mean(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)

        matching_vector_1.extend([cosine_max_1, cosine_mean_1])
        matching_vector_2.extend([cosine_max_2, cosine_mean_2])

        # Step 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence

        # (batch, 1, hidden_dim)
        last_position_1 = (len_1 - 1).clamp(min=0)
        last_position_1 = last_position_1.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
        last_position_2 = (len_2 - 1).clamp(min=0)
        last_position_2 = last_position_2.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)

        context_1_forward_last = context_1.gather(1, last_position_1)
        context_2_forward_last = context_2.gather(1, last_position_2)
        context_1_backward_last = context_1[:, 0:1, :]
        context_2_backward_last = context_2[:, 0:1, :]

        # (batch, seq_len*, num_perspectives)
        matching_vector_1_forward_full = multi_perspective_match(context_1,
                                                                context_2_forward_last,
                                                                self.full_forward_match_weights)
        matching_vector_2_forward_full = multi_perspective_match(context_2,
                                                                context_1_forward_last,
                                                                self.full_forward_match_weights_reversed)
        matching_vector_1_backward_full = multi_perspective_match(context_1,
                                                                context_2_backward_last,
                                                                self.full_backward_match_weights)
        matching_vector_2_backward_full = multi_perspective_match(context_2,
                                                                context_1_backward_last,
                                                                self.full_backward_match_weights_reversed)

        matching_vector_1.extend(matching_vector_1_forward_full)
        matching_vector_1.extend(matching_vector_1_backward_full)
        matching_vector_2.extend(matching_vector_2_forward_full)
        matching_vector_2.extend(matching_vector_2_backward_full)

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.

        # (batch, seq_len1, seq_len2, num_perspectives)
        matching_vector_max = multi_perspective_match_pairwise(context_1,
                                                                context_2,
                                                                self.maxpool_match_weights)

        # (batch, seq_len*, num_perspectives)
        matching_vector_1_max = masked_max(matching_vector_max,
                                            mask_2.unsqueeze(-2).unsqueeze(-1),
                                            dim=2)
        matching_vector_1_mean = masked_mean(matching_vector_max,
                                                mask_2.unsqueeze(-2).unsqueeze(-1),
                                                dim=2)
        matching_vector_2_max = masked_max(matching_vector_max.permute(0, 2, 1, 3),
                                            mask_1.unsqueeze(-2).unsqueeze(-1),
                                            dim=2)
        matching_vector_2_mean = masked_mean(matching_vector_max.permute(0, 2, 1, 3),
                                                mask_1.unsqueeze(-2).unsqueeze(-1),
                                                dim=2)

        matching_vector_1.extend([matching_vector_1_max, matching_vector_1_mean])
        matching_vector_2.extend([matching_vector_2_max, matching_vector_2_mean])


        # Step 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_2 = context_2.unsqueeze(-3) * cosine_sim.unsqueeze(-1)

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_1 = context_1.unsqueeze(-2) * cosine_sim.unsqueeze(-1)

        # (batch, seq_len*, hidden_dim)
        att_mean_2 = masked_softmax(att_2.sum(dim=2), mask_1.unsqueeze(-1))
        att_mean_1 = masked_softmax(att_1.sum(dim=1), mask_2.unsqueeze(-1))

        # (batch, seq_len*, num_perspectives)
        matching_vector_1_att_mean = multi_perspective_match(context_1,
                                                                att_mean_2,
                                                                self.attentive_match_weights)
        matching_vector_2_att_mean = multi_perspective_match(context_2,
                                                                att_mean_1,
                                                                self.attentive_match_weights_reversed)
        matching_vector_1.extend(matching_vector_1_att_mean)
        matching_vector_2.extend(matching_vector_2_att_mean)

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.

        # (batch, seq_len*, hidden_dim)
        att_max_2 = masked_max(att_2, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2)
        att_max_1 = masked_max(att_1.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2)

        # (batch, seq_len*, num_perspectives)
        matching_vector_1_att_max = multi_perspective_match(context_1,
                                                            att_max_2,
                                                            self.max_attentive_match_weights)
        matching_vector_2_att_max = multi_perspective_match(context_2,
                                                            att_max_1,
                                                            self.max_attentive_match_weights_reversed)

        matching_vector_1.extend(matching_vector_1_att_max)
        matching_vector_2.extend(matching_vector_2_att_max)

        return matching_vector_1, matching_vector_2

class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1, activation="relu"):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        
        # init
        scale = 1/input_dim**0.5
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.constant_(layer.bias[:input_dim], 0.0)
            nn.init.constant_(layer.bias[input_dim:], 1.0)

    def forward(self, x):
        for layer in self.layers:
            o, g = layer(x).chunk(2, dim=-1)
            o = self.activation(o)
            g = F.sigmoid(g)
            x = g * x + (1 - g) * o
        return x

class CnnHighway(nn.Module):
    def __init__(self, input_dim, filters, output_dim, num_highway=1, activation="relu", projection_location="after_highway", layer_norm=False):
        super().__init__()

        assert projection_location in ["after_cnn", "after_highway"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_location = projection_location

        self.activation = nn.ReLU()
        # Create the convolutions
        self.convs = nn.ModuleList()
        for i, (width, num) in enumerate(filters):
            conv = nn.Conv1d(in_channels=input_dim, out_channels=num, kernel_size=width, bias=True)
            self.convs.append(conv)

        # Create the highway layers
        num_filters = sum(num for _, num in filters)
        if projection_location == 'after_cnn':
            highway_dim = output_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self.highways = Highway(highway_dim, num_highway, activation=activation)

        # Projection layer: always num_filters -> output_dim
        self.proj = nn.Linear(num_filters, output_dim)

        # And add a layer norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        # init
        scale = 1/num_filters**0.5
        for layer in self.convs:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.proj.weight, 0.0, scale)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x, mask):

        # convolutions want (batch_size, input_dim, num_characters)
        x = x.transpose(1, 2)

        output = []
        for conv in self.convs:
            c = conv(x)
            c = torch.max(c, dim=-1)[0]
            c = self.activation(c)
            output.append(c)

        # (batch_size, n_filters)
        output = torch.cat(output, dim=-1)

        if self.projection_location == 'after_cnn':
            output = self.proj(output)

        # apply the highway layers (batch_size, highway_dim)
        output = self.highways(output)

        if self.projection_location == 'after_highway':
            # final projection  (batch_size, output_dim)
            output = self.proj(output)

        # apply layer norm if appropriate
        if self.layer_norm:
            output = self.layer_norm(output)

        return output

    def get_output_dim(self):
        return self.output_dim


class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.backbone = AutoModel.from_pretrained(config.backbone)
        # ------------------------ # 
        # 选择骨干网络,决定输出维度
        # ------------------------ # 
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base" or config.backbone == "microsoft/deberta-v3-base":
            output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large" or config.backbone == "microsoft/deberta-v3-large":
            output_dim = 1024
        
        if config.modelingMethod == "interaction":
            self.bimpm = BiMpmMatching(output_dim)
            output_dim += self.bimpm.get_output_dim()
            self.conv_layer = CnnHighway(
                input_dim=output_dim,
                output_dim=256,
                filters=[(1, 64), (2, 64)], # the shortest length is 2
                num_highway=1,
                activation="relu",
                layer_norm=False)
        # ------------------------ # 
        # 根据建模方式，确定输出维度
        # ------------------------ # 
        if config.modelingMethod == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif config.modelingMethod == "prompt":
            self.connLabelMapping = getConnLabelMapping()
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, len(self.connLabelMapping)),
            )

        elif config.modelingMethod == "interaction":
            self.classifier = nn.Sequential(
                nn.Linear(2*output_dim, output_dim),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(output_dim, 4),
            )

            # self.classifier = nn.Sequential(
            #     nn.Linear(2*output_dim, output_dim),
            #     nn.Dropout(),
            #     nn.ReLU(),
            #     nn.Linear(output_dim, 4),
            # )
    def forward(self, arg):
        # ------------------------ #
        # 输入骨干网络进行特征提取
        # ------------------------ #
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
            or config.backbone == "microsoft/deberta-v3-base" or config.backbone == "microsoft/deberta-v3-large":
            attention_mask = arg[2]
            res = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            attention_mask = arg[1]
            res = self.backbone(input_ids=arg[0], attention_mask=arg[1])

        # ------------------------ #
        # 根据建模方式进行特征交互
        # ------------------------ #
        if config.modelingMethod == "interaction":
            arg1_matched_feats, arg2_matched_feats = self.bimpm(res.last_hidden_state[:, :config.max_length, :], attention_mask[:, :config.max_length],
                                                                res.last_hidden_state[:, config.max_length:, :], attention_mask[:, config.max_length:])
            
            arg1_matched_feats = torch.cat(arg1_matched_feats, dim=2)
            arg2_matched_feats = torch.cat(arg2_matched_feats, dim=2)
            
            arg1_self_attned_feats = torch.cat([res.last_hidden_state[:, :config.max_length, :], arg1_matched_feats], dim=2)
            arg2_self_attned_feats = torch.cat([res.last_hidden_state[:, config.max_length:, :], arg2_matched_feats], dim=2)

            # arg1_conv = self.conv_layer(arg1_self_attned_feats, attention_mask[:, :config.max_length])
            # arg2_conv = self.conv_layer(arg2_self_attned_feats, attention_mask[:, config.max_length:])


        # ------------------------ #
        # 根据建模方式输入分类器
        # ------------------------ #
        if config.modelingMethod == "classification":
            out = torch.mean(res.last_hidden_state, dim=1)
            out = self.classifier(out)
        elif config.modelingMethod == "prompt":
            # 首先预测连接词，取出连接词位置的特征向量
            tmp_out = res.last_hidden_state[:, config.max_length, :]
            tmp_out = self.classifier(tmp_out)
            # 根据概率分布将conn的概率转换为label的概率
            out = torch.zeros((tmp_out.size()[0], 4)).to(config.device)
            for i, key in enumerate(self.connLabelMapping.keys()):
                for j, value in enumerate(self.connLabelMapping[key]):
                    out[:, j] += tmp_out[:, i] * value
        elif config.modelingMethod == "interaction":
            arg1_self_attned_feats = torch.mean(arg1_self_attned_feats, dim=1)
            arg2_self_attned_feats = torch.mean(arg2_self_attned_feats, dim=1)
            out = self.classifier(torch.cat([arg1_self_attned_feats, arg2_self_attned_feats], dim=1))
            # out = self.classifier(torch.cat([arg1_conv, arg2_conv], dim=1))
        return out