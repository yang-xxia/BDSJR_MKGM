import pickle
import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class mamba_1024(nn.Module):
    def __init__(self):
        super(mamba_1024,self).__init__()
        self.layer= AutoModel.from_pretrained("MambaVision-B-1K", trust_remote_code=True)
        self.fc = nn.Linear(1024, 4)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
    
    def forward(self,x):
        out_avg_pool, features=self.layer(x)
        x=self.fc(out_avg_pool)
        return x
    
    def get_config_optim(self, lr, lrp):
        return [

                {'params': self.layer.parameters(), 'lr': lr * lrp}

                ]


class mamba(nn.Module):
    def __init__(self):
        super(mamba,self).__init__()
        self.layer= AutoModel.from_pretrained("MambaVision-B-1K", trust_remote_code=True)
        self.conv_layer = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 29)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
    
    def forward(self,x):
        out_avg_pool, features=self.layer(x)
        x = self.conv_layer(features[3])
        x = self.global_avg_pool(x)
        x = self.fc(out_avg_pool)
        return x
    
    def get_config_optim(self, lr, lrp):
        return [

                {'params': self.layer.parameters(), 'lr': lr * lrp}

                ]
    

class TransformerFusionModule(nn.Module):
    def __init__(self, mamba_dim=640, fast_dim=1000, kg_dim=1000, embed_dim=512, num_heads=1):
        super(TransformerFusionModule, self).__init__()
        
        # 将输入投影到 embedding 维度
        self.q_proj = nn.Linear(mamba_dim, embed_dim)
        self.k_proj = nn.Linear(fast_dim, embed_dim)
        self.v_proj = nn.Linear(kg_dim, embed_dim)
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, mamba_feature, fast_x, KG_x):
        batch_size = mamba_feature.size(0)
        
        # 复制 KG_x 到 (batch_size, 29, 1000)
        KG_x = KG_x.expand(batch_size, -1, -1)  # (batch_size, 29, 1000)

        # 投影 mamba_feature, fast_x 和 KG_x 到相同的嵌入维度
        Q = self.q_proj(mamba_feature).unsqueeze(1)  # (batch_size, 1, embed_dim)
        Q = Q.repeat(1, 29, 1)                        # (batch_size, 29, embed_dim)
        K = self.k_proj(fast_x).expand(batch_size, -1, -1)  # (batch_size, 29, embed_dim)
        V = self.v_proj(KG_x)                             # (batch_size, 29, embed_dim)

        # 使用多头注意力融合
        attn_output, attn_weights = self.multihead_attn(Q, K, V)  # (batch_size, 29, embed_dim)
        # 输出层
        output = self.out_proj(attn_output)  # (batch_size, 29, embed_dim)
        output = output[:, 0, :]  # 取第一个时间步的输出作为最终输出

        return output



class ROIToGCNFeatureMapper(nn.Module):
    def __init__(self, input_channels=1024, input_height=14, input_width=14, num_nodes=29, node_features=100):
        super(ROIToGCNFeatureMapper, self).__init__()
        
        # 计算 flatten 后的特征大小
        self.flatten_size = input_channels * input_height * input_width
        self.num_nodes = num_nodes
        self.node_features = node_features
        
        # 定义全连接层，将 flatten_size 映射到 num_nodes * node_features
        self.fc = nn.Linear(self.flatten_size, num_nodes * node_features)

    def forward(self, x):
        # 动态获取 batch size
        batch_size = x.size(0)
        
        # Step 1: 将 (input_channels, input_height, input_width) 展开成一个大向量
        x = x.view(batch_size, -1)  # (batch_size, flatten_size)
        
        # Step 2: 通过全连接层，将特征映射为 (batch_size, num_nodes * node_features)
        x = self.fc(x)
        
        # Step 3: Reshape 为 (batch_size, num_nodes, node_features)
        x = x.view(batch_size, self.num_nodes, self.node_features)
        
        return x



# 在 rcnn_mamba 类中使用 FeatureFusionModule
class rcnn_mamba(nn.Module):
    def __init__(self):
        super(rcnn_mamba, self).__init__()
        self.layer = AutoModel.from_pretrained("MambaVision-B-1K", trust_remote_code=True)
        self.RGFM = ROIToGCNFeatureMapper()
        self.gc1 = GraphConvolution(100, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.relu = nn.LeakyReLU(0.2)
        self.KG_gc1 = GraphConvolution(100, 1024)
        self.KG_gc2 = GraphConvolution(1024, 1024)

        self.fusion_module = EnhancedFusionModule()  # 使用自定义的特征融合模块
        
        self.fc = nn.Linear(1024, 29)  # 确保输出维度与类别数匹配
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.adj = torch.from_numpy(pickle.load(open('dataset/T-G-adj.pkl', 'rb'))['adj']).float().cuda()
        self.KG_inp = torch.from_numpy(pickle.load(open('dataset/ent_emb.pkl', 'rb'))).float().cuda()
        self.adj_rcnn = torch.from_numpy(pickle.load(open('dataset/co_occurrence_matrix.pkl', 'rb'))['adj']).float().cuda()


    def forward(self, x, fast_pool):
        outputs = self.layer(x.float().cuda())
        if isinstance(outputs, tuple):
            mamba_feature = outputs[0]  # 取第一个元素作为输出
        else:
            mamba_feature = outputs

        fast_inp = self.RGFM(fast_pool)
        fast_x = self.gc1(fast_inp, self.adj_rcnn)
        fast_x = self.relu(fast_x)
        fast_x = self.gc2(fast_x, self.adj_rcnn)  # (b, 29, 1000)

        KG_x = self.KG_gc1(self.KG_inp, self.adj)
        KG_x = self.relu(KG_x)
        KG_x = self.KG_gc2(KG_x, self.adj)  # (batch_size, 29, 1000)
        KG_x = KG_x.unsqueeze(0).repeat(mamba_feature.size(0), 1, 1)  # 复制到 batch_size

        fused_feature = self.fusion_module(mamba_feature, fast_x, KG_x)
        
        output = self.fc(fused_feature)
        return output  # (batch_size, 29)
    
    
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.layer.parameters(), 'lr': lr * lrp},
            {'params': self.fc.parameters(), 'lr': lr}  # 添加对fc层的学习率配置
        ]




class EnhancedFusionModule(nn.Module):
    def __init__(self, embed_dim=1024):
        super(EnhancedFusionModule, self).__init__()
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        
        # 可学习缩放因子，初始值设置为0
        self.fast_scale = nn.Parameter(torch.tensor(0.0))
        self.kg_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, mamba_feature, fast_x, KG_x):
        batch_size = mamba_feature.size(0)

        # 平均池化 fast 和 KG 特征
        fast_emb = fast_x.mean(dim=1).unsqueeze(1)  # (batch_size, 1, embed_dim)
        kg_emb = KG_x.expand(batch_size, -1, -1).mean(dim=1).unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # 自注意力机制
        combined_features = torch.cat([mamba_feature.unsqueeze(1), fast_emb, kg_emb], dim=1)  # (batch_size, 3, embed_dim)
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        
        # 融合特征
        weighted_fast = self.fast_scale * attn_output[:, 1, :]
        weighted_kg = self.kg_scale * attn_output[:, 2, :]
        fused_feature = mamba_feature + weighted_fast + weighted_kg  # (batch_size, embed_dim)
        
        # Dropout
        fused_feature = self.dropout(fused_feature)

        return fused_feature