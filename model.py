import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MultimodalGCN(nn.Module):
    def __init__(self, 
                 text_dim=768, 
                 graph_dim=300, 
                 hidden_dim=256,
                 num_classes=3):
        super().__init__()
        
        # 文本模态处理层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 图卷积网络层
        self.gc1 = GraphConvolution(graph_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # 多模态融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, text_feats, adj_matrix, graph_feats):
        # 文本特征处理
        text_out = F.relu(self.text_proj(text_feats))
        
        # 图特征处理
        graph_out = F.relu(self.gc1(graph_feats, adj_matrix))
        graph_out = F.relu(self.gc2(graph_out, adj_matrix))
        
        # 多模态融合
        combined = torch.cat([text_out, graph_out], dim=-1)
        fused = F.relu(self.fusion(combined))
        
        # 分类输出
        return self.classifier(fused)

class GraphConvolution(nn.Module):
    """简单的图卷积层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        x = torch.matmul(adj, x)  # 邻接矩阵传播
        x = self.linear(x)
        return x


def load_review_data(csv_path):
    import chardet
    # 检测文件编码
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())
    
    # 使用检测到的编码读取
    df = pd.read_csv(csv_path, encoding=result['encoding'])
    
    # 提取文本特征
    texts = df['review_full'].tolist()
    
    # 使用BERT提取文本特征（新增代码）
    from transformers import BertModel, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # 文本编码
    inputs = tokenizer(texts, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")
    
    # 获取最后一层隐藏状态
    with torch.no_grad():
        outputs = model(**inputs)
    
    text_features = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token的特征
    
    # 构建图结构（基于文本相似度的图构建）
    from sklearn.neighbors import kneighbors_graph
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    
    # 使用BERT特征计算相似度
    bert_features = text_features.numpy()
    
    # 1. 构建邻接矩阵（基于余弦相似度的KNN图）
    similarity_matrix = cosine_similarity(bert_features)
    
    # 生成稀疏邻接矩阵
    adj_sparse = kneighbors_graph(similarity_matrix, 
                                n_neighbors=5, 
                                mode='connectivity', 
                                metric='precomputed',
                                include_self=False)
    
    # 转换为稠密矩阵并对称化
    adj_dense = adj_sparse.toarray()
    adj_matrix = torch.FloatTensor((adj_dense + adj_dense.T > 0).astype(float))
    
    # 2. 构建图节点特征（使用PCA降维）
    # 动态计算最大可用维度
    max_components = min(len(texts)-1, 300)  # 取样本数-1和300的较小值
    pca = PCA(n_components=max_components)
    graph_features = torch.FloatTensor(pca.fit_transform(bert_features))
    
    return text_features, adj_matrix, graph_features, texts

def save_predictions(texts, predictions, output_path):
    """保存预测结果到CSV"""
    results = pd.DataFrame({
        'review_text': texts,
        'prediction': predictions.argmax(axis=1)
    })
    results.to_csv(output_path, index=False)

    # 加载数据
text_feats, adj, graph_feats, texts = load_review_data("data/New_Delhi_reviews.csv")

# 初始化模型
model = MultimodalGCN()

# 运行预测
outputs = model(text_feats, adj, graph_feats)

# 保存结果
save_predictions(texts, outputs.detach().numpy(), "predictions.csv")