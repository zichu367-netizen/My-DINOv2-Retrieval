import numpy as np

from scipy.ndimage import zoom

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class Embeddings:
    def __init__(self, weights):
        """
        NumPy 实现的 Dinov2 Embeddings 层。
        """
        self.hidden_size = 768 # D
        self.patch_size  = 14  # ps

        self.cls_token           = weights["embeddings.cls_token"] # (1, 1, D)
        self.position_embeddings = weights["embeddings.position_embeddings"] # (1, N+1, D)
        self.patch_embed_w       = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T
        self.patch_embed_b       = weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        patches = []
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch = pixel_values[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        # (B, N, C, ps, ps) -> (B, N, C*ps*ps)
        return np.array(patches).transpose(1, 0, 2, 3, 4).reshape(B, -1, C * self.patch_size * self.patch_size)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        处理位置编码的插值 (这是你刚才修复过的部分)
        """
        np_pos_embed = self.position_embeddings
        N = np_pos_embed.shape[1] - 1
        
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        
        if N == h_patches * w_patches:
            return self.position_embeddings

        cls_pos_embed = np_pos_embed[:, :1, :]
        patch_pos_embed = np_pos_embed[:, 1:, :]

        dim = np_pos_embed.shape[-1]
        N_sqrt = int(np.sqrt(N))

        patch_pos_embed = patch_pos_embed.reshape(N_sqrt, N_sqrt, dim)
        zoom_h = h_patches / N_sqrt
        zoom_w = w_patches / N_sqrt
        
        # 双线性插值
        patch_pos_embed = zoom(patch_pos_embed, (zoom_h, zoom_w, 1), order=1)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        
        return np.concatenate((cls_pos_embed, patch_pos_embed), axis=1)

    # =============== 缺失的重点在这里！ ===============
    def __call__(self, pixel_values):
        B, C, H, W = pixel_values.shape
        
        # 1. 像素切块 (Patch Partition)
        patches = self.pixel2patches(pixel_values)
        
        # 2. 线性投影 (Linear Projection)
        # (B, N, Input_dim) @ (Input_dim, Hidden_dim) -> (B, N, Hidden_dim)
        embeddings = np.dot(patches, self.patch_embed_w) + self.patch_embed_b
        
        # 3. 加上分类标记 (CLS Token)
        cls_token = np.tile(self.cls_token, (B, 1, 1))
        embeddings = np.concatenate((cls_token, embeddings), axis=1)
        
        # 4. 加上位置编码 (Position Embeddings)
        pos_embed = self.interpolate_pos_encoding(embeddings, H, W)
        embeddings = embeddings + pos_embed
        
        return embeddings
class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight
        self.bias   = bias
        self.eps    = eps

    def __call__(self, x, ):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale: 
    def __init__(self, lambda1): 
        self.lambda1 = lambda1

    def __call__(self, x): 
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias   = bias

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class SingleHeadAttention:
    def __init__(self, config, prefix, weights):
        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        q = self.q_proj(x) # (B, h*w+1, D)
        k = self.k_proj(x) # (B, h*w+1, D)
        v = self.v_proj(x) # (B, h*w+1, D)
        att = np.matmul(q, k.transpose(0,2,1)) / np.sqrt(self.hidden_size) # (B, h*w+1, h*w+1)
        att = softmax(att)
        out = np.matmul(att, v) # (B, h*w+1, D)
        return self.out_proj(out)

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj   = Linear(q_w, q_b)
        self.k_proj   = Linear(k_w, k_b)
        self.v_proj   = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        # ************* ToDo, multi-head attention *************
        B, N, C = x.shape
        
        # 1. 计算 Q, K, V
        # 经过线性层后形状是 (B, N, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分成多个头 (Multi-Head)
        # 形状变换: (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. 计算注意力分数 (Attention Score) = Q @ K^T / sqrt(d)
        # (B, heads, N, dim) @ (B, heads, dim, N) -> (B, heads, N, N)
        scale = 1.0 / np.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        # 4. 归一化 (Softmax)
        attn = softmax(attn, axis=-1)

        # 5. 加权求和 (Weighted Sum) = Attn @ V
        # (B, heads, N, N) @ (B, heads, N, dim) -> (B, heads, N, dim)
        x = attn @ v

        # 6. 合并头 (Merge Heads)
        # (B, heads, N, dim) -> (B, N, heads, dim) -> (B, N, D)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)

        # 7. 最后的线性输出
        x = self.out_proj(x)
        return x

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(f"{prefix}", weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks     = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm       = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        pos_embed = self.embeddings(pixel_values)
        for blk in self.blocks:
            pos_embed = blk(pos_embed)
        pos_embed = self.norm(pos_embed)
        return pos_embed[:, 0]
