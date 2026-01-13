import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 如果没有安装，可以通过 pip install tqdm 安装，或者直接删掉相关进度条代码

# 导入你之前写好的模块
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# ================= 配置区域 =================
WEIGHTS_PATH = "vit-dinov2-base.npz"  # 模型权重文件路径
GALLERY_FOLDER = "gallery_images"     # 存放图库图片的文件夹
FEATURE_FILE = "gallery_features.npz" # 保存特征库的文件名
# ===========================================

def load_model():
    """加载模型"""
    print(f"正在加载模型权重: {WEIGHTS_PATH} ...")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"找不到权重文件 {WEIGHTS_PATH}，请确认路径正确！")
    weights = np.load(WEIGHTS_PATH)
    model = Dinov2Numpy(weights)
    print("模型加载完成！")
    return model

def build_gallery(model):
    """
    遍历文件夹，提取特征，建立索引库
    """
    if not os.path.exists(GALLERY_FOLDER):
        print(f"错误：找不到图库文件夹 {GALLERY_FOLDER}")
        return

    image_files = [f for f in os.listdir(GALLERY_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"发现 {len(image_files)} 张图片，开始提取特征...")

    features_list = []
    paths_list = []

    for img_file in tqdm(image_files): # tqdm用于显示进度条
        img_path = os.path.join(GALLERY_FOLDER, img_file)
        
        try:
            # 1. 预处理：注意这里用的是 resize_short_side
            pixel_values = resize_short_side(img_path) # 形状 (1, 3, H, W)
            
            # 2. 提取特征
            # model() 返回的是 (1, 768)
            feat = model(pixel_values) 
            
            # 3. 归一化特征 (这一步对余弦相似度很重要！)
            # 归一化后，向量长度为1。此时 向量A点乘向量B = 余弦相似度
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            
            features_list.append(feat)
            paths_list.append(img_file)
            
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")

    if len(features_list) > 0:
        # 堆叠成大矩阵 (N, 768)
        all_features = np.vstack(features_list)
        all_paths = np.array(paths_list)
        
        # 保存到文件，下次就不用重新提取了
        np.savez(FEATURE_FILE, features=all_features, paths=all_paths)
        print(f"特征库建立完成！已保存到 {FEATURE_FILE}")
        print(f"特征矩阵形状: {all_features.shape}")
    else:
        print("没有成功提取到任何特征。")

def search_image(model, query_img_path, top_k=5):
    """
    搜索相似图片
    """
    # 1. 加载特征库
    if not os.path.exists(FEATURE_FILE):
        print("特征库文件不存在，请先运行 build_gallery()！")
        return

    data = np.load(FEATURE_FILE)
    gallery_features = data['features'] # (N, 768)
    gallery_paths = data['paths']       # (N,)

    # 2. 提取查询图片的特征
    print(f"\n正在搜索图片: {query_img_path}")
    try:
        pixel_values = resize_short_side(query_img_path)
        query_feat = model(pixel_values) # (1, 768)
        
        # 归一化查询向量
        norm = np.linalg.norm(query_feat)
        if norm > 0:
            query_feat = query_feat / norm
            
    except Exception as e:
        print(f"处理查询图片出错: {e}")
        return

    # 3. 计算相似度 (矩阵乘法)
    # (1, 768) @ (768, N) -> (1, N)
    # 因为都归一化了，点积就是余弦相似度
    sims = np.dot(query_feat, gallery_features.T)[0] 

    # 4. 排序并取 Top K
    # argsort 是从小到大排，所以我们要取最后 k 个，然后倒序
    top_indices = np.argsort(sims)[-top_k:][::-1]

    # 5. 显示结果
    print(f"--- 搜索结果 (Top {top_k}) ---")
    for rank, idx in enumerate(top_indices):
        score = sims[idx]
        file_name = gallery_paths[idx]
        print(f"Rank {rank+1}: {file_name} (相似度: {score:.4f})")
    
    return [gallery_paths[i] for i in top_indices]

# ================= 主程序入口 =================
if __name__ == "__main__":
    # 1. 初始化模型
    vit_model = load_model()
    
    # 2. 建立特征库 (如果你是第一次运行，或者加了新图片，取消下面这行的注释)
    build_gallery(vit_model)
    
    # 3. 进行搜索测试 (换成你想搜的图片路径)
    # 假设你有一张叫 test_cat.jpg 的图想搜
    test_image = "./demo_data/cat.jpg" 
    if os.path.exists(test_image):
        search_image(vit_model, test_image)
    else:
        print(f"请指定一张存在的图片进行测试，{test_image} 不存在。")