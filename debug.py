import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

cat_pixel_values = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values)

dog_pixel_values = center_crop("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values)
# ==========================================
# 下面是需要你自己补全的“检查输出”部分
# ==========================================
import os

print("特征提取完成，开始检查结果...")

# 1. 打印一下特征的形状，看看是不是 (1, 768)
print(f"Cat feature shape: {cat_feat.shape}")
print(f"Dog feature shape: {dog_feat.shape}")

# 2. 加载老师给的标准答案 (./demo_data/cat_dog_feature.npy)
ref_path = "./demo_data/cat_dog_feature.npy"
if os.path.exists(ref_path):
    ref_feat = np.load(ref_path) # 标准答案应该是 (2, 768)
    
    # 把我们算出来的猫和狗拼起来，变成 (2, 768)
    my_feat = np.concatenate([cat_feat, dog_feat], axis=0)
    
    # 3. 计算误差 (你的结果 - 标准答案)
    diff = np.abs(my_feat - ref_feat).mean()
    
    print("-" * 30)
    print(f"标准答案形状: {ref_feat.shape}")
    print(f"我的结果形状: {my_feat.shape}")
    print(f"平均误差 (Mean Difference): {diff:.8f}")
    
    # 4. 判断是否合格
    if diff < 1e-5:
        print("✅ 恭喜！结果正确 (Difference is very small).")
    else:
        print("❌ 失败！误差过大，请检查 dinov2_numpy.py 的实现。")
    print("-" * 30)
else:
    print(f"⚠️ 警告：找不到参考文件 {ref_path}，无法进行自动对比。")