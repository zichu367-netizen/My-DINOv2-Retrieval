# download_data.py
import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor

def download_one(url, save_folder, idx):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            # 保存为 00001.jpg 这种格式
            with open(os.path.join(save_folder, f"{idx:05d}.jpg"), "wb") as f:
                f.write(resp.content)
            print(f"Downloaded {idx}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images(csv_path="data.csv", save_folder="gallery_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 读取csv，这里假设没有表头，第一列是url
    # 如果有表头，请把 header=None 改为 header=0
    df = pd.read_csv(csv_path, header=None) 
    urls = df[0].tolist() # 获取第一列

    # 使用多线程加速下载
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, url in enumerate(urls):
            executor.submit(download_one, url, save_folder, i)

if __name__ == "__main__":
    # 运行前确保你有 data.csv
    download_images() 
    pass