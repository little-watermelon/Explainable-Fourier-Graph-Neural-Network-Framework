import torch
import numpy as np
import os
from utils.utils import load_model


def extract_fgnn_weights(model_dir, epoch, save_path=None):

    print(f"Loading model from {model_dir} for epoch {epoch}...")

    # 1. Load the model using the load_model function defined in utils.py
    model = load_model(model_dir, epoch)
    if model is None:
        print("Failed to load the model. Please check if model_dir and epoch are correct.")
        return None

    print("Model loaded successfully!\n")

    model.eval()
    model.cpu()

    w1 = model.w1.detach().numpy()
    w2 = model.w2.detach().numpy()
    w3 = model.w3.detach().numpy()

    token_emb = model.embeddings.detach().numpy()
    proj_emb = model.embeddings_10.detach().numpy()

    # 打印权重形状信息
    print("--- Extracted Weight Dimensions ---")
    print(f"Frequency-domain filter Layer 1 (w1): {w1.shape}")
    print(f"Frequency-domain filter Layer 2 (w2): {w2.shape}")
    print(f"Frequency-domain filter Layer 3 (w3): {w3.shape}")
    print(f"Global Token Embedding (embeddings): {token_emb.shape}")
    print(f"Output Projection Embedding (embeddings_10): {proj_emb.shape}\n")


    extracted_weights = {
        'w1': w1,
        'w2': w2,
        'w3': w3,
        'token_embeddings': token_emb,
        'projection_embeddings': proj_emb
    }

    if save_path:
        np.savez(save_path, **extracted_weights)
        print(f"✅ All weights have been successfully saved to: {save_path}")

    return extracted_weights


if __name__ == '__main__':
    # ================= 配置区 =================
    # Set the paths here based on the args used when running main.py
    DATASET_NAME = 'SHALEGAS'
    TARGET_EPOCH = 99  # Assuming the last saved model is 99_dhfm.pt

    MODEL_DIR = os.path.join('output', DATASET_NAME, 'train')
    SAVE_FILE = f'fouriergnn_{DATASET_NAME}_weights.npz'
    # ==========================================

    weights = extract_fgnn_weights(model_dir=MODEL_DIR, epoch=TARGET_EPOCH, save_path=SAVE_FILE)
