import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FIG 1: ML vs LLM multiclass (fine-tuned, fewer vs more)
# ============================================================

models_ml = ["MLP", "XGBoost", "LightGBM", "Random Forest"]
f1_ml_fewer = [0.1774, 0.4987, 0.4942, 0.4554]
f1_ml_more  = [0.2018, 0.4937, 0.4933, 0.4388]

models_llm = ["DeepSeek", "LLaMA", "Qwen"]
f1_llm_fewer = [0.7479, 0.6789, 0.4546]
f1_llm_more  = [0.7154, 0.7048, 0.5940]

# Combine both graphs
models_all = models_ml + models_llm
f1_fewer_all = f1_ml_fewer + f1_llm_fewer
f1_more_all  = f1_ml_more + f1_llm_more

x = np.arange(len(models_all))
width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, f1_fewer_all, width, label="Fewer samples")
plt.bar(x + width/2, f1_more_all,  width, label="More samples")

plt.xticks(x, models_all, rotation=30, ha="right")
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("Multiclass classification – ML vs LLM (fine-tuned)")
plt.legend()
plt.tight_layout()
plt.savefig("1_fig_multiclass_ml_vs_llm_f1.png", dpi=300)
plt.close()


# ============================================================
# FIG 2: Zero-shot vs Few-shot vs Fine-tuning per LLM
# ============================================================

llm_names = ["DeepSeek", "LLaMA", "Qwen"]


binary_zero = [0.3529, 0.4028, 0.2819]
binary_few  = [0.6564, 0.4542, 0.9067]
binary_ft   = [1.0,    1.0,    1.0] 

x = np.arange(len(llm_names))
width = 0.25

plt.figure(figsize=(8, 4))
plt.bar(x - width, binary_zero, width, label="Zero-shot")
plt.bar(x,         binary_few,  width, label="Few-shot")
plt.bar(x + width, binary_ft,   width, label="Fine-tuned")

plt.xticks(x, llm_names)
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.title("Binary classification – LLM strategies")
plt.legend()
plt.tight_layout()
plt.savefig("2_fig_binary_zero_few_finetune_llm.png", dpi=300)
plt.close()

multi_zero = [0.0065, 0.0126, 0.0120]   
multi_few  = [0.0665, 0.0117, 0.0178]   
multi_ft   = [0.7154, 0.7048, 0.5940]   

plt.figure(figsize=(8, 4))
plt.bar(x - width, multi_zero, width, label="Zero-shot")
plt.bar(x,         multi_few,  width, label="Few-shot")
plt.bar(x + width, multi_ft,   width, label="Fine-tuned")

plt.xticks(x, llm_names)
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("Multiclass classification – LLM strategies")
plt.legend()
plt.tight_layout()
plt.savefig("2_fig_multiclass_zero_few_finetune_llm.png", dpi=300)
plt.close()


# ============================================================
# FIG 3: Heatmap per-class F1 – DeepSeek (balanced subset)
# ============================================================

classes = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "SQL_injection",
    "Password",
    "Vulnerability_scanner",
    "DDoS_TCP",
    "DDoS_HTTP",
    "Uploading",
    "Backdoor",
    "Port_Scanning",
    "XSS",
    "Ransomware",
    "MiTM",
    "Fingerprinting",
]

f1_per_class = [
    0.9982,
    0.4212,
    0.2355,
    0.9784,
    0.9856,
    0.9961,
    0.6604,
    0.9774,
    0.9104,
    0.8668,
    0.3603,
    0.9349,
    0.8915,
    0.4844,
    0.2626,
]

# Heatmap 1D (F1 as color)
plt.figure(figsize=(6, 6))
# 2D matrix of size (n_classes x 1)
data = np.array(f1_per_class).reshape(-1, 1)
plt.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)
plt.colorbar(label="F1-score")
plt.yticks(np.arange(len(classes)), classes)
plt.xticks([0], ["DeepSeek (F1)"])
# Add horizontal lines to separate classes
for i in range(len(classes) + 1):
    plt.axhline(i - 0.5, color="white", linewidth=0.6)
plt.title("Per-class F1-score – DeepSeek (balanced subset)")
plt.tight_layout()
plt.savefig("3_fig_heatmap_deepseek_per_class_f1_balanced.png", dpi=300)
plt.close()

# ============================================================
# FIG 4: Heatmap per-class F1 – DeepSeek (Imbalanced subset)
# ============================================================

classes = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "SQL_injection",
    "Password",
    "Vulnerability_scanner",
    "DDoS_TCP",
    "DDoS_HTTP",
    "Uploading",
    "Backdoor",
    "Port_Scanning",
    "XSS",
    "Ransomware",
    "MiTM",
    "Fingerprinting",
]

f1_per_class = [
    0.9993,
    0.5823,
    0.3015,
    0.9912,
    0.9907,
    0.9947,
    0.7975,
    0.9882,
    0.9193,
    0.8660,
    0.3395,
    0.9225,
    0.8218,
    0.0000,
    0.0217,
]

# Heatmap 1D (F1 as color)
plt.figure(figsize=(6, 6))
# 2D Matrix of size (n_classes x 1)
data = np.array(f1_per_class).reshape(-1, 1)
plt.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)
plt.colorbar(label="F1-score")
plt.yticks(np.arange(len(classes)), classes)
plt.xticks([0], ["DeepSeek (F1)"])
# Add horizontal lines to separate classes
for i in range(len(classes) + 1):
    plt.axhline(i - 0.5, color="white", linewidth=0.6)

plt.title("Per-class F1-score – DeepSeek (Imbalanced subset)")
plt.tight_layout()
plt.savefig("4_fig_heatmap_deepseek_per_class_f1_imbalanced.png", dpi=300)
plt.close()


# ============================================================
# FIG 5: Balanced vs Imbalanced – DeepSeek vs XGBoost
# ============================================================

models_compare = ["XGBoost", "DeepSeek"]
f1_imbalanced = [0.8470, 0.9250]  
f1_balanced   = [0.5169, 0.7423]  

x = np.arange(len(models_compare))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width/2, f1_imbalanced, width, label="Imbalanced inference")
plt.bar(x + width/2, f1_balanced,   width, label="Balanced inference")

plt.xticks(x, models_compare)
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("DeepSeek vs XGBoost – Balanced vs Imbalanced (5.25k samples to train)")
plt.legend()
plt.tight_layout()
plt.savefig("5_fig_deepseek_xgb_balanced_imbalanced_5k.png", dpi=300)
plt.close()

# ============================================================
# FIG 6: Balanced vs Imbalanced – DeepSeek vs XGBoost
# ============================================================

models_compare = ["XGBoost", "DeepSeek"]
f1_imbalanced = [0.8472, 0.9122] 
f1_balanced   = [0.5147, 0.7239] 

x = np.arange(len(models_compare))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width/2, f1_imbalanced, width, label="Imbalanced inference")
plt.bar(x + width/2, f1_balanced,   width, label="Balanced inference")

plt.xticks(x, models_compare)
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("DeepSeek vs XGBoost – Balanced vs Imbalanced (10.5k samples to train)")
plt.legend()
plt.tight_layout()
plt.savefig("6_fig_deepseek_xgb_balanced_imbalanced_10k.png", dpi=300)
plt.close()