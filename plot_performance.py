import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = os.path.join("logs", "performance")
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_csv(csv_path, out_path, x_label):
    # 只用逗号分隔符读取
    df = pd.read_csv(csv_path, sep=",")
    print(f"Read {csv_path} shape: {df.shape}, columns: {df.columns.tolist()}")

    # 依据列数修正列名
    if df.shape[1] == 4:
        df.columns = [x_label, "params_M", "sec_epoch", "accuracy"]
    elif df.shape[1] == 3:
        df.columns = [x_label, "params_M", "sec_epoch"]
        df["accuracy"] = np.nan
    else:
        raise SystemExit(f"Unexpected column count in {csv_path}: {df.shape[1]}")

    # 类型转换
    df[x_label] = pd.to_numeric(df[x_label], errors="coerce")
    df["params_M"] = pd.to_numeric(df["params_M"], errors="coerce")
    df["sec_epoch"] = pd.to_numeric(df["sec_epoch"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    # 计算百分比 accuracy
    df["acc_pct"] = df["accuracy"] * 100.0

    # 排序
    df = df.sort_values(x_label).reset_index(drop=True)

    # 绘图
    fig, ax2 = plt.subplots(figsize=(8,4))
    n = len(df)
    x = np.arange(n)
    bar_width = 0.28

    # sec/epoch 柱
    sec_pos = x + bar_width/2
    sec_bars = ax2.bar(sec_pos, df["sec_epoch"], width=bar_width, color="green",
                       edgecolor="k", linewidth=0.4, label="sec/epoch", zorder=2)
    ax2.set_ylabel("sec/epoch", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # params 柱
    ax1 = ax2.twinx()
    ax1.yaxis.set_label_position("left")
    ax1.yaxis.tick_left()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(True)
    ax1.spines["left"].set_position(("outward", 0))
    ax2.patch.set_alpha(0.0)

    params_pos = x - bar_width/2
    params_bars = ax1.bar(params_pos, df["params_M"], width=bar_width, color="lightgray",
                          edgecolor="k", linewidth=0.4, label="Params (M)", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df[x_label].astype(str))
    ax1.set_xlabel(x_label.capitalize())
    ax1.set_ylabel("Params (M)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Accuracy 折线
    max_params = df["params_M"].max() if not df["params_M"].isna().all() else 1.0
    acc_scaled = (df["acc_pct"].fillna(0) / 100.0) * max_params
    acc_line, = ax1.plot(x, acc_scaled, color="black", marker="o", linewidth=2,
                         label="Accuracy (%)", zorder=20)

    # Accuracy 数值标注
    label_offset = 0.03 * max_params
    for xi, pct in zip(x, df["acc_pct"]):
        if not np.isnan(pct):
            y = (pct / 100.0) * max_params - label_offset
            ax1.text(float(xi), y, f"{pct:.2f}%", ha="center", va="top",
                     color="black", fontsize=10, fontweight="bold", zorder=21)

    # 图例
    handles_left, labels_left = ax1.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    if handles:
        ax1.legend(handles, labels, loc="upper right")

    plt.title(f"{x_label.capitalize()}: Params (gray bars, left) — sec/epoch (green bars, right) — Accuracy (black line, %)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

# 遍历 logs/performance 下所有 csv
for fname in os.listdir(IN_DIR):
    if not fname.endswith(".csv"):
        continue
    csv_path = os.path.join(IN_DIR, fname)
    # 自动推断 x_label
    if fname.startswith("ps"):
        x_label = "patch"
    elif fname.startswith("dim"):
        x_label = "dim"
    elif fname.startswith("depth"):
        x_label = "depth"
    elif fname.startswith("heads"):
        x_label = "heads"
    elif fname.startswith("best"):
        x_label = "best"
    else:
        x_label = "x"
    out_path = os.path.join(OUT_DIR, fname.replace(".csv", "_params_sec_plot.png"))
    plot_csv(csv_path, out_path, x_label)