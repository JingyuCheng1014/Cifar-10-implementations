import os
import argparse
import matplotlib.pyplot as plt
import glob

# 尝试使用 pandas，如果不可用则回退到 csv
try:
    import pandas as pd
    def read_csv(path):
        return pd.read_csv(path)
except Exception:
    import csv
    def read_csv(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # 转换为 dict of lists，尽量把数值转换为 float
        data = {k: [] for k in rows[0].keys()}
        for r in rows:
            for k, v in r.items():
                try:
                    data[k].append(float(v))
                except Exception:
                    data[k].append(v)
        return data

def plot_from_csv(csv_path, outdir="logs/plots", show=True):
    os.makedirs(outdir, exist_ok=True)
    data = read_csv(csv_path)

    # 支持 pandas DataFrame 或 dict-of-lists
    if hasattr(data, "columns"):
        df = data
        epoch = df.get("epoch") if "epoch" in df.columns else None
        train_loss = df.get("train_loss") if "train_loss" in df.columns else None
        val_loss = df.get("val_loss") if "val_loss" in df.columns else None
        train_acc = df.get("train_acc") if "train_acc" in df.columns else None
        val_acc = df.get("val_acc") if "val_acc" in df.columns else None
    else:
        df = data
        epoch = df.get("epoch")
        train_loss = df.get("train_loss")
        val_loss = df.get("val_loss")
        train_acc = df.get("train_acc")
        val_acc = df.get("val_acc")

    name = os.path.splitext(os.path.basename(csv_path))[0]

    # Loss
    if train_loss is not None or val_loss is not None:
        plt.figure(figsize=(8,5))
        x = epoch if epoch is not None else range(1, len(train_loss)+1 if train_loss is not None else len(val_loss)+1)
        if train_loss is not None:
            plt.plot(x, train_loss, label="train")
        if val_loss is not None:
            plt.plot(x, val_loss, label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss over epochs — {name}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
        out_path = os.path.join(outdir, f"loss_curve_{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show: plt.show()
        plt.close()
        print(f"Saved {out_path}")

    # Accuracy
    if train_acc is not None or val_acc is not None:
        plt.figure(figsize=(8,5))
        x = epoch if epoch is not None else range(1, len(train_acc)+1 if train_acc is not None else len(val_acc)+1)
        if train_acc is not None:
            plt.plot(x, train_acc, label="train")
        if val_acc is not None:
            plt.plot(x, val_acc, label="val")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title(f"Accuracy over epochs — {name}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
        out_path = os.path.join(outdir, f"acc_curve_{name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show: plt.show()
        plt.close()
        print(f"Saved {out_path}")

def plot_all_in_dir(logs_dir, outdir="logs/plots", show=False):
    os.makedirs(outdir, exist_ok=True)
    pattern = os.path.join(logs_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No csv files found in {logs_dir}")
        return
    for f in files:
        print(f"Plotting {os.path.basename(f)} ...")
        try:
            plot_from_csv(f, outdir=outdir, show=show)
        except Exception as e:
            print(f"Failed to plot {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs from CSV (loss/accuracy).")
    parser.add_argument("--dir", "-d", default="logs", help="directory containing csv log files")
    parser.add_argument("--outdir", "-o", default="logs/plots", help="output directory for pngs")
    parser.add_argument("--no-show", action="store_true", help="do not display plots (only save)")
    args = parser.parse_args()

    # 如果不显示图像，使用无界面后端以避免弹窗/阻塞
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    plot_all_in_dir(args.dir, outdir=args.outdir, show=not args.no_show)