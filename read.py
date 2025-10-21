# read_from4col.py  —— 读取形如 ['MDD', ch1, ch2, ch3] 的 (N,4) object npy，自动5折分层导出GFCC分片
import os
import re
import json
import argparse
import numpy as np
from typing import List, Sequence, Tuple, Any, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow

# =========================
# 固定参数 / 默认路径（可用CLI覆盖）
# =========================
_ALL_DATA_PATH = "./all_data.npy"
_OUT_ROOT = "./slices"
_CHANNELS = [1, 2, 3]   # 1-based：列1..3 -> 三个通道
_FS = 250               # 采样率固定为 250Hz

# GFCC 帧长/帧移：保持与原逻辑一致（16/8 个采样点 ≈ 64/32ms）
_WIN_SAMP = 32
_HOP_SAMP = 16
_GFCC_WIN_LEN_S = _WIN_SAMP / _FS
_GFCC_WIN_HOP_S = _HOP_SAMP / _FS
_NUM_CEPS, _NFILTS, _NFFT = 64, 64, 1024
window = SlidingWindow(win_len=_GFCC_WIN_LEN_S, win_hop=_GFCC_WIN_HOP_S, win_type="hamming")

# 切片窗口：6s 窗、3s 步（overlap）；nonoverlap 用 6s 步
_WINDOW_SEC  = 6.0
_STRIDE_SEC  = 3.0
_WINDOW_SIZE = int(_WINDOW_SEC * _FS)   # 1500
_STRIDE      = int(_STRIDE_SEC * _FS)   # 750

# 导出
_SHARD_SIZE = 2048
_DTYPE = np.float32

# =========================
# 工具：解析 (N,4) 结构
# =========================
def _label_from_str(s: str) -> int:
    s_up = str(s).strip().upper()
    if s_up in {"MDD", "DEP", "DEPRESSION"}:
        return 1
    if s_up in {"HC", "HEALTHY", "CONTROL", "CTL", "NORMAL"}:
        return 0
    raise ValueError(f"无法识别的标签字符串: {s!r}（期望 MDD/HC/Healthy/Control 等）")

def _label_name(y: int) -> str:
    return "MDD" if int(y) == 1 else "HC"

def _preprocess_from_4col(all_path: str, channels: Sequence[int]) -> Tuple[List[np.ndarray], np.ndarray, int, int]:
    """
    读取 (N,4) object 数组；每行：['MDD' or 'HC', ch1(1D), ch2(1D), ch3(1D)]
    返回：
      processed: List[np.ndarray]，每 trial 形状 [C, L]
      labels   : np.ndarray[int]，每 trial 标签 0/1
      n_mdd, n_hc
    """
    data = np.load(all_path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{all_path} 需要是 (N,4) 的object数组（label+3通道），实际 shape={data.shape}")
    N = data.shape[0]

    processed, labels = [], []
    for i in range(N):
        row = data[i]
        lab = _label_from_str(row[0])
        # 选择通道（1-based -> row[1], row[2], row[3]）
        sigs = []
        for cj in channels:
            col = cj  # 因为第0列是标签，所以第1列就是通道1
            if col < 1 or col >= row.shape[0]:
                raise IndexError(f"第 {i} 行：请求通道列 {cj} 越界（应在 1..{row.shape[0]-1}）")
            arr = np.asarray(row[col], dtype=np.float32).ravel()
            arr = StandardScaler().fit_transform(arr.reshape(-1, 1)).ravel()
            sigs.append(arr)
        L = min(len(x) for x in sigs)
        processed.append(np.vstack([x[:L] for x in sigs]))  # [C, L]
        labels.append(lab)

    labels = np.asarray(labels, dtype=np.int64)
    n_mdd = int((labels == 1).sum())
    n_hc  = int((labels == 0).sum())
    return processed, labels, n_mdd, n_hc

# =========================
# GFCC 与切窗
# =========================
def _segment_to_gfcc(seg_C_L: np.ndarray) -> np.ndarray:
    feats = []
    for ch in seg_C_L:
        f = gfcc(
            sig=ch.astype(np.float32, copy=False),
            fs=_FS, num_ceps=_NUM_CEPS, nfilts=_NFILTS, nfft=_NFFT,
            window=window, pre_emph=True, pre_emph_coeff=0.97,
            scale="constant", dct_type=2, lifter=22
        )
        feats.append(f)
    return np.stack(feats, axis=0)  # [C, T, num_ceps]

def _iter_gfcc_segments(processed, labels, idx_list, overlap: bool):
    """
    迭代给定 idx_list（受试者索引）的所有窗口。
    返回： (seg, lab, subject_id_1based)
    """
    step = _STRIDE if overlap else _WINDOW_SIZE
    for i in idx_list:
        trial = processed[i]  # [C, L]
        L = trial.shape[1]
        if L < _WINDOW_SIZE:
            continue
        sid = int(i) + 1  # 1-based 受试者编号
        for st in range(0, L - _WINDOW_SIZE + 1, step):
            seg = trial[:, st:st + _WINDOW_SIZE]
            yield _segment_to_gfcc(seg), int(labels[i]), sid

def _save_npz_shards(pairs_iter, out_dir: str, prefix: str, counts: Dict[int, int] | None = None):
    """
    将 (seg, lab, sid) 写入压缩分片。若提供 counts，则统计每个 sid 的窗口数。
    额外：在分片中写入 subjects（与 labels 同长度的 1-based 受试者编号）
    """
    os.makedirs(out_dir, exist_ok=True)
    buf_x, buf_y, buf_sid = [], [], []
    shard_id, total = 0, 0

    for item in pairs_iter:
        # 你的 _iter_gfcc_segments 已经保证三元组 (seg, lab, sid)
        if len(item) == 3:
            seg, lab, sid = item
        else:
            # 兼容异常情况：没有 sid 的旧生成器
            seg, lab = item
            sid = -1

        # 可选：实时统计每个受试者的窗口数
        if counts is not None:
            counts[sid] = counts.get(sid, 0) + 1

        buf_x.append(np.asarray(seg, dtype=_DTYPE))
        buf_y.append(int(lab))
        buf_sid.append(int(sid))  # ← 收集 sid

        if len(buf_x) >= _SHARD_SIZE:
            x = np.stack(buf_x, axis=0)                   # [B,C,T,num_ceps]
            y = np.asarray(buf_y, dtype=np.int64)         # [B]
            s = np.asarray(buf_sid, dtype=np.int64)       # [B] 1-based subject ids
            np.savez_compressed(
                os.path.join(out_dir, f"{prefix}_{shard_id:05d}.npz"),
                segments=x, labels=y, subjects=s          # ← 写入 subjects
            )
            total += len(buf_x)
            buf_x.clear(); buf_y.clear(); buf_sid.clear()
            shard_id += 1

    # 尾分片
    if buf_x:
        x = np.stack(buf_x, axis=0)
        y = np.asarray(buf_y, dtype=np.int64)
        s = np.asarray(buf_sid, dtype=np.int64)
        np.savez_compressed(
            os.path.join(out_dir, f"{prefix}_{shard_id:05d}.npz"),
            segments=x, labels=y, subjects=s              # ← 写入 subjects
        )
        total += len(buf_x)

    return total

# =========================
# 自动 5 折（分层、受试者独立）
# =========================
def _stratified_5fold_indices(labels: np.ndarray, seed: int = 42) -> List[List[int]]:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    N = len(labels)

    if n_pos >= 5 and n_neg >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [[] for _ in range(5)]
        for k, (_, test_idx) in enumerate(skf.split(np.zeros(N), labels)):
            folds[k] = sorted(test_idx.tolist())
        return folds

    # 回退（极端少样本时）
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0].tolist()
    neg_idx = np.where(labels == 0)[0].tolist()
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    folds = [[] for _ in range(5)]
    for i, idx in enumerate(pos_idx): folds[i % 5].append(idx)
    for i, idx in enumerate(neg_idx): folds[i % 5].append(idx)
    for k in range(5): folds[k] = sorted(folds[k])
    return folds
def _write_cv_fold_counts(summary: dict, out_root: str):
    """
    基于 summary["folds"]（含每折每受试者 overlap / nonoverlap 切片数）汇总：
      - 对于 val_fold=k：
          训练集 = 其它4折的 overlap（计数按标签汇总）
          验证集 = 第k折的 nonoverlap（计数按标签汇总）
    写入 out_root/cv_fold_counts.json
    """
    # fold_id -> subjects 列表（每项含: subject_id, label, n_windows_overlap, n_windows_nonoverlap）
    by_fold = {f["fold_id"]: f["subjects"] for f in summary.get("folds", [])}

    cv_summaries = []
    for k in sorted(by_fold.keys()):
        # 训练（其余折 overlap）
        train_mdd = train_hc = 0
        for j in sorted(by_fold.keys()):
            if j == k:
                continue
            for s in by_fold[j]:
                if s["label"] == "MDD":
                    train_mdd += int(s.get("n_windows_overlap", 0))
                else:
                    train_hc += int(s.get("n_windows_overlap", 0))

        # 验证（第k折 nonoverlap）
        val_mdd = val_hc = 0
        for s in by_fold[k]:
            if s["label"] == "MDD":
                val_mdd += int(s.get("n_windows_nonoverlap", 0))
            else:
                val_hc += int(s.get("n_windows_nonoverlap", 0))

        cv_summaries.append({
            "val_fold": int(k),
            "train": {
                "mode": "overlap",
                "MDD": int(train_mdd),
                "HC": int(train_hc),
                "total": int(train_mdd + train_hc),
            },
            "val": {
                "mode": "nonoverlap",
                "MDD": int(val_mdd),
                "HC": int(val_hc),
                "total": int(val_mdd + val_hc),
            }
        })

    out_path = os.path.join(out_root, "cv_fold_counts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"splits": cv_summaries}, f, ensure_ascii=False, indent=2)
    print(f"[export] 交叉验证（训练/验证）MDD/HC 切片计数已写入 {out_path}")

# =========================
# 主流程
# =========================
def export_per_fold_auto(seed: int = 42):
    processed, labels, n_mdd, n_hc = _preprocess_from_4col(_ALL_DATA_PATH, _CHANNELS)
    total_n = len(labels)
    print(f"[export] n_mdd={n_mdd}, n_hc={n_hc}, total={total_n}")
    print(f"[export] 采样率 fs={_FS}Hz，GFCC帧长={_WIN_SAMP}点({1000*_GFCC_WIN_LEN_S:.1f}ms)，"
          f"帧移={_HOP_SAMP}点({1000*_GFCC_WIN_HOP_S:.1f}ms)")
    print(f"[export] 切窗：win={_WINDOW_SIZE}点({int(_WINDOW_SEC*1000)}ms)，"
          f"stride_overlap={_STRIDE}点({int(_STRIDE_SEC*1000)}ms)")

    subject_folds = _stratified_5fold_indices(labels, seed=seed)

    os.makedirs(_OUT_ROOT, exist_ok=True)
    with open(os.path.join(_OUT_ROOT, "auto_folds.json"), "w", encoding="utf-8") as f:
        json.dump({"n_mdd": n_mdd, "n_hc": n_hc, "total": total_n,
                   "folds_1based": [[i+1 for i in fold] for fold in subject_folds]},
                  f, ensure_ascii=False, indent=2)
    print(f"[export] 五折划分已写入 {_OUT_ROOT}/auto_folds.json")

    # ========= 新增：按折汇总每个受试者（编号/类型/overlap与nonoverlap切片数） =========
    summary = {
        "meta": {
            "fs": _FS,
            "gfcc": {"num_ceps": _NUM_CEPS, "nfilts": _NFILTS, "nfft": _NFFT,
                     "win_len_s": _GFCC_WIN_LEN_S, "win_hop_s": _GFCC_WIN_HOP_S},
            "window": {"win_sec": _WINDOW_SEC, "stride_sec_overlap": _STRIDE_SEC, "win_size_samples": _WINDOW_SIZE, "stride_samples_overlap": _STRIDE},
            "shard_size": _SHARD_SIZE,
            "total_subjects": int(total_n),
            "n_mdd": int(n_mdd),
            "n_hc": int(n_hc)
        },
        "folds": []  # 每折一个对象
    }

    for fold_id in range(1, 6):
        fold_idx = np.asarray(subject_folds[fold_id - 1], dtype=np.int64)
        y_fold = labels[fold_idx]
        print(f"\n[Fold {fold_id}] subjects(1-based)={[(i+1) for i in fold_idx.tolist()]}"
              f"  (pos={int((y_fold==1).sum())}, neg={int((y_fold==0).sum())})")

        # 为本折建立受试者统计表
        # subs_stats[sid] = {"subject_id": sid, "label": "MDD"/"HC", "n_windows_overlap": 0, "n_windows_nonoverlap": 0}
        subs_stats: Dict[int, Dict[str, Any]] = {}
        for i in fold_idx.tolist():
            sid = int(i) + 1
            subs_stats[sid] = {
                "subject_id": sid,
                "label": _label_name(int(labels[i])),
                "n_windows_overlap": 0,
                "n_windows_nonoverlap": 0,
            }

        # 两种模式下分别统计数量并写分片
        for mode_name, ov, key in [("overlap", True, "n_windows_overlap"), ("nonoverlap", False, "n_windows_nonoverlap")]:
            fold_dir   = os.path.join(_OUT_ROOT, f"fold{fold_id}_{mode_name}")
            shards_dir = os.path.join(fold_dir, "shards")
            os.makedirs(shards_dir, exist_ok=True)
            print(f"  - {mode_name}: Overlap={ov}")

            # counts_map 统计本模式下每个 sid 的窗口数
            counts_map: Dict[int, int] = {sid: 0 for sid in subs_stats.keys()}
            pairs = _iter_gfcc_segments(processed, labels, fold_idx, overlap=ov)
            n = _save_npz_shards(pairs, out_dir=shards_dir, prefix="part", counts=counts_map)
            print(f"    - Saved {n} windows -> {shards_dir}")

            # 写回到 subs_stats
            for sid, cnt in counts_map.items():
                subs_stats[sid][key] = int(cnt)

        # 将本折的受试者列表写入 summary
        fold_entry = {
            "fold_id": fold_id,
            "subjects": sorted(subs_stats.values(), key=lambda d: d["subject_id"]),
            "totals": {
                "overlap": int(sum(d["n_windows_overlap"] for d in subs_stats.values())),
                "nonoverlap": int(sum(d["n_windows_nonoverlap"] for d in subs_stats.values()))
            }
        }
        summary["folds"].append(fold_entry)

    # 汇总文件（单一文件包含所有折的受试者编号/类型/切片数）
    summary_path = os.path.join(_OUT_ROOT, "fold_subject_windows.json")
    _write_cv_fold_counts(summary, _OUT_ROOT)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[export] 每折受试者统计（编号/类型/overlap与nonoverlap切片数）已写入 {summary_path}")

    print("\n[Done] 共 10 个文件夹（每折仅该折受试者）导出完成。")

# =========================
# CLI
# =========================
def main():
    global _ALL_DATA_PATH, _OUT_ROOT, _CHANNELS, _WINDOW_SEC, _STRIDE_SEC, _WINDOW_SIZE, _STRIDE

    ap = argparse.ArgumentParser(description="读取 (N,4) object npy（label+3通道），自动5折分层导出GFCC切片（fs=250）")
    ap.add_argument("--all_data", type=str, default=_ALL_DATA_PATH, help="all_data.npy 路径")
    ap.add_argument("--out_root", type=str, default=_OUT_ROOT, help="导出根目录（默认 ./slices）")
    ap.add_argument("--channels", type=str, default="1,2,3", help="1-based 通道列（逗号分隔，对应行的第1..3列）")
    ap.add_argument("--seed", type=int, default=42, help="分层随机种子")
    ap.add_argument("--win_sec", type=float, default=_WINDOW_SEC, help="切片窗口秒数（默认6.0）")
    ap.add_argument("--stride_sec", type=float, default=_STRIDE_SEC, help="overlap步长秒数（默认3.0=50%重叠）")
    args = ap.parse_args()

    _ALL_DATA_PATH = args.all_data
    _OUT_ROOT      = args.out_root
    _CHANNELS      = [int(x) for x in re.findall(r"\d+", args.channels)]
    _WINDOW_SEC    = float(args.win_sec)
    _STRIDE_SEC    = float(args.stride_sec)
    _WINDOW_SIZE   = int(_WINDOW_SEC * _FS)
    _STRIDE        = int(_STRIDE_SEC * _FS)

    export_per_fold_auto(seed=args.seed)

if __name__ == "__main__":
    main()
