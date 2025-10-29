# -*- coding: utf-8 -*-
"""
InfoFlow Regime Backtest (No Look-Ahead)
- 事件识别基于过去窗口滚动 z-score，避免未来信息泄露
- K 线优先从 1m 重采样到 5min/15min/1h（right-closed），兼容旧工程路径
- 指标来自 TradingLite 三条线：clarity(i.e. Line_0), iem(Line_1), ici(Line_2)

Author: OrderFlow Project
Date: 2025-10-29
"""

from __future__ import annotations
import os
import glob
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------- 默认配置（可用 CLI 覆盖） ----------------------
BASE_DIR = r"C:\Users\69557\OrderFlow-V6"
BASE_INDICATOR_PATH = os.path.join(BASE_DIR, r"data\stats\tradinglite")
BASE_KLINE_STATS    = os.path.join(BASE_DIR, r"data\stats\binance")
BASE_KLINE_EXCH     = os.path.join(BASE_DIR, r"data\exchange")
BASE_OUTPUT_PATH    = os.path.join(BASE_DIR, r"data\stats\output_infoflow_multi_stoploss")

DEFAULT_FREQS   = ["5min", "15min", "1h"]
DEFAULT_SYMBOLS = ["BTCUSDT"]  # 用于筛选文件名（宽松）
FREQ_TO_RULE    = {"1m": "1T", "5min": "5T", "15min": "15T", "1h": "1H"}

# 事件识别滚动窗口（仅用过去数据）
ZSCORE_WINDOW     = 500      # 过去 500 样本
ZSCORE_MIN_PERIOD = 50       # 至少 50 才开始出 z-score
EVENT_COOLDOWN    = 3        # 事件去连发的冷却（单位：bar）

# 前瞻收益窗口（分钟）
FORWARD_HORIZONS_MIN = [30, 60, 120]

os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

# ---------------------- 工具函数 ----------------------
def _read_csv_any(fp: str) -> pd.DataFrame:
    """容错 CSV 读取：自动分隔符、UTF-8 BOM、跳坏行。"""
    return pd.read_csv(
        fp,
        engine="python",
        sep=None,
        encoding="utf-8-sig",
        dtype="unicode",
        on_bad_lines="skip",
    )

def _to_utc_index_from_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
    mask = ts.notna()
    df = df.loc[mask].copy()
    df.index = pd.DatetimeIndex(ts[mask])
    df.index.name = "time"
    return df.sort_index()

def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    """优先匹配旧工程常见时间列名；失败再自动探测。"""
    candidates = ["time", "Time", "datetime", "Datetime", "Date", "date", "ts", "open_time"]
    for c in candidates:
        if c in df.columns:
            # 验证是否可解析
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            if ts.notna().sum() >= 3:
                return c
    # 自动探测：非空最多的可解析列
    best, best_n = None, -1
    for c in df.columns:
        ts = pd.to_datetime(df[c], errors="coerce", utc=True)
        non_na = ts.notna().sum()
        if non_na > best_n:
            best, best_n = c, non_na
    if best is not None and best_n >= 3:
        return best
    return None

def _safe_read_tradinglite_single(fp: str, value_col_name: str) -> pd.DataFrame:
    """
    读取 TradingLite 单线 CSV：
      - 顶部可混说明/标题行
      - 自动识别时间列并设为 UTC 索引
      - 数值列：自动从剩余列中选择“最像数值”的一列
    """
    df = _read_csv_any(fp)
    if df.empty:
        raise ValueError(f"文件为空：{fp}")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty:
        raise ValueError(f"文件内容全空：{fp}")

    tcol = _pick_time_col(df)
    if tcol is None:
        raise ValueError(f"未能识别时间列：{fp}")
    df = _to_utc_index_from_col(df, tcol)

    # 选择数值列
    non_value_cols = {tcol}
    candidates = [c for c in df.columns if c not in non_value_cols]
    picked_name, picked_series, picked_score = None, None, -1.0
    for c in candidates:
        series = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        num = pd.to_numeric(series, errors="coerce")
        valid = int(num.notna().sum())
        if valid == 0:
            continue
        score = valid + (float(np.nanvar(num.values)) if valid > 1 else 0.0)
        if score > picked_score:
            picked_name, picked_series, picked_score = c, num, score

    if picked_series is None:
        raise ValueError(f"未找到可用数值列：{fp}")

    out = pd.DataFrame({value_col_name: picked_series.values}, index=df.index)
    out = out[~out.index.duplicated(keep="first")].dropna(subset=[value_col_name])
    return out.sort_index()

def _search_kline_files() -> List[str]:
    """
    搜索可能的 OHLCV 文件（csv/parquet）：
      - data\stats\binance\{1m,5min,15min,1h}\
      - data\exchange\**\
    """
    roots = [
        os.path.join(BASE_KLINE_STATS, "1m"),
        os.path.join(BASE_KLINE_STATS, "5min"),
        os.path.join(BASE_KLINE_STATS, "15min"),
        os.path.join(BASE_KLINE_STATS, "1h"),
        BASE_KLINE_EXCH,
    ]
    cands: List[str] = []
    for root in roots:
        if os.path.isdir(root):
            cands += glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
            cands += glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)

    # 轻度排序：优先 1m
    def _is_1m(p: str) -> int:
        s = p.lower()
        return 0 if ("1m" in os.path.basename(s) or "1m" in os.path.dirname(s)) else 1

    cands.sort(key=lambda p: (_is_1m(p), p))
    return cands

def _read_kline_file(fp: str) -> Optional[pd.DataFrame]:
    """读取单个 K 线文件并统一列名；至少包含 close 和可解析的时间列。"""
    try:
        if fp.lower().endswith(".parquet"):
            k = pd.read_parquet(fp)
        else:
            k = _read_csv_any(fp)
        if k is None or len(k) == 0:
            return None

        # 时间列
        tcol = None
        for c in ["open_time", "timestamp", "time", "datetime", "Date", "date", "ts"]:
            if c in k.columns:
                tcol = c
                break
        if tcol is None:
            # 退路：可能索引是时间
            if k.index.name:
                k = k.reset_index()
                tcol = k.columns[0]
            else:
                return None

        k[tcol] = pd.to_datetime(k[tcol], errors="coerce", utc=True)
        k = k.dropna(subset=[tcol]).set_index(tcol).sort_index()
        if k.empty:
            return None

        # 价格列
        close_col = None
        for c in ["close", "Close", "c", "closing_price"]:
            if c in k.columns:
                close_col = c
                break
        if close_col is None:
            return None

        # 保留常见列
        keep = [close_col]
        for c in ["open", "high", "low", "volume", "o", "h", "l", "v"]:
            if c in k.columns and c not in keep:
                keep.append(c)

        df = k[keep].copy()
        df.columns = [c.lower() for c in df.columns]
        # 索引确保 UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception:
        return None

def _resample_from_1m(df_1m: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """从 1m 重采样到目标频率（right-closed），不引入未来信息。"""
    rule = FREQ_TO_RULE.get(target_freq)
    if rule is None:
        return df_1m

    out = pd.DataFrame(index=df_1m.index)

    # close（必要）
    close_c = None
    for c in df_1m.columns:
        if c.lower() in ("close", "c", "closing_price"):
            close_c = c
            break
    if close_c is None:
        raise ValueError("1m 数据缺少 close 列，无法重采样。")

    # right-closed: 按已结束的目标周期聚合
    out["close"] = df_1m[close_c].resample(rule, label="right", closed="right").last()

    # 可选：open/high/low/volume
    def pick(cols: List[str]) -> Optional[str]:
        low = [c.lower() for c in df_1m.columns]
        for name in cols:
            if name in low:
                return df_1m.columns[low.index(name)]
        return None

    oc = pick(["open", "o"])
    hc = pick(["high", "h"])
    lc = pick(["low", "l"])
    vc = pick(["volume", "vol", "v"])

    if oc:
        out["open"] = df_1m[oc].resample(rule, label="right", closed="right").first()
    if hc:
        out["high"] = df_1m[hc].resample(rule, label="right", closed="right").max()
    if lc:
        out["low"] = df_1m[lc].resample(rule, label="right", closed="right").min()
    if vc:
        out["volume"] = df_1m[vc].resample(rule, label="right", closed="right").sum()

    return out.dropna(subset=["close"])

def _load_kline_freq(freq: str, symbols: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    读取目标频率 K 线：
      1) 若存在目标频率文件，优先直接读取；
      2) 否则寻找 1m 文件并重采样到目标频率；
      3) 符号过滤尽量放宽（文件名不含符号也允许）。
    """
    files = _search_kline_files()
    if not files:
        return pd.DataFrame(), ""

    def match_symbol(path: str) -> bool:
        name = (os.path.basename(path) + " | " + os.path.dirname(path)).lower()
        return any(sym.lower() in name for sym in symbols)

    direct: Optional[Tuple[pd.DataFrame, str]] = None
    base1m: Optional[Tuple[pd.DataFrame, str]] = None

    for fp in files:
        base = os.path.basename(fp).lower()
        folder = os.path.dirname(fp).lower()
        in_freq = (freq in base) or (freq in folder)

        # 1) 直接命中目标频率
        if in_freq:
            df = _read_kline_file(fp)
            if df is not None and not df.empty:
                if symbols and match_symbol(fp) is False:
                    # 文件名不含符号就放行；含符号但不匹配则跳过
                    pass
                direct = (df, fp)
                break

        # 2) 记下 1m 备选
        is_1m = ("1m" in base) or ("1m" in folder)
        if is_1m and base1m is None:
            df = _read_kline_file(fp)
            if df is not None and not df.empty:
                if symbols and (match_symbol(fp) is False):
                    pass
                base1m = (df, fp)

    if direct:
        return direct

    if base1m:
        df1m, used = base1m
        try:
            res = _resample_from_1m(df1m, freq)
            if not res.empty:
                return res, used + f" (resampled -> {freq})"
        except Exception:
            pass

    return pd.DataFrame(), ""

# ---------------------- 指标读取与事件识别（无未来） ----------------------
def load_indicator_data(freq: str) -> pd.DataFrame:
    """
    加载 TradingLite 三线（Line_0/1/2）并按索引对齐：
      - Line_0 -> clarity
      - Line_1 -> iem
      - Line_2 -> ici
    """
    fdir = os.path.join(BASE_INDICATOR_PATH, freq)
    file_0 = glob.glob(os.path.join(fdir, "*_Line_0_*.csv"))
    file_1 = glob.glob(os.path.join(fdir, "*_Line_1_*.csv"))
    file_2 = glob.glob(os.path.join(fdir, "*_Line_2_*.csv"))
    if not (file_0 and file_1 and file_2):
        print(f"[WARN] {freq} 缺少 Line_0/1/2 CSV，路径：{fdir}")
        return pd.DataFrame()

    print(f"[INFO] {freq} 指标文件：\n  0: {file_0[0]}\n  1: {file_1[0]}\n  2: {file_2[0]}")
    df0 = _safe_read_tradinglite_single(file_0[0], "clarity")
    df1 = _safe_read_tradinglite_single(file_1[0], "iem")
    df2 = _safe_read_tradinglite_single(file_2[0], "ici")

    df = pd.concat([df0, df1, df2], axis=1)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _rolling_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    过去窗口滚动 z-score（不居中、不使用未来）：
      z_t = (x_t - mean_{t-window..t}) / std_{t-window..t}
    """
    x = pd.to_numeric(x, errors="coerce")
    roll = x.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std(ddof=1)
    z = (x - mu) / (sd + 1e-9)
    return z

def detect_regime_events(ind: pd.DataFrame) -> pd.DataFrame:
    """
    五态定义（基于过去滚动 z）：
      Dissipation:             ici_z < -1.0                         （过滤）
      Overload:                clarity_z < -1.0 & iem_z >  1.0 & ici_z < -0.5   （做空）
      Release:                 clarity_z >  0.0 & iem_z >  1.0 & ici_z >  1.0   （做多）
      Reversion(Fake_Decay):   clarity_z >  0.5 & iem_z < -1.0 & ici_z >  0.0   （做多）
      Latent:                  clarity_z >  1.0 & iem_z < -0.5 & -0.5<ici_z<0.5 （观望/持仓）
    去连发：EVENT_COOLDOWN bars 内不重复记同类事件。
    """
    if ind.empty:
        return ind

    df = ind.copy()
    df["clarity_z"] = _rolling_zscore(df["clarity"], ZSCORE_WINDOW, ZSCORE_MIN_PERIOD)
    df["iem_z"]     = _rolling_zscore(df["iem"],     ZSCORE_WINDOW, ZSCORE_MIN_PERIOD)
    df["ici_z"]     = _rolling_zscore(df["ici"],     ZSCORE_WINDOW, ZSCORE_MIN_PERIOD)

    def classify_row(row) -> str:
        c, i, ic = row["clarity_z"], row["iem_z"], row["ici_z"]
        if pd.isna(c) or pd.isna(i) or pd.isna(ic):
            return "None"
        if ic < -1.0: return "Dissipation"
        if (c < -1.0) and (i > 1.0) and (ic < -0.5): return "Overload"
        if (c >  0.0) and (i > 1.0) and (ic >  1.0): return "Release"
        if (c >  0.5) and (i < -1.0) and (ic >  0.0): return "Reversion"
        if (c >  1.0) and (i < -0.5) and (-0.5 < ic < 0.5): return "Latent"
        return "None"

    events: List[Tuple[pd.Timestamp, str]] = []
    last_time_by_type: Dict[str, pd.Timestamp] = {}
    prev_label = "None"

    for ts, row in df.iterrows():
        label = classify_row(row)
        if label == "None":
            prev_label = label
            continue

        # 仅在状态变化时触发
        if label != prev_label:
            # 冷却检查
            last_ts = last_time_by_type.get(label)
            if (last_ts is None) or (ts - last_ts >= pd.Timedelta(minutes=1) * EVENT_COOLDOWN):
                events.append((ts, label))
                last_time_by_type[label] = ts
        prev_label = label

    if not events:
        return pd.DataFrame(columns=["etype"], index=pd.DatetimeIndex([], tz="UTC"))

    ev = pd.DataFrame(events, columns=["time", "etype"]).set_index("time")
    ev.index = ev.index.tz_convert("UTC") if ev.index.tz is not None else ev.index.tz_localize("UTC")
    return ev.sort_index()

# ---------------------- 收益计算（无未来） ----------------------
def attach_forward_returns(
    events: pd.DataFrame,
    price_close: pd.Series,
    horizons_min: List[int],
) -> pd.DataFrame:
    """
    对每个事件，计算多重前瞻收益：
      - p0：事件时刻（ts）当根 K 线已知的“最近已收盘价”（<= ts 的最后一个 close）
      - pH：事件发生后 H 分钟时刻（ts + Hm）已知的“最近已收盘价”（<= ts+H 的最后一个 close）
      - 收益 = sign * (pH / p0 - 1)
      - 不做任何未来插值；若 pH 不存在则记 NaN
    """
    if events.empty or price_close.empty:
        return pd.DataFrame()

    px = price_close.copy().sort_index()
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")

    out_rows: List[Dict] = []

    # 帮助函数：取目标时刻的“最近已收盘价”（<= t）
    def last_close_le(t: pd.Timestamp) -> float:
        pos = px.index.searchsorted(t)
        if pos == 0:
            return np.nan
        return float(px.iloc[pos - 1])

    for ts, row in events.iterrows():
        et = row["etype"]

        p0 = last_close_le(ts)
        if not np.isfinite(p0):
            # 事件早于价格区间
            continue

        if et in ("Release", "Reversion"):
            sign = +1
        elif et == "Overload":
            sign = -1
        else:
            sign = 0  # Dissipation / Latent：只统计不交易

        rec = {"time": ts, "etype": et, "price_t0": p0, "direction": sign}
        for h in horizons_min:
            tgt = ts + pd.Timedelta(minutes=int(h))
            pH = last_close_le(tgt)
            if np.isfinite(pH) and sign != 0:
                rec[f"ret_{h}m"] = (pH / p0 - 1.0) * sign
            elif np.isfinite(pH):
                rec[f"ret_{h}m"] = (pH / p0 - 1.0)  # 纯观察型
            else:
                rec[f"ret_{h}m"] = np.nan
        out_rows.append(rec)

    if not out_rows:
        return pd.DataFrame()

    ret = pd.DataFrame(out_rows).set_index("time").sort_index()
    ret.index = ret.index.tz_convert("UTC") if ret.index.tz is not None else ret.index.tz_localize("UTC")
    return ret

def summarize_results(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for et in df["etype"].unique():
        sub = df[df["etype"] == et]
        row = {"etype": et, "n": int(len(sub))}
        for h in horizons:
            col = f"ret_{h}m"
            x = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(x) >= 2:
                row[f"mean_{h}m"] = float(x.mean())
                row[f"std_{h}m"]  = float(x.std(ddof=1))
                row[f"win_{h}m"]  = float((x > 0).mean())
                row[f"t_{h}m"]    = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
            elif len(x) == 1:
                row[f"mean_{h}m"] = float(x.iloc[0])
                row[f"std_{h}m"]  = np.nan
                row[f"win_{h}m"]  = float(1.0 if x.iloc[0] > 0 else 0.0)
                row[f"t_{h}m"]    = np.nan
            else:
                row[f"mean_{h}m"] = np.nan
                row[f"std_{h}m"]  = np.nan
                row[f"win_{h}m"]  = np.nan
                row[f"t_{h}m"]    = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------- 主流程 ----------------------
def run_once(freq: str, symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ind = load_indicator_data(freq)
    if ind.empty:
        print(f"[SKIP] {freq} 指标缺失。")
        return pd.DataFrame(), pd.DataFrame()

    events = detect_regime_events(ind)
    if events.empty:
        print(f"[INFO] {freq} 未识别到事件。")
        return pd.DataFrame(), pd.DataFrame()

    kline, used_fp = _load_kline_freq(freq, symbols)
    if kline.empty:
        print(f"[WARN] 未找到 {freq} K 线；收益将无法计算。")
        price = pd.Series(dtype=float)
    else:
        # 对齐交集时间（避免 price 在事件开始前/后过多空段）
        common_start = max(events.index.min(), kline.index.min())
        common_end   = min(events.index.max(), kline.index.max())
        if common_end > common_start:
            events = events.loc[common_start:common_end]
            price  = kline["close"].loc[common_start:common_end]
        else:
            price  = kline["close"]

    # 收益
    returns = attach_forward_returns(events, price, FORWARD_HORIZONS_MIN) if not price.empty else pd.DataFrame()

    # 导出
    out_dir = os.path.join(BASE_OUTPUT_PATH, freq)
    os.makedirs(out_dir, exist_ok=True)
    events.to_csv(os.path.join(out_dir, f"events_{freq}.csv"))
    if not returns.empty:
        returns.to_csv(os.path.join(out_dir, f"returns_{freq}.csv"))
        summ = summarize_results(returns, FORWARD_HORIZONS_MIN)
        summ.to_csv(os.path.join(out_dir, f"summary_{freq}.csv"), index=False)
    else:
        summ = pd.DataFrame()

    with open(os.path.join(out_dir, "_README.txt"), "w", encoding="utf-8") as f:
        f.write(f"InfoFlow Regime 回测产物（{freq})\n")
        f.write(f"- 指标目录：{os.path.join(BASE_INDICATOR_PATH, freq)}\n")
        f.write(f"- K线来源：{used_fp or '未找到'}\n")
        f.write(f"- 事件数：{len(events)}\n")
        f.write(f"- 结果文件：events_{freq}.csv / returns_{freq}.csv / summary_{freq}.csv\n")

    return returns, summ

def main():
    parser = argparse.ArgumentParser(description="InfoFlow Regime Backtest (No Look-Ahead)")
    parser.add_argument("--freqs", type=str, default=",".join(DEFAULT_FREQS),
                        help="逗号分隔，如 5min,15min,1h")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                        help="用于文件名/路径的宽松筛选（可留空表示不筛）")
    args = parser.parse_args()

    freqs   = [s.strip() for s in args.freqs.split(",") if s.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    all_summaries = []
    for freq in freqs:
        print(f"\n===== 运行 {freq} 回测 =====")
        try:
            _, summ = run_once(freq, symbols)
            if summ is not None and not summ.empty:
                summ.insert(0, "freq", freq)
                all_summaries.append(summ)
        except Exception as e:
            print(f"[ERROR] {freq} 回测异常：{e}")

    if all_summaries:
        final = pd.concat(all_summaries, ignore_index=True)
        final.to_csv(os.path.join(BASE_OUTPUT_PATH, "infoflow_summary_regime_v7_ALL.csv"), index=False)
        print("\n--- 总汇总 ---")
        try:
            print(final.to_markdown(index=False, floatfmt=".6f"))
        except Exception:
            print(final)
    else:
        print("\n[INFO] 没有可汇总的数据。")

if __name__ == "__main__":
    main()
