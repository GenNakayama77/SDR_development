#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLUTO SDR 送信テスト - 正弦波送信
"""

import numpy as np
import adi
import time
import matplotlib.pyplot as plt
import argparse
import gc

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description='PLUTO SDR 送信テスト')
parser.add_argument('--freq', type=float, default=915e6, help='送信周波数 (Hz)')
parser.add_argument('--gain', type=int, default=0, help='送信ゲイン (dB)')
parser.add_argument('--rate', type=float, default=2e6, help='サンプリングレート (Hz)')
parser.add_argument('--duration', type=int, default=30, help='送信時間 (秒)')
args = parser.parse_args()

# PLUTOの初期化
print(f"PLUTO SDRの初期化 (周波数: {args.freq/1e6} MHz, ゲイン: {args.gain} dB)")
# 接続フォールバック: 既定 → ip:192.168.2.1 → usb
try:
    sdr = adi.Pluto()
except Exception:
    try:
        sdr = adi.Pluto(uri="ip:192.168.2.1")
    except Exception:
        sdr = adi.Pluto(uri="usb:1.2.5")
sdr.tx_lo = int(args.freq)
sdr.sample_rate = int(args.rate)
sdr.tx_rf_bandwidth = int(args.rate)
sdr.tx_hardwaregain_chan0 = args.gain

# パラメータ設定
fs = args.rate  # サンプリングレート
ts = 1/fs       # サンプリング間隔
N = 1024        # バッファサイズ

# 正弦波の生成（3つの異なる周波数の重ね合わせ）
t = np.arange(0, N*ts, ts)
signal = 0.3 * np.sin(2*np.pi*50e3*t) + 0.3 * np.sin(2*np.pi*100e3*t) + 0.3 * np.sin(2*np.pi*200e3*t)

# 複素信号に変換
iq = signal + 1j * np.zeros_like(signal)

# 信号の可視化
plt.figure(figsize=(10, 6))
plt.plot(t[:100]*1e6, iq.real[:100])
plt.title('送信信号 (最初の100サンプル)')
plt.xlabel('時間 (μs)')
plt.ylabel('振幅')
plt.grid(True)
plt.savefig('tx_signal.png')
print("信号プロットを保存しました: tx_signal.png")

# 信号送信
print(f"正弦波信号の送信を開始します (周波数成分: 50kHz, 100kHz, 200kHz)...")
print(f"送信時間: {args.duration}秒")
sdr.tx_cyclic_buffer = True
sdr.tx(iq)

# 指定時間送信を継続
try:
    for i in range(args.duration):
        print(f"送信中... {i+1}/{args.duration}秒", end="\r")
        time.sleep(1)
    print("\n送信完了")
except KeyboardInterrupt:
    print("\n送信中断")

# 送信停止とクリーンアップ
try:
    sdr.tx_destroy_buffer()
except Exception:
    pass
try:
    import matplotlib.pyplot as plt
    plt.close('all')
except Exception:
    pass
try:
    del sdr
except Exception:
    pass
time.sleep(0.2)
gc.collect()
print("PLUTO SDR送信テスト終了")