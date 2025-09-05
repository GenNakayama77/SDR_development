#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLUTO SDR FMラジオ受信テスト
"""

import numpy as np
import adi
import matplotlib.pyplot as plt
import scipy.signal as signal
import sounddevice as sd
import argparse
import time
import gc

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description='PLUTO SDR FMラジオ受信テスト')
parser.add_argument('--freq', type=float, default=90.5e6, help='受信周波数 (Hz)')
parser.add_argument('--gain', type=int, default=30, help='受信ゲイン (dB)')
parser.add_argument('--rate', type=float, default=2.4e6, help='サンプリングレート (Hz)')
parser.add_argument('--duration', type=int, default=30, help='受信時間 (秒)')
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
        sdr = adi.Pluto(uri="usb:1.2.5")  # 環境により異なるため最後の手段
sdr.rx_lo = int(args.freq)
sdr.sample_rate = int(args.rate)
sdr.rx_rf_bandwidth = int(args.rate)
sdr.rx_buffer_size = 64 * 1024
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = args.gain

# FMラジオのパラメータ
audio_rate = 48000  # オーディオサンプリングレート
deviation = 75e3    # FM偏移 (75kHz for 標準FMラジオ)
tau = 75e-6         # プリエンファシス時定数

# FMデモジュレーション関数
def fm_demod(x):
    # 微分位相の計算
    y = np.zeros(len(x), dtype=np.float32)
    y[1:] = np.angle(x[1:] * np.conj(x[:-1]))
    return y

# 受信処理
print("FMラジオ受信を開始します...")
print(f"受信周波数: {args.freq/1e6:.1f} MHz")
print(f"受信時間: {args.duration}秒")

# スペクトラム表示用の設定
plt.ion()  # インタラクティブモード有効化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
data_buffer = []

try:
    # 受信ループ
    start_time = time.time()
    
    # オーディオ出力設定
    stream = sd.OutputStream(
        samplerate=audio_rate,
        channels=1,
        dtype='float32',
        blocksize=1024
    )
    stream.start()
    
    while time.time() - start_time < args.duration:
        # サンプル受信
        samples = sdr.rx()
        
        # 短時間フーリエ変換でスペクトラム表示
        if len(data_buffer) < 5:
            data_buffer.append(samples)
        else:
            # すべてのデータを結合
            all_samples = np.concatenate(data_buffer)
            
            # スペクトラム計算
            f, t, Sxx = signal.spectrogram(all_samples, fs=args.rate, 
                                          nperseg=1024, noverlap=512)
            
            # プロット更新
            ax1.clear()
            ax1.pcolormesh(t, f/1e6, 10 * np.log10(Sxx), shading='gouraud')
            ax1.set_ylabel('周波数 (MHz)')
            ax1.set_title('受信スペクトログラム')
            
            # 波形も表示
            ax2.clear()
            ax2.plot(np.real(samples[:1000]))
            ax2.plot(np.imag(samples[:1000]))
            ax2.set_xlabel('サンプル')
            ax2.set_ylabel('振幅')
            ax2.set_title('I/Q波形')
            
            plt.pause(0.1)
            data_buffer = [samples]  # バッファをリセット
            
        # FMデモジュレーション
        demod = fm_demod(samples)
        
        # ダウンサンプリング (デシメーション)
        audio_decim = int(args.rate / audio_rate)
        audio = signal.decimate(demod, audio_decim)
        
        # デエンファシス
        b, a = signal.butter(1, 1/(2*np.pi*tau*audio_rate))
        audio = signal.lfilter(b, a, audio)
        
        # 正規化
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        # オーディオ出力
        stream.write(audio.astype(np.float32))
        
        # 進捗表示
        elapsed = time.time() - start_time
        print(f"受信中... {elapsed:.1f}/{args.duration}秒", end="\r")
        
    print("\n受信完了")
    
except KeyboardInterrupt:
    print("\n受信中断")
finally:
    plt.ioff()
    try:
        stream.stop()
        stream.close()
    except:
        pass
    # 最終結果の保存
    try:
        plt.savefig('fm_reception.png')
        print("受信スペクトログラムを保存しました: fm_reception.png")
    except:
        pass
    # 図を明示的にクローズ
    try:
        plt.close('all')
    except:
        pass
    # SDRリソースの解放
    try:
        if 'sdr' in locals():
            del sdr
    except:
        pass
    # 少し待ってからGC
    try:
        time.sleep(0.2)
    except:
        pass
    gc.collect()
    
print("PLUTO SDR FMラジオ受信テスト終了")