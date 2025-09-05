#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPSK変調/復調のテスト
1Mbps, BPSK変調
"""

import numpy as np
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("iq_test")

class BPSKModulator:
    """BPSK変調器クラス"""
    
    def __init__(self, sample_rate=10e6, symbol_rate=1e6):
        """
        初期化
        Args:
            sample_rate (float): サンプリングレート [Hz]
            symbol_rate (float): シンボルレート [bps]
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        logger.info(f"BPSKModulator initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Symbol rate: {symbol_rate} bps")
        logger.info(f"  Samples per symbol: {self.samples_per_symbol}")
    
    def modulate(self, data):
        """
        BPSK変調を実行
        Args:
            data (bytearray): 入力データ
        Returns:
            tuple: (I信号, Q信号)
        """
        try:
            # バイト列をビット列に変換（MSB first）
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            logger.info(f"Input data: {len(data)} bytes, {len(bits)} bits")
            
            # デバッグ: 最初の数ビットを表示
            logger.info(f"First 16 bits: {bits[:16]}")
            
            # 各ビットをシンボルに変換（0→-1, 1→1）
            symbols = 2 * bits.astype(np.float32) - 1
            
            # デバッグ: 最初の数シンボルを表示
            logger.info(f"First 16 symbols: {symbols[:16]}")
            
            # シンボルをサンプルに変換（シンボルレートからサンプリングレートに）
            i_samples = np.repeat(symbols, self.samples_per_symbol)
            q_samples = np.zeros_like(i_samples)  # BPSKではQ成分は0
            
            logger.info(f"BPSK modulation completed: {len(data)} bytes -> {len(i_samples)} samples")
            return i_samples, q_samples
            
        except Exception as e:
            logger.error(f"BPSK modulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

class BPSKDemodulator:
    """BPSK復調器クラス"""
    
    def __init__(self, sample_rate=10e6, symbol_rate=1e6):
        """
        初期化
        Args:
            sample_rate (float): サンプリングレート [Hz]
            symbol_rate (float): シンボルレート [bps]
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        logger.info(f"BPSKDemodulator initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Symbol rate: {symbol_rate} bps")
        logger.info(f"  Samples per symbol: {self.samples_per_symbol}")
    
    def demodulate(self, i_samples, q_samples):
        """
        BPSK復調を実行
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
        Returns:
            bytearray: 復調されたデータ
        """
        try:
            # シンボル判定（I信号の符号で判定）
            # シンボル中央のサンプルを使用
            symbol_indices = np.arange(self.samples_per_symbol//2, len(i_samples), self.samples_per_symbol)
            symbol_samples = i_samples[symbol_indices]
            
            # デバッグ: シンボルサンプルの最初の数値を表示
            logger.info(f"First 16 symbol samples: {symbol_samples[:16]}")
            
            # シンボル判定: 0未満なら-1、それ以外なら1
            symbols = np.where(symbol_samples < 0, -1, 1)
            
            # デバッグ: 判定後のシンボルの最初の数値を表示
            logger.info(f"First 16 decided symbols: {symbols[:16]}")
            
            # シンボルをビットに変換（-1→0, 1→1）
            # ここが重要なポイント - 整数型に変換することを確実に
            bits = np.zeros_like(symbols, dtype=np.uint8)
            bits[symbols > 0] = 1  # 1のシンボルに対応するビットを1に設定
            
            # デバッグ: 変換後のビットの最初の数値を表示
            logger.info(f"First 16 bits after conversion: {bits[:16]}")
            
            # ビット列をバイト列に変換（MSB first）
            # packbitsはuint8型配列を期待する
            bytes_needed = (len(bits) + 7) // 8  # 必要なバイト数（切り上げ）
            bit_padding = bytes_needed * 8 - len(bits)  # 足りないビット数
            
            if bit_padding > 0:
                # 8の倍数になるようにパディング
                padded_bits = np.concatenate([bits, np.zeros(bit_padding, dtype=np.uint8)])
                logger.info(f"Added {bit_padding} padding bits for byte alignment")
            else:
                padded_bits = bits
                
            # パッキングしてバイト列に変換
            decoded_data = np.packbits(padded_bits).tobytes()
            
            logger.info(f"BPSK demodulation completed: {len(i_samples)} samples -> {len(decoded_data)} bytes")
            return decoded_data
            
        except Exception as e:
            logger.error(f"BPSK demodulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def plot_signals(i_samples, q_samples, title, max_samples=1000):
    """信号をプロット"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(i_samples[:max_samples], label='I')
        plt.plot(q_samples[:max_samples], label='Q')
        plt.title(title)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error plotting signals: {e}")
        return None

def compare_data(original, processed):
    """データの比較"""
    try:
        # バイト長の確認
        if len(original) != len(processed):
            logger.error(f"Data size mismatch: original={len(original)} bytes, processed={len(processed)} bytes")
            
            # サイズを合わせる
            min_len = min(len(original), len(processed))
            original = original[:min_len]
            processed = processed[:min_len]
            logger.warning(f"Comparing first {min_len} bytes for analysis")
        
        # バイト単位での比較
        match_count = sum(1 for a, b in zip(original, processed) if a == b)
        match_percentage = (match_count / len(original)) * 100
        
        # ビット単位での詳細分析（最初の数バイトのみ）
        for i in range(min(8, len(original))):
            orig_byte = original[i]
            proc_byte = processed[i]
            orig_bits = format(orig_byte, '08b')
            proc_bits = format(proc_byte, '08b')
            logger.info(f"Byte {i}: Original: {orig_bits} ({orig_byte:02x}), Processed: {proc_bits} ({proc_byte:02x})")
        
        if match_percentage < 100:
            logger.error(f"Data mismatch: {match_percentage:.2f}% match")
            # 不一致位置を表示
            mismatch_positions = [(i, a, b) for i, (a, b) in enumerate(zip(original, processed)) if a != b]
            logger.error(f"Total mismatches: {len(mismatch_positions)}")
            # 最初の10個の不一致位置を表示
            for pos, orig, proc in mismatch_positions[:10]:
                logger.error(f"Position {pos}: original={orig:02x}, processed={proc:02x}")
            return False
        
        logger.info(f"Data match: 100% ({match_count}/{len(original)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Error comparing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン処理"""
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # 変調器の初期化
        modulator = BPSKModulator(sample_rate=10e6, symbol_rate=1e6)
        
        # 復調器の初期化
        demodulator = BPSKDemodulator(sample_rate=10e6, symbol_rate=1e6)
        
        # テストデータの生成（1115バイト）
        test_data = bytearray([0x55] * 1115)  # パターンデータ
        
        # テストデータの保存
        test_data_path = os.path.join(timestamp_dir, "test_data.bin")
        with open(test_data_path, 'wb') as f:
            f.write(test_data)
        logger.info(f"Saved test data to {test_data_path}")
        
        # BPSK変調
        i_samples, q_samples = modulator.modulate(test_data)
        if i_samples is None or q_samples is None:
            logger.error("BPSK modulation failed")
            return 1
        
        # 変調信号のプロット
        fig = plot_signals(i_samples, q_samples, "BPSK Modulated Signal")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "modulated_signal.png")
            fig.savefig(plot_path)
            logger.info(f"Saved plot to {plot_path}")
        
        # 変調信号の保存
        signal_path = os.path.join(timestamp_dir, "modulated_signal.npz")
        np.savez(signal_path, i_samples=i_samples, q_samples=q_samples)
        logger.info(f"Saved modulated signal to {signal_path}")
        
        # BPSK復調
        decoded_data = demodulator.demodulate(i_samples, q_samples)
        if decoded_data is None:
            logger.error("BPSK demodulation failed")
            return 1
        
        # 復調データの保存
        decoded_path = os.path.join(timestamp_dir, "decoded_data.bin")
        with open(decoded_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved decoded data to {decoded_path}")
        
        # データの比較
        if not compare_data(test_data, decoded_data):
            logger.error("Data comparison failed")
            return 1
        
        logger.info("\nTest completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())