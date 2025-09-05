#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlutoSDR送信テスト - 1Mbps本番仕様
CCSDS準拠のフレームを送信します。
"""

import numpy as np
import time
import sys
import reedsolo
import logging
import struct
import os
import signal
import argparse
from datetime import datetime

# SDRのインポート確認とエラーハンドリング
try:
    import adi
except ImportError:
    print("Error: pyadi-iio library not found.")
    print("Please install it with: pip install pyadi-iio")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("origamisat2_tx")

# テスト設定
MAX_TEST_ITERATIONS = 5  # テスト回数
test_instance = None  # グローバル変数としてテストインスタンスを保持

# シグナルハンドラーの定義
def signal_handler(sig, frame):
    global test_instance
    print('\nプログラムを終了します...')
    if test_instance is not None:
        test_instance.stop_test()
    sys.exit(0)

# SIGINTシグナル（Ctrl+C）のハンドラーを設定
signal.signal(signal.SIGINT, signal_handler)

class CCSDSFrameGenerator:
    """CCSDS準拠のデータフレームを生成するクラス"""
    
    def __init__(self):
        # リードソロモンコーデック (RS(255,223)、インターリーブ5)
        self.rs_n = 255  # RSコードワード長
        self.rs_k = 223  # メッセージ長
        self.rs_codec = reedsolo.RSCodec(self.rs_n - self.rs_k)
        self.interleave_depth = 5  # インターリーブ深さ
        
        # CCSDSフレームパラメータ（仕様書より）
        self.vcdu_size = 1115  # VCDUデータサイズ
        self.sync_marker = bytes.fromhex("1ACFFC1D")  # 同期マーカー (0x1ACFFC1D)
        self.rs_ecc_size = 160  # リードソロモン誤り訂正符号サイズ
        self.cadu_size = self.vcdu_size + len(self.sync_marker) + self.rs_ecc_size
        
        # VCDUヘッダーパラメータ
        self.version = 0  # バージョン (2ビット)
        self.scid = 0     # 宇宙機ID (8ビット)
        self.vcid = 0x01  # 仮想チャネルID (6ビット)
        self.replay_flag = 0  # リプレイフラグ (1ビット)
        self.vcdu_counter = 0  # VCDUカウンター (24ビット)
        
        logger.info(f"CCSDS FrameGenerator initialized. VCDU size: {self.vcdu_size}, CADU size: {self.cadu_size}")
    
    def generate_vcdu_primary_header(self):
        """VCDU Primary Headerを生成"""
        # バージョン(2bit) + SCID(8bit) + VCID(6bit)の最初の2バイト
        byte1 = (self.version << 6) | ((self.scid & 0xFC) >> 2)
        byte2 = ((self.scid & 0x03) << 6) | (self.vcid & 0x3F)
        
        # VCDUカウンター (24ビット = 3バイト)
        counter_bytes = self.vcdu_counter.to_bytes(3, byteorder='big')
        
        # リプレイフラグ(1bit) + スペア(7bit)
        byte6 = (self.replay_flag << 7) | 0x00  # スペアは0
        
        # MPDUヘッダー (16bit)
        mpdu_header = (0x7FE).to_bytes(2, byteorder='big')  # Fill用のポインタ値 (0x7FE)
        
        # ヘッダーを結合
        header = bytes([byte1, byte2]) + counter_bytes + bytes([byte6]) + bytes(7) + mpdu_header
        
        # カウンターを増加
        self.vcdu_counter = (self.vcdu_counter + 1) & 0xFFFFFF
        
        return header
    
    def generate_test_data(self):
        """テスト用のFillデータを生成 (仕様書6ページのフォーマットに基づく)"""
        # 基本的なA/Dデータ部分(バイト13-28)
        ad_data = bytes([0x00, 0x00] * 8)  # 8チャンネル分のA/Dデータ（未使用）
        
        # 予備領域(バイト29-32)
        spare = bytes([0x00, 0x00, 0x00, 0x00])
        
        # 送信機の設定状態(バイト33-48)
        tx_settings = bytes([
            0x14,  # UartHK出力設定
            0x00,  # 0x00
            0x00,  # データ選択
            0x05,  # RS及びCONV状態（インターリーブL=5）
            0x81,  # convolutional encoder mode
            0x30,  # RF ATT設定値
            0x50,  # Baseband GAIN ADJ
            0x1E,  # シンボルレート
            0x01,  # 変調方式 (1=BPSK)
            0x00,  # 動作モード
            0x00,  # IF normal/invert
            0x00,  # Baseband filter
            0x01, 0x01, 0x01, 0x01  # 予備
        ])
        
        # 同期マーカーをいくつか埋め込む（より検出しやすくするため）
        marker_data = self.sync_marker * 5
        
        # 残りのデータを0x5Aで埋める
        remaining_size = self.vcdu_size - len(ad_data) - len(spare) - len(tx_settings) - len(marker_data) - 12
        remaining = bytes([0x5A] * remaining_size)
        
        # 12はVCDUヘッダー長
        return ad_data + spare + tx_settings + marker_data + remaining
    
    def apply_reed_solomon(self, data):
        """リードソロモン符号化を適用（完全実装）"""
        # データをインターリーブブロックに分割
        block_size = self.rs_k * self.interleave_depth
        blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
        
        encoded_blocks = []
        for block in blocks:
            # 不足分を0で埋める
            if len(block) < block_size:
                block = block + bytes([0] * (block_size - len(block)))
            
            # インターリーブ分割
            sub_blocks = [block[i:i+self.rs_k] for i in range(0, len(block), self.rs_k)]
            
            # 各サブブロックを符号化
            encoded_sub_blocks = []
            for sub_block in sub_blocks:
                # 不足分を0で埋める
                if len(sub_block) < self.rs_k:
                    sub_block = sub_block + bytes([0] * (self.rs_k - len(sub_block)))
                
                # RS符号化
                encoded = self.rs_codec.encode(sub_block)
                encoded_sub_blocks.append(encoded)
            
            # インターリーブを元に戻す
            transposed = []
            for i in range(self.rs_n):
                for j in range(len(encoded_sub_blocks)):
                    if i < len(encoded_sub_blocks[j]):
                        transposed.append(encoded_sub_blocks[j][i])
            
            encoded_blocks.extend(transposed)
        
        # 必要なECCサイズ分だけを返す
        return bytes(encoded_blocks[:self.rs_ecc_size])

    def apply_convolutional_encoding(self, data):
        """畳み込み符号化を適用（レート1/2、拘束長7）"""
        # 生成多項式: G1 = 1 + x + x^2 + x^3 + x^6, G2 = 1 + x^2 + x^3 + x^5 + x^6
        g1 = [1, 1, 1, 1, 0, 0, 1]  # 1 + x + x^2 + x^3 + x^6
        g2 = [1, 0, 1, 1, 0, 1, 1]  # 1 + x^2 + x^3 + x^5 + x^6
        
        # シフトレジスタの初期化（拘束長7に合わせて7要素に）
        shift_reg = [0] * 7
        
        # 出力バッファ
        encoded = []
        
        # 各ビットに対して畳み込み符号化を適用
        for bit in data:
            # シフトレジスタを更新
            shift_reg.insert(0, bit)
            shift_reg.pop()
            
            # 生成多項式1の出力
            out1 = sum(g1[i] * shift_reg[i] for i in range(7)) % 2
            # 生成多項式2の出力
            out2 = sum(g2[i] * shift_reg[i] for i in range(7)) % 2
            
            # 出力を追加
            encoded.extend([out1, out2])
        
        return encoded
    
    def generate_cadu_frame(self, counter):
        """CADUフレームを生成（完全実装）"""
        # 同期マーカー
        frame = bytearray(self.sync_marker)
        
        # VCDUヘッダーを生成
        header = self.generate_vcdu_primary_header()
        
        # テストデータを生成
        payload = self.generate_test_data()
        
        # VCDUを生成 (ヘッダー + ペイロード)
        vcdu = header + payload
        
        # リードソロモン符号を追加
        rs_ecc = self.apply_reed_solomon(vcdu)
        
        # 畳み込み符号化を適用
        vcdu_with_rs = vcdu + rs_ecc
        vcdu_bits = self.bytes_to_bits(vcdu_with_rs)
        conv_encoded = self.apply_convolutional_encoding(vcdu_bits)
        
        # 結合
        frame.extend(self.bits_to_bytes(conv_encoded))
        
        logger.info(f"Generated CADU frame, {len(frame)} bytes, Counter: {counter}")
        return frame

    def bytes_to_bits(self, data):
        """バイト列をビット列に変換"""
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def bits_to_bytes(self, bits):
        """ビット列をバイト列に変換"""
        return np.packbits(bits).tobytes()

class BPSKModulator:
    """BPSK変調クラス"""
    
    def __init__(self, sample_rate, symbol_rate):
        """BPSK変調器の初期化"""
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.sps = int(sample_rate / symbol_rate)
        
        logger.info(f"BPSK Modulator initialized. Sample rate: {sample_rate/1e6} MHz, Symbol rate: {symbol_rate/1e6} MHz")
        logger.info(f"Samples per symbol: {self.sps}")
    
    def bytes_to_bits(self, data_bytes):
        """バイトデータをビット配列に変換"""
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        return bits
    
    def modulate(self, data):
        """バイトデータをBPSK変調してIQサンプルに変換"""
        # バイトからビットに変換
        bits = self.bytes_to_bits(data)
        
        # サンプル整形（sps倍に伸ばす）
        symbols = 2 * bits.astype(float) - 1
        samples = np.repeat(symbols, self.sps)
        
        # 複素サンプルに変換
        iq_samples = samples + 0j
        
        logger.info(f"Modulated {len(data)} bytes to {len(iq_samples)} IQ samples")
        return iq_samples
    
class Transmitter:
    """PlutoSDRを使用した送信クラス"""
    
    def __init__(self, args):
        """送信機の初期化"""
        # 出力ディレクトリの作成
        self.timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.timestamp_dir, exist_ok=True)
        logger.info(f"データファイルの保存先ディレクトリを作成しました: {self.timestamp_dir}")
        
        # 引数の解析
        self.freq = args.freq
        self.sample_rate = args.sample_rate
        self.bit_rate = args.bit_rate
        self.tx_gain = args.tx_gain
        self.device_uri = args.device_uri
        
        # CCSDSフレーム生成オブジェクト
        self.frame_gen = CCSDSFrameGenerator()
        
        # BPSK変調オブジェクト
        self.modulator = BPSKModulator(
            self.sample_rate,
            self.bit_rate
        )
        
        # SDRの初期化
        logger.info(f"Initializing PlutoSDR with URI: {self.device_uri}")
        try:
            self.sdr = adi.Pluto(uri=self.device_uri)
            
            # TX設定
            self.sdr.tx_rf_bandwidth = int(self.sample_rate * 0.8)  # 帯域幅（サンプルレートの80%）
            self.sdr.sample_rate = int(self.sample_rate)  # サンプリングレート
            self.sdr.tx_lo = int(self.freq)  # 中心周波数
            self.sdr.tx_hardwaregain = int(self.tx_gain)  # 送信ゲイン
            
        except Exception as e:
            logger.error(f"PlutoSDRの初期化に失敗しました: {e}")
            raise
        
        logger.info(f"Transmitter configuration:")
        logger.info(f"  Frequency: {self.freq/1e6} MHz")
        logger.info(f"  Sample rate: {self.sample_rate/1e6} MSps")
        logger.info(f"  Bit rate: {self.bit_rate/1e3} kbps")
        logger.info(f"  TX gain: {self.tx_gain} dB")
    
    def transmit_frame(self, frame):
        """フレームを送信"""
        try:
            # BPSK変調
            iq_samples = self.modulator.modulate(frame)
                    
                    # サンプルスケーリング
                    tx_samples = iq_samples * 2**14
                    
                    # SDRへの送信
            self.sdr.tx(tx_samples)
            logger.info(f"Transmitted frame with counter: {self.frame_gen.vcdu_counter-1}, size: {len(tx_samples)} samples")
                    
                    # 送信データを保存（デバッグ用）
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    np.save(os.path.join(self.timestamp_dir, f"tx_samples_{timestamp}.npy"), tx_samples)
                    
            return True
            
            except Exception as e:
                logger.error(f"Transmission error: {e}")
            return False
    
    def run_test(self):
        """送信テストを実行"""
        logger.info("Starting transmission test (1Mbps)")
        
        try:
            for i in range(MAX_TEST_ITERATIONS):
                # フレーム生成
                cadu_frame = self.frame_gen.generate_cadu_frame(i)
                
                # フレーム送信
                if self.transmit_frame(cadu_frame):
                    logger.info(f"Successfully transmitted frame {i+1}/{MAX_TEST_ITERATIONS}")
                else:
                    logger.error(f"Failed to transmit frame {i+1}/{MAX_TEST_ITERATIONS}")
                
                # 適度な間隔を空ける
                time.sleep(1.0)
            
            logger.info("Transmission test completed")
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            self.stop_test()
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    def close(self):
        """リソースのクリーンアップ"""
        logger.info("Closing SDR resources")
        if hasattr(self, 'sdr'):
            try:
                self.sdr.tx([0]) # 送信を停止
                del self.sdr
                logger.info("SDR resources closed")
            except Exception as e:
                logger.error(f"Error closing SDR: {e}")
    
    def stop_test(self):
        """テストを停止する"""
        logger.info("テストを停止しています...")
        self.close()
        logger.info("テストが停止されました")
        return True

def main():
    """メイン関数"""
    global test_instance
    
    parser = argparse.ArgumentParser(description='1 Mbps PlutoSDR CCSDS transmission test')
    parser.add_argument('--freq', type=float, default=5.84e9,
                        help='Center frequency in Hz (default: 5.84 GHz)')
    parser.add_argument('--sample-rate', type=float, default=4e6,
                        help='Sample rate in Hz (default: 4 MSps)')
    parser.add_argument('--bit-rate', type=float, default=1e6,
                        help='Bit rate in bps (default: 1 Mbps)')
    parser.add_argument('--tx-gain', type=float, default=-10,
                        help='TX gain in dB (default: -10 dB)')
    parser.add_argument('--device-uri', type=str, default='ip:192.168.2.1',
                        help='SDR device URI (default: ip:192.168.2.1)')

    args = parser.parse_args()

    try:
        # テストインスタンスをグローバル変数として保存
        test_instance = Transmitter(args)
        test_instance.run_test()
    except KeyboardInterrupt:
        logger.info("KeyboardInterruptを検出しました")
        if test_instance:
            test_instance.stop_test()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nプログラムを終了します...')
        if test_instance is not None:
            test_instance.stop_test()
        sys.exit(0)