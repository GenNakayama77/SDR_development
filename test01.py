#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Origamisat-2 Cバンド通信実験コード
CCSDS準拠のデータフレーム送受信テスト
- PCでダミーデータを符号化
- PCで変調BPSK(1 Mbps)
- IQデータをSDRへ
- SDRからRFに変換、送信
- SDRで受信、IQに変換
- IQデータをPCへ
- PCで復調BPSK(1 Mbps)
- PCでダミーデータを復号
"""

import numpy as np
import argparse
import time
import sys
import reedsolo
import logging
import struct
import os
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
logger = logging.getLogger("origamisat2")

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
        
        # 残りのデータを0x5Aで埋める
        remaining = bytes([0x5A] * (self.vcdu_size - len(ad_data) - len(spare) - len(tx_settings) - 12))
        
        # 12はVCDUヘッダー長
        return ad_data + spare + tx_settings + remaining
    
    def apply_reed_solomon(self, data):
        """リードソロモン符号化を適用"""
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
                encoded = self.rs_codec.encode(sub_block)
                encoded_sub_blocks.append(encoded)
            
            # インターリーブを元に戻す
            transposed = []
            for i in range(self.rs_n):
                for j in range(len(encoded_sub_blocks)):
                    if i < len(encoded_sub_blocks[j]):
                        transposed.append(encoded_sub_blocks[j][i])
            
            encoded_blocks.extend(transposed)
        
        return bytes(encoded_blocks[:self.rs_ecc_size])
    
    def generate_cadu_frame(self):
        """CADUフレームを生成"""
        # VCDUヘッダーを生成
        header = self.generate_vcdu_primary_header()
        
        # テストデータを生成
        payload = self.generate_test_data()
        
        # VCDUを生成 (ヘッダー + ペイロード)
        vcdu = header + payload
        
        # リードソロモン符号を追加
        rs_ecc = self.apply_reed_solomon(vcdu)
        
        # CADUフレームを生成 (同期マーカー + VCDU + RS符号)
        cadu = self.sync_marker + vcdu + rs_ecc
        
        logger.info(f"Generated CADU frame, {len(cadu)} bytes")
        return cadu


class CCSDSFrameParser:
    """CCSDS準拠のデータフレームを解析するクラス"""
    
    def __init__(self):
        # リードソロモンコーデック (RS(255,223)、インターリーブ5)
        self.rs_n = 255
        self.rs_k = 223
        self.rs_codec = reedsolo.RSCodec(self.rs_n - self.rs_k)
        self.interleave_depth = 5
        
        # CCSDSフレームパラメータ
        self.vcdu_size = 1115
        self.sync_marker = bytes.fromhex("1ACFFC1D")
        self.sync_marker_len = len(self.sync_marker)
        self.rs_ecc_size = 160
        self.cadu_size = self.vcdu_size + self.sync_marker_len + self.rs_ecc_size
        
        logger.info(f"CCSDS FrameParser initialized. VCDU size: {self.vcdu_size}, CADU size: {self.cadu_size}")
    
    def find_sync_marker(self, data):
        """データストリーム内の同期マーカーを検索"""
        if len(data) < len(self.sync_marker):
            return -1
        
        for i in range(len(data) - len(self.sync_marker) + 1):
            if data[i:i+len(self.sync_marker)] == self.sync_marker:
                return i
        
        return -1
    
    def decode_vcdu_header(self, header_bytes):
        """VCDUヘッダーを解析"""
        if len(header_bytes) < 12:
            logger.error("Header too short for decoding")
            return None
        
        # バージョン、SCID、VCID
        version = (header_bytes[0] >> 6) & 0x03
        scid = ((header_bytes[0] & 0x3F) << 2) | ((header_bytes[1] >> 6) & 0x03)
        vcid = header_bytes[1] & 0x3F
        
        # VCDUカウンター
        counter = int.from_bytes(header_bytes[2:5], byteorder='big')
        
        # リプレイフラグ
        replay_flag = (header_bytes[5] >> 7) & 0x01
        
        # MPDUヘッダーポインタ
        mpdu_pointer = int.from_bytes(header_bytes[10:12], byteorder='big')
        
        return {
            'version': version,
            'scid': scid,
            'vcid': vcid,
            'counter': counter,
            'replay_flag': replay_flag,
            'mpdu_pointer': mpdu_pointer
        }
    
    def apply_reed_solomon_decode(self, encoded_data, vcdu_data):
        """リードソロモン復号を適用"""
        # インターリーブ処理を元に戻す
        
        # 簡易実装：ここでは実際の復号は行わず、エラーがないと仮定して元のデータを返す
        logger.info("Reed-Solomon decoding would happen here in a real implementation")
        
        # 実際の実装では、インターリーブ処理を元に戻し、RSコーデックを使用して復号する
        return True, 0  # 成功、修正エラー数
    
    def parse_cadu_frame(self, cadu_data):
        """CADUフレームを解析"""
        if len(cadu_data) < self.cadu_size:
            logger.error(f"CADU data too short: {len(cadu_data)} < {self.cadu_size}")
            return None
        
        # 同期マーカーを検証
        sync_pos = self.find_sync_marker(cadu_data)
        if sync_pos != 0:
            logger.error(f"Sync marker not found at start of frame, found at: {sync_pos}")
            if sync_pos < 0:
                return None
            cadu_data = cadu_data[sync_pos:]
            if len(cadu_data) < self.cadu_size:
                logger.error("Truncated frame after sync marker")
                return None
        
        # 同期マーカーを取り除く
        vcdu_with_rs = cadu_data[self.sync_marker_len:]
        
        # VCDUデータとRS ECC部分を分離
        vcdu_data = vcdu_with_rs[:self.vcdu_size]
        rs_ecc = vcdu_with_rs[self.vcdu_size:]
        
        # ヘッダーを解析
        header_info = self.decode_vcdu_header(vcdu_data[:12])
        if not header_info:
            logger.error("Failed to decode VCDU header")
            return None
        
        # リードソロモン復号
        success, errors = self.apply_reed_solomon_decode(rs_ecc, vcdu_data)
        
        # ペイロードを取得
        payload = vcdu_data[12:]
        
        result = {
            'header': header_info,
            'payload': payload,
            'rs_success': success,
            'rs_errors': errors,
            'cadu_size': len(cadu_data)
        }
        
        logger.info(f"Parsed CADU frame: VCID={header_info['vcid']}, Counter={header_info['counter']}")
        return result


class SDRInterface:
    """PlutoSDRとのインターフェースクラス"""
    
    def __init__(self, args):
        """SDRインターフェースの初期化"""
        self.args = args
        self.sdr = None
        self.simulation = args.simulation
        
        if not self.simulation:
            try:
                # SDRに接続
                self.sdr = adi.Pluto(args.device_uri)
                
                # 送信設定
                self.sdr.tx_lo = int(args.freq)
                self.sdr.tx_rf_bandwidth = int(args.sample_rate)
                self.sdr.tx_hardwaregain_chan0 = int(args.tx_gain)
                self.sdr.tx_cyclic_buffer = args.repeat
                
                # 受信設定
                self.sdr.rx_lo = int(args.freq)
                self.sdr.rx_rf_bandwidth = int(args.sample_rate)
                self.sdr.rx_hardwaregain_chan0 = int(args.rx_gain)
                self.sdr.rx_buffer_size = int(args.buffer_size)
                
                # サンプリングレート設定
                self.sdr.sample_rate = int(args.sample_rate)
                
                logger.info(f"SDR connected to {args.device_uri}")
                logger.info(f"Frequency: {args.freq/1e6} MHz")
                logger.info(f"Sample rate: {args.sample_rate/1e6} MSps")
            except Exception as e:
                logger.error(f"Failed to initialize SDR: {e}")
                if not args.force_simulation:
                    sys.exit(1)
                else:
                    logger.warning("Falling back to simulation mode")
                    self.simulation = True
    
    def transmit(self, samples):
        """IQサンプルを送信"""
        if self.simulation:
            # シミュレーションモードでは何もしない
            logger.info(f"Simulation: Transmitting {len(samples)} samples")
            # 送信サンプルを保存（受信シミュレーション用）
            np.save("tx_samples.npy", samples)
            return
        
        try:
            # サンプルスケーリング
            samples = samples * 2**14
            
            # SDRへの送信
            self.sdr.tx(samples)
            logger.info(f"Transmitted {len(samples)} samples")
        except Exception as e:
            logger.error(f"Transmission error: {e}")
    
    def receive(self):
        """SDRからIQサンプルを受信"""
        if self.simulation:
            # シミュレーションモード
            try:
                # 保存された送信サンプルを読み込む
                tx_samples = np.load("tx_samples.npy")
                rx_len = self.args.buffer_size
                
                # 送信サンプルの繰り返し
                repeats = int(np.ceil(rx_len / len(tx_samples)))
                tx_repeated = np.tile(tx_samples, repeats)[:rx_len]
                
                # 雑音の多いチャネルをシミュレーション
                snr_db = 20  # SNR (dB)
                signal_power = np.mean(np.abs(tx_repeated)**2)
                noise_power = signal_power / (10**(snr_db/10))
                noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, rx_len) + 
                                                1j * np.random.normal(0, 1, rx_len))
                
                # 位相オフセットを追加
                phase_offset = np.exp(1j * np.random.uniform(0, 2*np.pi))
                
                # 受信信号の生成
                rx_samples = tx_repeated * phase_offset + noise
                
                logger.info(f"Simulation: Received {len(rx_samples)} samples")
                # 受信データを保存
                np.save("rx_samples.npy", rx_samples)
                return rx_samples
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                # ノイズのみを返す
                rx_len = self.args.buffer_size
                return 0.1 * (np.random.normal(0, 1, rx_len) + 1j * np.random.normal(0, 1, rx_len))
        
        try:
            # SDRからの受信
            rx_samples = self.sdr.rx()
            logger.info(f"Received {len(rx_samples)} samples")
            
            # スケールを元に戻す
            rx_samples = rx_samples / 2**14
            return rx_samples
        except Exception as e:
            logger.error(f"Reception error: {e}")
            return np.zeros(self.args.buffer_size, dtype=complex)
    
    def close(self):
        """SDRリソースの解放"""
        if not self.simulation and self.sdr:
            logger.info("Closing SDR connection")


class BPSKModulator:
    """BPSK変調/復調クラス"""
    
    def __init__(self, sample_rate, symbol_rate):
        """BPSK変調器の初期化"""
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.sps = int(sample_rate / symbol_rate)  # サンプル/シンボル比率
        
        logger.info(f"BPSK Modulator initialized. Sample rate: {sample_rate/1e6} MHz, Symbol rate: {symbol_rate/1e6} MHz")
        logger.info(f"Samples per symbol: {self.sps}")
    
    def bytes_to_bits(self, data_bytes):
        """バイトデータをビット配列に変換"""
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        return bits
    
    def bits_to_bytes(self, bits):
        """ビット配列をバイトデータに変換"""
        # ビット数を8の倍数に調整
        padding = (8 - len(bits) % 8) % 8
        if padding:
            bits = np.append(bits, np.zeros(padding, dtype=np.uint8))
        
        # バイトに変換
        bytes_data = np.packbits(bits)
        return bytes(bytes_data)
    
    def modulate(self, data_bytes):
        """バイトデータをBPSK変調してIQサンプルに変換"""
        # バイトからビットに変換
        bits = self.bytes_to_bits(data_bytes)
        
        # BPSK変調: 0→-1, 1→1
        symbols = 2 * bits.astype(float) - 1
        
        # パルス整形（シンボル反復による簡易版）
        samples = np.repeat(symbols, self.sps)
        
        # 複素サンプルに変換（実部のみにデータを載せる）
        iq_samples = samples + 0j
        
        logger.info(f"Modulated {len(data_bytes)} bytes to {len(iq_samples)} IQ samples")
        return iq_samples
    
    def demodulate(self, iq_samples):
        """IQサンプルをBPSK復調してビット配列に変換"""
        # ダウンサンプリング（簡易版）
        symbols = iq_samples[::self.sps]
        
        # 実部の符号でビット判定
        bits = (np.real(symbols) > 0).astype(np.uint8)
        
        logger.info(f"Demodulated {len(iq_samples)} IQ samples to {len(bits)} bits")
        return bits


def transmit_mode(args):
    """送信モード: PCでデータを符号化・変調してSDRに送信"""
    # CCSDSフレーム生成器を初期化
    frame_gen = CCSDSFrameGenerator()
    
    # BPSK変調器を初期化
    modulator = BPSKModulator(args.sample_rate, args.bit_rate)
    
    # SDRインターフェースを初期化
    sdr = SDRInterface(args)
    
    # テストフレームを生成
    cadu_frame = frame_gen.generate_cadu_frame()
    
    # フレームをファイルに保存
    with open("tx_cadu_frame.bin", "wb") as f:
        f.write(cadu_frame)
    logger.info(f"Saved CADU frame to tx_cadu_frame.bin, {len(cadu_frame)} bytes")
    
    # BPSK変調
    iq_samples = modulator.modulate(cadu_frame)
    
    # SDRで送信
    sdr.transmit(iq_samples)
    
    # 繰り返し送信モード
    if args.repeat:
        try:
            logger.info("Transmitting in repeat mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Transmission stopped by user")
    
    # リソースのクリーンアップ
    sdr.close()
    logger.info("Transmission completed")


def receive_mode(args):
    """受信モード: SDRから受信したIQサンプルを復調・復号"""
    # CCSDSフレーム解析器を初期化
    frame_parser = CCSDSFrameParser()
    
    # BPSK復調器を初期化
    demodulator = BPSKModulator(args.sample_rate, args.bit_rate)
    
    # SDRインターフェースを初期化
    sdr = SDRInterface(args)
    
    try:
        logger.info("Reception started. Press Ctrl+C to stop.")
        
        while True:
            # SDRから受信
            rx_samples = sdr.receive()
            
            # BPSK復調
            bits = demodulator.demodulate(rx_samples)
            
            # ビットをバイトに変換
            rx_bytes = demodulator.bits_to_bytes(bits)
            
            # フレームを解析
            cadu_info = frame_parser.parse_cadu_frame(rx_bytes)
            if cadu_info:
                logger.info(f"Successfully received CADU frame:")
                logger.info(f"  VCID: {cadu_info['header']['vcid']}")
                logger.info(f"  Counter: {cadu_info['header']['counter']}")
                logger.info(f"  RS decoding success: {cadu_info['rs_success']}")
                
                # 受信フレームをファイルに保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rx_cadu_frame_{timestamp}.bin"
                with open(filename, "wb") as f:
                    # 実装を簡素化するため、受信データのバイトシーケンスをそのまま保存
                    f.write(rx_bytes[:frame_parser.cadu_size])
                logger.info(f"Saved received frame to {filename}")
            else:
                logger.warning("Failed to parse CADU frame from received data")
            
            # 次の受信まで少し待機
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Reception stopped by user")
    
    # リソースのクリーンアップ
    sdr.close()
    logger.info("Reception completed")


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Origamisat-2 C-band Communication Test')
    parser.add_argument('--mode', choices=['tx', 'rx'], required=True,
                      help='Operating mode (tx: transmit, rx: receive)')
    parser.add_argument('--freq', type=float, default=5.84e9,
                      help='Center frequency in Hz (default: 5.84 GHz)')
    parser.add_argument('--sample-rate', type=float, default=10e6,
                      help='Sample rate in Hz (default: 10 MSps)')
    parser.add_argument('--bit-rate', type=float, default=1e6,
                      help='Bit rate in bps (default: 1 Mbps)')
    parser.add_argument('--tx-gain', type=float, default=-10,
                      help='TX gain in dB (default: -10 dB)')
    parser.add_argument('--rx-gain', type=float, default=50,
                      help='RX gain in dB (default: 50 dB)')
    parser.add_argument('--buffer-size', type=int, default=16384,
                      help='RX buffer size (default: 16384)')
    parser.add_argument('--repeat', action='store_true',
                      help='Continuously repeat transmission')
    parser.add_argument('--device-uri', type=str, default='ip:192.168.2.1',
                      help='SDR device URI (default: ip:192.168.2.1)')
    parser.add_argument('--simulation', action='store_true',
                      help='Run in simulation mode (no SDR required)')
    parser.add_argument('--force-simulation', action='store_true',
                      help='Force simulation mode if SDR initialization fails')
    
    args = parser.parse_args()
    
    try:
        # 送信モード
        if args.mode == 'tx':
            logger.info("Starting transmitter mode")
            transmit_mode(args)
        
        # 受信モード
        elif args.mode == 'rx':
            logger.info("Starting receiver mode")
            receive_mode(args)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())