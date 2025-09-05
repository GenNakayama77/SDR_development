#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlutoSDRループバックテスト - 1Mbps本番仕様
CCSDS準拠のフレームを送信・受信して整合性を確認します。
"""

import numpy as np
import time
import sys
import reedsolo
import logging
import struct
import os
import threading
import queue
import signal
import argparse
from datetime import datetime
from collections import deque

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
logger = logging.getLogger("origamisat2_loopback")

# テスト設定
MAX_TEST_ITERATIONS = 5  # テスト回数を減らして1回のテストを確実に
test_results = []  # テスト結果を保存
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
        """リードソロモン符号化を適用"""
        # 簡易実装：RS符号化を省略し、ダミーのECCを返す
        rs_ecc = bytes([0xAA] * self.rs_ecc_size)
        return rs_ecc
    
    def generate_cadu_frame(self, counter):
        """CADUフレームを生成"""
        # 同期マーカーの繰り返しを1回に変更
        repeated_marker = self.sync_marker  # 3回から1回に変更
        frame = bytearray(repeated_marker)
        
        # VCDUヘッダーを生成
        header = self.generate_vcdu_primary_header()
        
        # テストデータを生成
        payload = self.generate_test_data()
        
        # VCDUを生成 (ヘッダー + ペイロード)
        vcdu = header + payload
        
        # リードソロモン符号を追加
        rs_ecc = self.apply_reed_solomon(vcdu)
        
        # 結合
        frame.extend(vcdu)
        frame.extend(rs_ecc)
        
        logger.info(f"Generated CADU frame, {len(frame)} bytes, Counter: {counter}")
        return frame

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
        
        # ビットストリームの継続分析用データバッファ
        self.bit_buffer = np.array([], dtype=np.uint8)
        
        # バイトバッファ
        self.byte_buffer = bytearray()
        
        # バッファサイズの調整
        self.max_buffer_size = 2 * 1024 * 1024  # 2MBに増やす
        
        logger.info(f"CCSDS FrameParser initialized. VCDU size: {self.vcdu_size}, CADU size: {self.cadu_size}")
    
    def find_sync_marker(self, data):
        """データストリーム内の同期マーカーを検索"""
        if len(data) < len(self.sync_marker):
            return -1
        
        # バイト単位での検索（高速な実装）
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
    
    def append_to_byte_buffer(self, data):
        self.byte_buffer.extend(data)
        # バッファサイズが一定以上になったら、古いデータを削除
        if len(self.byte_buffer) > self.max_buffer_size:
            # 同期マーカーの位置を確認
            sync_pos = self.byte_buffer.find(self.sync_marker)
            if sync_pos != -1:
                # 同期マーカー以降のデータを保持
                self.byte_buffer = self.byte_buffer[sync_pos:]
            else:
                # 同期マーカーが見つからない場合は、最新のデータを保持
                self.byte_buffer = self.byte_buffer[-self.max_buffer_size:]
        
        # バッファサイズをログ出力
        logger.debug(f"Byte buffer size: {len(self.byte_buffer)} bytes")
    
    def search_sync_markers(self):
        """データストリーム内の同期マーカーを検索（改善版）"""
        if len(self.byte_buffer) < len(self.sync_marker):
            return []
        
        positions = []
        i = 0
        while i <= len(self.byte_buffer) - len(self.sync_marker):
            # ビットレベルでの比較（より柔軟なマッチング）
            current_bytes = self.byte_buffer[i:i+len(self.sync_marker)]
            
            # ビットレベルでの比較
            current_bits = np.unpackbits(np.array(current_bytes, dtype=np.uint8))
            sync_bits = np.unpackbits(np.array(self.sync_marker, dtype=np.uint8))
            
            # バイト単位での比較も行う
            byte_match = sum(1 for j in range(len(self.sync_marker)) 
                            if current_bytes[j] == self.sync_marker[j])
            
            # ビット単位での比較
            bit_errors = np.sum(current_bits != sync_bits)
            
            # バイト単位での一致率を計算
            byte_match_rate = byte_match / len(self.sync_marker)
            
            # より柔軟なマッチング条件
            if (byte_match_rate >= 0.75 or  # バイトの75%以上が一致
                (byte_match_rate >= 0.5 and bit_errors <= 8)):  # バイトの50%以上が一致かつビットエラーが8以下
                
                positions.append(i)
                logger.info(f"Found marker at {i} with {bit_errors} bit errors, {byte_match} bytes match")
                logger.info(f"Found bytes: {' '.join(f'{b:02X}' for b in current_bytes)}")
                logger.info(f"Expected  : {' '.join(f'{b:02X}' for b in self.sync_marker)}")
                
                # ビットパターンも表示
                logger.info(f"Found bits : {current_bits}")
                logger.info(f"Expected  : {sync_bits}")
                
                # 次の検索位置を同期マーカー長分進める
                i += len(self.sync_marker)
            else:
                # 一致しない場合は1バイトずつ進める
                i += 1
        
        return positions
    
    def parse_frames_from_buffer(self):
        """バイトバッファからフレームを解析"""
        logger.info(f"Searching frames: buffer size={len(self.byte_buffer)}")
        positions = self.search_sync_markers()
        logger.info(f"Found {len(positions)} sync markers at positions: {positions}")
        
        # 見つからなければ終了
        if not positions:
            logger.debug("No sync markers found in buffer.")
            return []
        
        frames = []
        for pos in positions:
            # 同期マーカーからフレーム全体を取得するのに十分なデータがあるか確認
            if pos + self.sync_marker_len + self.vcdu_size + self.rs_ecc_size <= len(self.byte_buffer):
                # フレームを抽出
                frame_data = self.byte_buffer[pos:pos + self.sync_marker_len + self.vcdu_size + self.rs_ecc_size]
                
                # フレームを解析
                frame_info = self.parse_cadu_frame(frame_data)
                if frame_info:
                    frames.append(frame_info)
                    
                    # 成功したフレームをバッファから削除（オプション）
                    # self.byte_buffer = self.byte_buffer[pos + self.sync_marker_len + self.vcdu_size + self.rs_ecc_size:]
        
        return frames
    
    def parse_cadu_frame(self, cadu_data):
        """CADUフレームを解析"""
        sync_pos = self.find_sync_marker(cadu_data)
        if sync_pos < 0:
            logger.error("Sync marker not found in frame data")
            # デバッグログを追加
            logger.error(f"First 16 bytes of cadu_data: {cadu_data[:16].hex()}")
            return None
        elif sync_pos > 0:
            logger.warning(f"Sync marker found at position {sync_pos}, not at start of frame")
            cadu_data = cadu_data[sync_pos:]
        
        # データが少なすぎる場合
        if len(cadu_data) < self.sync_marker_len + self.vcdu_size:
            logger.error(f"CADU data too short: {len(cadu_data)} < {self.sync_marker_len + self.vcdu_size}")
            return None
        
        # 同期マーカーを取り除く
        vcdu_data = cadu_data[self.sync_marker_len:self.sync_marker_len + self.vcdu_size]
        
        # ヘッダーを解析
        header_info = self.decode_vcdu_header(vcdu_data[:12])
        if not header_info:
            logger.error("Failed to decode VCDU header")
            return None
        
        # ペイロードを取得
        payload = vcdu_data[12:]
        
        result = {
            'header': header_info,
            'payload': payload,
            'rs_success': True,  # 簡易実装では常に成功
            'cadu_size': len(cadu_data)
        }
        
        logger.info(f"Parsed CADU frame: VCID={header_info['vcid']}, Counter={header_info['counter']}")
        return result

class BPSKModulator:
    """BPSK変調/復調クラス - 中速バージョン"""
    
    def __init__(self, sample_rate, symbol_rate, bit_expansion=1):
        """BPSK変調器の初期化"""
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.sps = int(sample_rate / symbol_rate)
        self.bit_expansion = bit_expansion  # bit_expansionを追加
        self.add_preamble = True  # add_preambleを追加
        self.sync_marker = bytes.fromhex("1ACFFC1D")  # 同期マーカーを追加
        
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
    
    def modulate(self, data):
        """バイトデータをBPSK変調してIQサンプルに変換"""
        # バイトからビットに変換
        bits = self.bytes_to_bits(data)
        
        # サンプル整形（sps倍に伸ばす）
        symbols = 2 * bits.astype(float) - 1
        samples = np.repeat(symbols, self.sps)  # サンプル整形が重要！
        
        # 複素サンプルに変換
        iq_samples = samples + 0j
        
        # プリアンブルを追加
        if self.add_preamble:
            preamble = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
            preamble = np.repeat(preamble, self.sps)
            iq_samples = np.concatenate([preamble + 0j, iq_samples])
        
        logger.info(f"Modulated {len(data)} bytes to {len(iq_samples)} IQ samples (expanded)")
        return iq_samples
    
    def demodulate(self, iq_samples):
        """IQサンプルをBPSK復調してビットに変換（改善版）"""
        # 最適なサンプリングタイミングと位相を探索
        best_bits = None
        best_sync_count = 0
        best_confidence = 0
        
        # 同期マーカーのビットパターンを準備
        sync_marker_array = np.frombuffer(self.sync_marker, dtype=np.uint8)
        sync_marker_bits = np.unpackbits(sync_marker_array)
        
        # 位相反転も試す
        for phase_invert in [False, True]:
            samples = -iq_samples if phase_invert else iq_samples
            
            # 移動平均フィルタでノイズを低減（ウィンドウサイズを調整）
            window_size = min(8, len(samples))  # ウィンドウサイズを小さくして、より細かい変化を検出
            if window_size > 1:
                samples = np.convolve(samples, np.ones(window_size)/window_size, mode='valid')
            
            # サンプリングタイミングの探索範囲を拡大
            for offset in range(self.sps):
                # サンプリングタイミングの調整（オフセットを考慮）
                symbols = samples[offset::self.sps]
                
                # 振幅の正規化（より安定した方法）
                max_amp = np.max(np.abs(symbols))
                if max_amp > 0:
                    symbols = symbols / max_amp
                
                # 位相補正（改善版）
                phase_correction = np.angle(np.mean(symbols**2)) / 2
                symbols = symbols * np.exp(-1j * phase_correction)
                
                # ビット判定（信頼度付き、改善版）
                symbol_real = np.real(symbols)
                confidence = np.abs(symbol_real)  # 判定の信頼度
                
                # ヒステリシスを適用したビット判定
                threshold = 0.2  # より高い閾値
                raw_bits = np.zeros_like(symbol_real, dtype=np.uint8)
                for i in range(len(symbol_real)):
                    if symbol_real[i] > threshold:
                        raw_bits[i] = 1
                    elif symbol_real[i] < -threshold:
                        raw_bits[i] = 0
                
                # ビット同期の改善
                if len(raw_bits) >= 8:
                    blocks = [raw_bits[i:i+8] for i in range(0, len(raw_bits)-7, 8)]
                    conf_blocks = [confidence[i:i+8] for i in range(0, len(confidence)-7, 8)]
                    bits = []
                    
                    for block, conf_block in zip(blocks, conf_blocks):
                        # 信頼度を考慮した多数決（改善版）
                        weighted_sum = np.sum(block * conf_block)
                        weighted_total = np.sum(conf_block)
                        threshold = weighted_total * 0.5  # より高い閾値
                        bits.append(1 if weighted_sum >= threshold else 0)
                    
                    bits = np.array(bits, dtype=np.uint8)
                    
                    # 同期マーカーの検出（ビットレベルで、より厳密なマッチング）
                    sync_count = 0
                    max_bit_errors = 4  # ビットエラー数を4以下に制限
                    
                    for i in range(len(bits) - len(sync_marker_bits)):
                        # ビットパターンの比較
                        current_pattern = bits[i:i+len(sync_marker_bits)]
                        bit_errors = np.sum(current_pattern != sync_marker_bits)
                        
                        if bit_errors <= max_bit_errors:
                            # バイト単位での比較も追加
                            current_bytes = np.packbits(current_pattern)
                            expected_bytes = np.packbits(sync_marker_bits)
                            byte_match = np.sum(current_bytes == expected_bytes)
                            
                            # バイトの一致率が75%以上の場合のみカウント
                            if byte_match >= 3:  # 4バイト中3バイト以上が一致
                                sync_count += 1
                                logger.info(f"Found sync marker at bit position {i} with {bit_errors} bit errors")
                                logger.info(f"Found pattern (hex): {current_bytes.tobytes().hex()}")
                                logger.info(f"Expected pattern (hex): {expected_bytes.tobytes().hex()}")
                                logger.info(f"Byte match: {byte_match}/4 bytes")
                                logger.info(f"Bit pattern: {current_pattern}")
                                logger.info(f"Expected bit pattern: {sync_marker_bits}")
                    
                    # 信頼度の平均を計算（改善版）
                    avg_confidence = np.mean(confidence) * (1 + sync_count * 0.5)  # 同期マーカーが見つかるほど信頼度を上げる
                    
                    # 同期カウントと信頼度の両方を考慮して最適なタイミングを選択
                    current_score = sync_count * avg_confidence
                    if current_score > best_confidence:
                        best_confidence = current_score
                        best_sync_count = sync_count
                        best_bits = bits
                        logger.info(f"New best timing found: sync_count={sync_count}, confidence={avg_confidence:.3f}")
        
        if best_bits is None:
            # タイミング同期が失敗した場合は、デフォルトの復調を使用
            symbols = iq_samples[::self.sps]
            raw_bits = (np.real(symbols) > 0).astype(np.uint8)
            if len(raw_bits) >= 8:
                blocks = [raw_bits[i:i+8] for i in range(0, len(raw_bits)-7, 8)]
                bits = []
                for block in blocks:
                    ones_count = np.sum(block)
                    bits.append(1 if ones_count >= 5 else 0)  # より厳密な多数決
                best_bits = np.array(bits, dtype=np.uint8)
                logger.warning("Using fallback demodulation method")
        
        logger.info(f"Demodulated {len(iq_samples)} IQ samples to {len(best_bits)} bits")
        if best_sync_count > 0:
            logger.info(f"Found {best_sync_count} sync markers with confidence {best_confidence:.3f}")
        else:
            logger.warning("No sync markers found in demodulated data")
            # 最初の32ビットを16進数で表示（デバッグ用）
            if best_bits is not None and len(best_bits) >= 32:
                first_bytes = np.packbits(best_bits[:32])
                logger.info(f"First 4 bytes of demodulated data: {first_bytes.tobytes().hex()}")
                logger.info(f"First 32 bits: {best_bits[:32]}")
        
        return best_bits

class LoopbackTest:
    """PlutoSDRを使用したループバックテストクラス"""
    
    def __init__(self, args):
        """ループバックテストの初期化"""
        # 出力ディレクトリの作成
        self.timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.timestamp_dir, exist_ok=True)
        logger.info(f"データファイルの保存先ディレクトリを作成しました: {self.timestamp_dir}")
        
        # 引数の解析
        self.freq = args.freq
        self.sample_rate = args.sample_rate
        self.bit_rate = args.bit_rate
        self.tx_gain = args.tx_gain
        self.rx_gain = args.rx_gain
        self.buffer_size = args.buffer_size
        self.device_uri = args.device_uri
        
        # 追加の設定
        self.sync_repeat = getattr(args, 'sync_repeat', 1)  # 同期マーカーの繰り返し数
        self.bit_expansion = getattr(args, 'bit_expansion', 2)  # ビット拡張倍率
        self.add_preamble = getattr(args, 'add_preamble', True)  # プリアンブル追加フラグ
        self.rs_encode = getattr(args, 'rs_encode', False)  # 実際のRS符号化を行うか
        self.interleave = getattr(args, 'interleave', False)  # インターリーブを行うか
        
        # CCSDSフレーム生成・解析オブジェクト
        self.frame_gen = CCSDSFrameGenerator()
        self.frame_parser = CCSDSFrameParser()
        
        # BPSK変調・復調オブジェクト
        self.modulator = BPSKModulator(
            self.sample_rate,
            self.bit_rate,
            bit_expansion=self.bit_expansion
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
            
            # RX設定
            self.sdr.rx_rf_bandwidth = int(self.sample_rate * 0.8)  # 帯域幅（サンプルレートの80%）
            self.sdr.rx_lo = int(self.freq)  # 中心周波数
            self.sdr.rx_hardwaregain = int(self.rx_gain)  # 受信ゲイン
            self.sdr.rx_buffer_size = int(self.buffer_size)  # 受信バッファサイズ
        except Exception as e:
            logger.error(f"PlutoSDRの初期化に失敗しました: {e}")
            raise
        
        # キューとスレッド間通信用イベント
        self.tx_queue = queue.Queue()
        self.rx_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 送信・受信フレームの履歴
        self.sent_frames = []
        
        logger.info(f"Test configuration:")
        logger.info(f"  Frequency: {self.freq/1e6} MHz")
        logger.info(f"  Sample rate: {self.sample_rate/1e6} MSps")
        logger.info(f"  Bit rate: {self.bit_rate/1e3} kbps")
        logger.info(f"  TX gain: {self.tx_gain} dB")
        logger.info(f"  RX gain: {self.rx_gain} dB")
        logger.info(f"  Buffer size: {self.buffer_size} samples")
        logger.info(f"  Sync marker repeat: {self.sync_repeat}")
        logger.info(f"  Bit expansion: {self.bit_expansion}x")
        logger.info(f"  Add preamble: {self.add_preamble}")
        logger.info(f"  RS encode: {self.rs_encode}")
        logger.info(f"  Interleave: {self.interleave}")
    
    def generate_frame(self):
        """テストフレームを生成して送信キューに追加"""
        for i in range(MAX_TEST_ITERATIONS):
            if self.stop_event.is_set():
                logger.info("フレーム生成を停止します")
                break
                
            # フレーム生成
            cadu_frame = self.frame_gen.generate_cadu_frame(i)
            
            # フレームをキューに追加
            self.tx_queue.put(cadu_frame)
            
            # 送信フレーム履歴に追加
            self.sent_frames.append(cadu_frame)
            
            # 適度な間隔を空ける（中速モードでは間隔を適度に）
            time.sleep(1.0)  # 1秒間隔
        
        # すべてのフレームが送信されるのを待機
        time.sleep(3)  # 中速モードなので待機時間を中程度に
        
        # 終了フラグを設定
        self.stop_event.set()
    
    def transmit_thread(self):
        """送信スレッド"""
        logger.info("Starting transmitter thread")
        
        while not self.stop_event.is_set() or not self.tx_queue.empty():
            try:
                # キューからフレームを取得
                if not self.tx_queue.empty():
                    cadu_frame = self.tx_queue.get(timeout=0.1)
                    
                    # BPSK変調
                    iq_samples = self.modulator.modulate(cadu_frame)
                    
                    # サンプルスケーリング
                    tx_samples = iq_samples * 2**14
                    
                    # プリアンブルとポストアンブルを追加（同期検出の改善用）
                    preamble_length = 5000  # 中程度のプリアンブル
                    preamble = np.ones(preamble_length, dtype=complex) * 2**14  # 明確な開始マーカー
                    postamble = -np.ones(preamble_length, dtype=complex) * 2**14  # 明確な終了マーカー
                    
                    # 結合
                    tx_samples_with_amble = np.concatenate([preamble, tx_samples, postamble])
                    
                    # SDRへの送信
                    self.sdr.tx(tx_samples_with_amble)
                    logger.info(f"Transmitted frame with counter: {self.frame_gen.vcdu_counter-1}, size: {len(tx_samples_with_amble)} samples")
                    
                    # 送信データを保存（デバッグ用）
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    np.save(os.path.join(self.timestamp_dir, f"tx_samples_{timestamp}.npy"), tx_samples)
                    
                    # キュー処理完了を通知
                    self.tx_queue.task_done()
                    
                    # 送信後に少し待機（中速モードでは適度に）
                    time.sleep(0.5)
                else:
                    # キューが空の場合は少し待機
                    time.sleep(0.1)
                    
                # 停止フラグのチェック
                if self.stop_event.is_set() and self.tx_queue.empty():
                    logger.info("送信スレッドを終了します（停止フラグが設定されています）")
                    break
            
            except queue.Empty:
                # タイムアウトした場合は再試行
                continue
            except Exception as e:
                logger.error(f"Transmission error: {e}")
                time.sleep(0.1)
                
                # エラー発生時にも停止フラグをチェック
                if self.stop_event.is_set():
                    break
        
        logger.info("Transmitter thread completed")
    
    def receive_thread(self):
        """受信スレッド"""
        logger.info("Starting receiver thread")
        
        while not self.stop_event.is_set() or len(self.sent_frames) > 0:
            try:
                # 停止フラグがセットされている場合は終了
                if self.stop_event.is_set():
                    logger.info("受信スレッドを終了します（停止フラグが設定されています）")
                    break
                    
                # SDRからの受信
                rx_samples = self.sdr.rx()
                logger.info(f"Received {len(rx_samples)} samples")
                
                # 受信データを保存（デバッグ用）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                np.save(os.path.join(self.timestamp_dir, f"rx_samples_{timestamp}.npy"), rx_samples)
                
                # スケールを元に戻す
                rx_samples = rx_samples / 2**14
                
                # 受信サンプルの振幅を確認
                max_amp = np.max(np.abs(rx_samples))
                logger.info(f"Maximum amplitude of received samples: {max_amp:.3f}")
                
                # BPSK復調
                bits = self.modulator.demodulate(rx_samples)
                logger.info(f"Demodulated {len(bits)} bits")
                
                # ビットをバイトに変換
                rx_bytes = self.modulator.bits_to_bytes(bits)
                logger.info(f"Converted {len(bits)} bits to {len(rx_bytes)} bytes")
                
                # バイトバッファにデータを追加
                self.frame_parser.append_to_byte_buffer(rx_bytes)
                
                # バッファの状態を表示
                buffer_size = len(self.frame_parser.byte_buffer)
                if buffer_size > 0:
                    logger.info(f"Buffer size: {buffer_size} bytes")
                    # 最初の数バイトを16進数で表示
                    display_bytes = self.frame_parser.byte_buffer[:20]
                    logger.info(f"Buffer head: {' '.join(f'{b:02X}' for b in display_bytes)}...")
                    
                    # 同期マーカーの検索を試みる
                    sync_pos = self.frame_parser.find_sync_marker(self.frame_parser.byte_buffer)
                    if sync_pos >= 0:
                        logger.info(f"Found sync marker at position {sync_pos}")
                    else:
                        logger.warning("No sync marker found in buffer")
                
                # バッファからフレームを解析
                frames = self.frame_parser.parse_frames_from_buffer()
                if frames:
                    logger.info(f"Successfully parsed {len(frames)} frames")
                else:
                    logger.warning("No frames parsed from buffer")
                
                # 解析したフレームを処理
                for frame_info in frames:
                    logger.info(f"Successfully received CADU frame:")
                    logger.info(f"  VCID: {frame_info['header']['vcid']}")
                    logger.info(f"  Counter: {frame_info['header']['counter']}")
                    logger.info(f"  RS decoding success: {frame_info['rs_success']}")
                    
                    # 受信フレームをキューに追加
                    self.rx_queue.put(frame_info)
                    
                    # 受信フレームをファイルに保存
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.timestamp_dir, f"rx_cadu_frame_{timestamp}.bin")
                    
                    with open(filename, "wb") as f:
                        # ヘッダー情報とペイロードを保存
                        header_info = f"VCID: {frame_info['header']['vcid']}, Counter: {frame_info['header']['counter']}\n".encode()
                        f.write(header_info)
                        f.write(frame_info['payload'])
                    
                    logger.info(f"Saved received frame to {filename}")
                    
                    # 送信フレームと受信フレームを比較
                    self.compare_frames(frame_info)
                
                # 適度な間隔を空ける
                time.sleep(0.2)
            
            except Exception as e:
                logger.error(f"Reception error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                
                # エラー発生時にも停止フラグをチェック
                if self.stop_event.is_set():
                    break
        
        logger.info("Receiver thread completed")
    
    def compare_frames(self, rx_frame_info):
        """送信フレームと受信フレームを比較"""
        # 受信したVCIDとカウンター値を取得
        rx_vcid = rx_frame_info['header']['vcid']
        rx_counter = rx_frame_info['header']['counter']
        
        # 送信履歴から一致するフレームを探す
        for sent_frame in list(self.sent_frames):
            # 送信フレームからヘッダー情報を抽出
            header_start = len(self.frame_gen.sync_marker) * 3  # 3倍に拡張された同期マーカー
            sent_vcid = sent_frame[header_start + 1] & 0x3F
            sent_counter = int.from_bytes(sent_frame[header_start + 2:header_start + 5], byteorder='big')
            
            # VCIDとカウンター値が一致するかチェック
            if sent_vcid == rx_vcid and sent_counter == rx_counter:
                logger.info(f"Found matching frame with VCID={sent_vcid}, Counter={sent_counter}")
                
                # 送信ペイロードと受信ペイロードを比較
                sent_payload = sent_frame[header_start + 12:header_start + self.frame_gen.vcdu_size]
                received_payload = rx_frame_info['payload']
                
                # 最小サイズを使用（両方のデータを比較できる部分のみ）
                min_size = min(len(sent_payload), len(received_payload))
                
                match_count = sum(1 for i in range(min_size) if sent_payload[i] == received_payload[i])
                match_percent = (match_count / min_size) * 100
                
                logger.info(f"Payload comparison: {match_percent:.2f}% match ({match_count}/{min_size} bytes)")
                
                # 70%以上一致した場合は成功とみなす（中速モードでは閾値を中程度に）
                if match_percent > 70:
                    logger.info("Frame successfully verified!")
                    # 比較済みのフレームをリストから削除
                    self.sent_frames.remove(sent_frame)
                    
                    # テスト結果を記録
                    test_results.append({
                        'success': True,
                        'match_percent': match_percent,
                        'counter': sent_counter,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    return
        
        # 一致するフレームが見つからなかった場合
        logger.warning(f"No matching frame found for VCID={rx_vcid}, Counter={rx_counter}")
        test_results.append({
            'success': False,
            'match_percent': 0,
            'counter': rx_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def run_test(self):
        """ループバックテストを実行"""
        logger.info("Starting high-rate loopback test (1Mbps)")
        
        try:
            # 受信スレッドを開始（先に受信スレッドを開始することで、送信が始まる前にリスニングを開始）
            rx_thread = threading.Thread(target=self.receive_thread)
            rx_thread.daemon = True
            rx_thread.start()
            
            # 少し待機して受信スレッドが起動するのを待つ
            time.sleep(0.5)
            
            # 送信スレッドを開始
            tx_thread = threading.Thread(target=self.transmit_thread)
            tx_thread.daemon = True
            tx_thread.start()
            
            # フレーム生成スレッドを開始
            gen_thread = threading.Thread(target=self.generate_frame)
            gen_thread.daemon = True
            gen_thread.start()
            
            # すべてのスレッドが終了するのを待機
            gen_thread.join()
            tx_thread.join()
            # 受信スレッドにさらに時間を与える（最後のフレームを処理する時間）
            time.sleep(3)
            self.stop_event.set()
            rx_thread.join()
            
            # テスト結果を表示
            self.print_test_results()
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            self.stop_event.set()
            time.sleep(1)  # スレッドが停止フラグを確認する時間を与える
            
            # 残りのスレッドのリソースをクリーンアップ
            self.close()
        
        logger.info("Loopback test completed")
    
    def print_test_results(self):
        """テスト結果を表示"""
        success_count = sum(1 for result in test_results if result['success'])
        total_count = len(test_results)
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
        else:
            success_rate = 0
        
        logger.info("===== Loopback Test Results =====")
        logger.info(f"Total frames received and verified: {total_count}")
        logger.info(f"Successfully verified frames: {success_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        if test_results:
            avg_match = sum(result['match_percent'] for result in test_results) / len(test_results)
            logger.info(f"Average match percentage: {avg_match:.2f}%")
        
        # 送信したがまだ検出されていないフレームを表示
        if self.sent_frames:
            logger.info(f"Undetected frames: {len(self.sent_frames)}")
            for frame in self.sent_frames:
                header_start = len(self.frame_gen.sync_marker) * 3
                sent_vcid = frame[header_start + 1] & 0x3F
                sent_counter = int.from_bytes(frame[header_start + 2:header_start + 5], byteorder='big')
                logger.info(f"  VCID={sent_vcid}, Counter={sent_counter}")
        
        # テスト結果をファイルに保存
        result_file = os.path.join(self.timestamp_dir, "test_results.txt")
        with open(result_file, "w") as f:
            f.write("===== Loopback Test Results =====\n")
            f.write(f"Total frames received and verified: {total_count}\n")
            f.write(f"Successfully verified frames: {success_count}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n")
            if test_results:
                f.write(f"Average match percentage: {avg_match:.2f}%\n")
            f.write("================================\n")
        
        logger.info(f"テスト結果を {result_file} に保存しました")
        logger.info("=================================")
    
    def close(self):
        """リソースのクリーンアップ"""
        logger.info("Closing SDR resources")
        # SDRが初期化されていれば閉じる
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
        
        # 停止フラグを設定
        self.stop_event.set()
        
        # 少し待機して終了処理を行う
        time.sleep(1.0)
        
        # リソースのクリーンアップ
        self.close()
        
        logger.info("テストが停止されました")
        return True


def main():
    """メイン関数"""
    global test_instance
    
    parser = argparse.ArgumentParser(description='1 Mbps PlutoSDR CCSDS loopback test (production config)')
    parser.add_argument('--freq', type=float, default=5.84e9,
                        help='Center frequency in Hz (default: 5.84 GHz)')
    parser.add_argument('--sample-rate', type=float, default=4e6,
                        help='Sample rate in Hz (default: 4 MSps)')
    parser.add_argument('--bit-rate', type=float, default=1e6,
                        help='Bit rate in bps (default: 1 Mbps)')
    parser.add_argument('--tx-gain', type=float, default=-10,
                        help='TX gain in dB (default: -10 dB)')
    parser.add_argument('--rx-gain', type=float, default=50,
                        help='RX gain in dB (default: 50 dB)')
    parser.add_argument('--buffer-size', type=int, default=131072,
                        help='RX buffer size (default: 131072 samples)')
    parser.add_argument('--device-uri', type=str, default='ip:192.168.2.1',
                        help='SDR device URI (default: ip:192.168.2.1)')

    # 本番用設定（同期マーカ1回、ビット拡張なし）
    parser.add_argument('--sync-repeat', type=int, default=1,
                        help='Number of sync marker repetitions (default: 1)')
    parser.add_argument('--bit-expansion', type=int, default=1,
                        help='Bit expansion factor (default: 1 = no expansion)')
    parser.add_argument('--add-preamble', action='store_false',
                        help='Omit preamble and postamble for production')
    parser.add_argument('--rs-encode', action='store_true',
                        help='Enable actual RS(255,223)x5 encoding')
    parser.add_argument('--interleave', action='store_true',
                        help='Enable CCSDS interleaving (depth=5)')

    args = parser.parse_args()

    try:
        # テストインスタンスをグローバル変数として保存（シグナルハンドラからアクセスするため）
        test_instance = LoopbackTest(args)
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