#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDSフレームのソフトウェアテスト - 1Mbps
SDRを使用せず、ソフトウェア上でCCSDSフレームの生成、変調、復調、復号を確認します。
"""

import numpy as np
import time
import sys
import reedsolo as rs
import logging
import struct
import os
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ccsds_test")

class CCSDSFrameGenerator:
    """CCSDSフレーム生成クラス"""
    
    def __init__(self):
        """CCSDSフレーム生成器の初期化"""
        # CCSDSフレームパラメータ
        self.vcdu_size = 1115
        self.sync_marker = bytes([0x1A, 0xCF, 0xFC, 0x1D])
        
        # リードソロモン符号のパラメータ
        self.rs_n = 255  # 符号長
        self.rs_k = 223  # 情報長
        self.rs_codec = rs.RSCodec(self.rs_n - self.rs_k)
        self.interleave_depth = 5
        
        # CADUサイズの計算
        self.cadu_size = len(self.sync_marker) + self.vcdu_size + (self.rs_n - self.rs_k) * self.interleave_depth
        
        logger.info(f"CCSDS FrameGenerator initialized. VCDU size: {self.vcdu_size}, CADU size: {self.cadu_size}")
    
    def generate_vcdu_header(self, frame_counter):
        """VCDUヘッダーを生成"""
        # VCDUヘッダー（8バイト）
        header = bytearray(8)
        header[0] = 0x00  # バージョン番号
        header[1] = 0x00  # スペースクラフトID
        header[2] = 0x00  # 仮想チャネルID
        header[3] = frame_counter & 0xFF  # フレームカウンター（下位バイト）
        header[4] = (frame_counter >> 8) & 0xFF  # フレームカウンター（上位バイト）
        header[5] = 0x00  # オペレーションモード
        header[6] = 0x00  # 予約
        header[7] = 0x00  # 予約
        return bytes(header)
    
    def apply_reed_solomon(self, data):
        """リードソロモン符号化を適用"""
        try:
            # サイズ整合チェック
            expected_size = self.rs_k * self.interleave_depth  # 223 * 5 = 1115
            if len(data) != expected_size:
                logger.error(f"RS input size mismatch: expected {expected_size} bytes, got {len(data)} bytes")
                return None
            
            # データをサブブロックに分割（列優先でインターリーブ）
            sub_blocks = []
            for i in range(self.interleave_depth):
                start_idx = i * self.rs_k
                end_idx = start_idx + self.rs_k
                sub_block = data[start_idx:end_idx]
                sub_blocks.append(sub_block)
            
            # 各サブブロックを符号化
            encoded_blocks = []
            for i, block in enumerate(sub_blocks):
                try:
                    # リードソロモン符号化
                    encoded_block = self.rs_codec.encode(block)
                    if encoded_block is not None:
                        encoded_blocks.append(encoded_block)
                        logger.info(f"RS encoding successful for block {i}")
                    else:
                        logger.error(f"RS encoding failed for block {i}")
                        return None
                except Exception as e:
                    logger.error(f"RS encoding error in block {i}: {e}")
                    return None
            
            # インターリーブ（列優先：5ブロック × 255B を縦詰め）
            interleaved_data = bytearray()
            for i in range(self.rs_n):  # 255回
                for block in encoded_blocks:  # 5ブロック
                    interleaved_data.append(block[i])
            
            # サイズ整合チェック
            expected_output_size = self.rs_n * self.interleave_depth  # 255 * 5 = 1275
            if len(interleaved_data) != expected_output_size:
                logger.error(f"RS output size mismatch: expected {expected_output_size} bytes, got {len(interleaved_data)} bytes")
                return None
            
            logger.info(f"RS encoding completed: {len(data)} bytes -> {len(interleaved_data)} bytes")
            return bytes(interleaved_data)
        
        except Exception as e:
            logger.error(f"RS encoding failed: {e}")
            return None

    def apply_convolutional_encoding(self, data):
        """畳み込み符号化を適用"""
        try:
            # サイズ整合チェック
            expected_input_size = self.rs_n * self.interleave_depth  # 255 * 5 = 1275
            if len(data) != expected_input_size:
                logger.error(f"Convolutional input size mismatch: expected {expected_input_size} bytes, got {len(data)} bytes")
                return None
            
            # 生成多項式 (g1 = 171, g2 = 133)
            g1 = [1, 1, 1, 1, 0, 0, 1]
            g2 = [1, 0, 1, 1, 0, 1, 1]
            
            # シフトレジスタ（7ビット）
            shift_reg = [0] * 7
        
            # 出力ビット列
            output = []
            
            # 畳み込み符号化（MSBファースト）
            for byte in data:
                for bit in range(8):
                    # シフトレジスタの更新（MSBファースト）
                    shift_reg.pop(0)
                    shift_reg.append((byte >> (7-bit)) & 1)
                    
                    # 出力ビットの計算
                    out1 = sum(g1[i] * shift_reg[i] for i in range(7)) % 2
                    out2 = sum(g2[i] * shift_reg[i] for i in range(7)) % 2
                    
                    output.extend([out1, out2])
            
            # トレーリングビットの追加（K-1=6回のゼロ入力で各回2bit出力）
            for _ in range(6):
                shift_reg.pop(0)
                shift_reg.append(0)
                out1 = sum(g1[i] * shift_reg[i] for i in range(7)) % 2
                out2 = sum(g2[i] * shift_reg[i] for i in range(7)) % 2
                output.extend([out1, out2])
            
            # サイズ整合チェック
            expected_output_bits = len(data) * 8 * 2 + 12  # 1275 * 8 * 2 + 12 = 20412
            if len(output) != expected_output_bits:
                logger.error(f"Convolutional output size mismatch: expected {expected_output_bits} bits, got {len(output)} bits")
                return None
            
            logger.info(f"Convolutional encoding completed: {len(data)} bytes -> {len(output)} bits")
            return np.array(output, dtype=np.uint8)
        
        except Exception as e:
            logger.error(f"Convolutional encoding failed: {e}")
            return None

    def generate_cadu_frame(self, frame_counter, payload_data=None):
        """CADUフレームを生成"""
        try:
            # 同期マーカー（非符号化）
            sync_marker = self.sync_marker
            
            # VCDUヘッダー
            vcdu_header = self.generate_vcdu_header(frame_counter)
            
            # ペイロードデータ（1107バイト）
            if payload_data is None:
                payload_data = bytes([0xCC] * 1107)
            
            # VCDU全体（ヘッダ8B + ペイロード1107B = 1115B）
            vcdu_data = vcdu_header + payload_data
            
            # VCDU全体をリードソロモン符号化
            encoded_data = self.apply_reed_solomon(vcdu_data)
            if encoded_data is None:
                return None
            
            # 畳み込み符号化
            encoded_bits = self.apply_convolutional_encoding(encoded_data)
            if encoded_bits is None:
                return None
            
            # ビットをバイトに変換
            encoded_bytes = np.packbits(encoded_bits)
            
            # CADUフレームの生成（ASM + 符号化済みVCDU）
            cadu_frame = sync_marker + encoded_bytes.tobytes()
            
            logger.info(f"Generated CADU frame: {len(cadu_frame)} bytes")
            return cadu_frame
            
        except Exception as e:
            logger.error(f"CADU frame generation failed: {e}")
            return None

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

class BPSKDemodulator:
    """BPSK復調クラス"""
    
    def __init__(self, sample_rate, symbol_rate):
        """BPSK復調器の初期化"""
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.sps = int(sample_rate / symbol_rate)
        self.best_confidence = 0.0  # 最良の信頼度を保持する属性を追加
        
        logger.info(f"BPSK Demodulator initialized. Sample rate: {sample_rate/1e6} MHz, Symbol rate: {symbol_rate/1e6} MHz")
        logger.info(f"Samples per symbol: {self.sps}")
    
    def demodulate(self, iq_samples):
        """IQサンプルをBPSK復調してビットに変換"""
        try:
            # デバッグ情報
            logger.info(f"Input IQ samples: {len(iq_samples)} samples")
            
            # 振幅の正規化
            max_amp = np.max(np.abs(iq_samples))
            if max_amp > 0:
                iq_samples = iq_samples / max_amp
            
            # 位相補正
            phase_correction = np.angle(np.mean(iq_samples**2)) / 2
            iq_samples = iq_samples * np.exp(-1j * phase_correction)
            
            # サンプリングタイミングの探索
            best_bits = None
            self.best_confidence = 0  # 信頼度をリセット
            
            # 移動平均フィルタでノイズを低減
            window_size = min(8, len(iq_samples))
            if window_size > 1:
                samples = np.convolve(iq_samples, np.ones(window_size)/window_size, mode='valid')
            else:
                samples = iq_samples
            
            # サンプリングタイミングの探索
            for offset in range(self.sps):
                # サンプリングタイミングの調整
                symbols = samples[offset::self.sps]
                
                # ビット判定（実部の符号で判定）
                bits = (np.real(symbols) > 0).astype(np.uint8)
                
                # 信頼度の計算（実部の絶対値の平均）
                confidence = np.mean(np.abs(np.real(symbols)))
                
                if confidence > self.best_confidence:
                    self.best_confidence = confidence
                    best_bits = bits
            
            # デバッグ情報
            if best_bits is not None:
                logger.info(f"Demodulated bits: {len(best_bits)} bits")
                logger.info(f"First 20 bits: {best_bits[:20]}")
                logger.info(f"Best confidence: {self.best_confidence:.3f}")
            
            return best_bits
            
        except Exception as e:
            logger.error(f"Demodulation failed: {e}")
            return None

class CCSDSFrameDecoder:
    """CCSDSフレームデコーダクラス"""
    def __init__(self):
        """初期化"""
        self.sync_marker = bytes([0x1A, 0xCF, 0xFC, 0x1D])
        self.original_data = None
        self.rs_codec = rs.RSCodec(32)  # 訂正能力32シンボル（255-223=32）
        self.interleave_depth = 5  # インターリーブ深さ
        self.rs_n = 255  # RS符号のブロック長
        self.rs_k = 223  # RS符号の情報長
        self.cadu_size = self.rs_n * self.interleave_depth  # CADUサイズ（1275バイト）
        self.vcdu_size = self.rs_k * self.interleave_depth  # VCDUサイズ（1115バイト）
        self.logger = logging.getLogger('ccsds_test')
        
    def set_original_data(self, data):
        """元のデータを設定"""
        if len(data) != self.vcdu_size:
            self.logger.error(f"Invalid original data size: {len(data)} bytes (expected {self.vcdu_size} bytes)")
            return False
        self.original_data = data
        return True
    
    def compare_data(self, vcdu_data):
        """デコードされたVCDUデータと元のデータを比較"""
        try:
            # 入力はVCDU全体(1115B)、内部でヘッダ(8B)を除去してペイロード(1107B)を比較
            if len(vcdu_data) != 1115:
                self.logger.error(f"VCDU size mismatch: expected 1115 bytes, got {len(vcdu_data)} bytes")
                return False
            
            # VCDUからペイロード部分を抽出（8Bヘッダを除去）
            decoded_payload = vcdu_data[8:]
            
            # 元データからペイロード部分を抽出
            original_payload = self.original_data[8:]
            
            if len(original_payload) != len(decoded_payload):
                self.logger.error(f"Payload size mismatch: original={len(original_payload)} bytes, decoded={len(decoded_payload)} bytes")
                return False
                
            # バイト単位での比較
            match_count = sum(1 for a, b in zip(original_payload, decoded_payload) if a == b)
            match_percentage = (match_count / len(original_payload)) * 100
            
            # 最初の32バイトを16進数で表示
            self.logger.info(f"Original payload (first 32 bytes): {original_payload[:32].hex()}")
            self.logger.info(f"Decoded payload (first 32 bytes): {decoded_payload[:32].hex()}")
            
            if match_percentage < 100:
                self.logger.error(f"Data mismatch: {match_percentage:.2f}% match")
                # 不一致位置を表示
                mismatch_positions = [(i, a, b) for i, (a, b) in enumerate(zip(original_payload, decoded_payload)) if a != b]
                self.logger.error(f"Total mismatches: {len(mismatch_positions)}")
                # 最初の10個の不一致位置を表示
                for pos, orig, dec in mismatch_positions[:10]:
                    self.logger.error(f"Position {pos}: original={orig:02x}, decoded={dec:02x}")
                return False
                
            self.logger.info(f"Data match: 100% ({match_count}/{len(original_payload)} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error comparing data: {e}")
            return False
    
    def viterbi_decode_bits(self, bit_data):
        """64状態(K=7) Viterbi復号を適用（ビット列入力）"""
        try:
            # サイズ整合チェック
            expected_input_bits = 1275 * 8 * 2 + 12  # 20412 bits
            if len(bit_data) != expected_input_bits:
                logger.error(f"Viterbi input size mismatch: expected {expected_input_bits} bits, got {len(bit_data)} bits")
                return None
            
            # 生成多項式 (171, 133)_oct = (1111001, 1011011)_bin
            g1 = [1, 1, 1, 1, 0, 0, 1]  # 171_oct
            g2 = [1, 0, 1, 1, 0, 1, 1]  # 133_oct
            
            # メトリクスの初期化（int32で安全に）
            metrics = np.full(64, np.iinfo(np.int32).max, dtype=np.int32)
            metrics[0] = 0  # 初期状態は0
            paths = [[] for _ in range(64)]
            
            # トレリスを探索
            for i in range(0, len(bit_data)-1, 2):
                if i+1 >= len(bit_data):
                    break
                    
                new_metrics = np.full(64, np.iinfo(np.int32).max, dtype=np.int32)
                new_paths = [[] for _ in range(64)]
                
                for state in range(64):
                    if metrics[state] == np.iinfo(np.int32).max:
                        continue
                        
                    for input_bit in [0, 1]:
                        # 仕様通りの状態遷移: next_state = ((state >> 1) | (u << 5)) & 0x3F
                        next_state = ((state >> 1) | (input_bit << 5)) & 0x3F
                        
                        # シフトレジスタの状態を計算
                        shift_reg = [(state >> j) & 1 for j in range(6)] + [input_bit]
                        
                        # 出力ビットの計算
                        out1 = sum(g1[j] * shift_reg[j] for j in range(7)) % 2
                        out2 = sum(g2[j] * shift_reg[j] for j in range(7)) % 2
                        
                        # ハード判定での距離計算（int32で安全に）
                        dist = int(abs(int(bit_data[i]) - out1) + abs(int(bit_data[i+1]) - out2))
                        new_metric = int(metrics[state]) + dist
                        
                        if new_metric < new_metrics[next_state]:
                            new_metrics[next_state] = new_metric
                            new_paths[next_state] = paths[state] + [input_bit]
                
                metrics = new_metrics
                paths = new_paths
            
            # 最適パスを選択（終端状態は0を選ぶ）
            best_state = 0  # テイル付きなので最終状態は0
            decoded_bits = np.array(paths[best_state], dtype=np.uint8)
            
            # テイルビットを除去（末尾6bitを捨てる）
            if len(decoded_bits) >= 6:
                decoded_bits = decoded_bits[:-6]
            
            # ビットをバイトに変換
            decoded_bytes = np.packbits(decoded_bits)
            
            # サイズ整合チェック
            expected_output_size = 1275  # 1275 bytes
            if len(decoded_bytes) != expected_output_size:
                logger.error(f"Viterbi output size mismatch: expected {expected_output_size} bytes, got {len(decoded_bytes)} bytes")
                return None
                
            self.logger.info(f"Viterbi decoding completed: {len(bit_data)} bits -> {len(decoded_bytes)} bytes")
            return decoded_bytes
            
        except Exception as e:
            self.logger.error(f"Viterbi decoding failed: {str(e)}")
            return None
    
    def deinterleave(self, data):
        """デインターリーブ処理（行優先で戻す）"""
        try:
            # サイズ整合チェック
            expected_input_size = self.rs_n * self.interleave_depth  # 255 * 5 = 1275
            if len(data) != expected_input_size:
                self.logger.error(f"Deinterleave input size mismatch: expected {expected_input_size} bytes, got {len(data)} bytes")
                return None
            
            # 入力データをbytes型に変換
            if isinstance(data, np.ndarray):
                data = data.tobytes()
            elif isinstance(data, bytearray):
                data = bytes(data)
            
            # デインターリーブ（行優先で戻す：受信は列優先の逆）
            blocks = [bytearray(self.rs_n) for _ in range(self.interleave_depth)]
            for i in range(self.rs_n):  # 255回
                for j in range(self.interleave_depth):  # 5ブロック
                    blocks[j][i] = data[i * self.interleave_depth + j]
            
            self.logger.info(f"Deinterleaving completed: {len(data)} bytes -> {len(blocks)} blocks of {self.rs_n} bytes each")
            return blocks
        except Exception as e:
            self.logger.error(f"Deinterleaving failed: {e}")
            return None

    def apply_reed_solomon_decode(self, data):
        """リードソロモン復号を適用"""
        try:
            # サイズ整合チェック
            expected_input_size = self.rs_n * self.interleave_depth  # 255 * 5 = 1275
            if len(data) != expected_input_size:
                self.logger.error(f"RS decode input size mismatch: expected {expected_input_size} bytes, got {len(data)} bytes")
                return None
            
            # デインターリーブ
            rs_blocks = self.deinterleave(data)
            if rs_blocks is None:
                return None
            
            # 各ブロックをbytes型に変換
            rs_blocks = [bytes(block) for block in rs_blocks]
            
            if not rs_blocks:
                self.logger.error("No valid blocks found after deinterleaving")
                return None
            
            # 各サブブロックを復号
            decoded_blocks = []
            for i, block in enumerate(rs_blocks):
                try:
                    # リードソロモン復号
                    decoded_block = self.rs_codec.decode(block)
                    if isinstance(decoded_block, tuple):
                        decoded_block = decoded_block[0]
                    # bytearrayをbytesに変換
                    if isinstance(decoded_block, bytearray):
                        decoded_block = bytes(decoded_block)
                    decoded_blocks.append(decoded_block)
                    self.logger.info(f"RS decoding successful for block {i}")
                except Exception as e:
                    self.logger.error(f"RS decoding error in block {i}: {e}")
                    return None
            
            # デコードされたブロックを結合（行優先で戻す）
            decoded_data = bytearray()
            for i in range(self.rs_k):  # rs_k = 223
                for block in decoded_blocks:  # 5ブロック
                    if i < len(block):
                        decoded_data.append(block[i])
            
            # サイズ整合チェック
            expected_output_size = self.rs_k * self.interleave_depth  # 223 * 5 = 1115
            if len(decoded_data) != expected_output_size:
                self.logger.error(f"RS decode output size mismatch: expected {expected_output_size} bytes, got {len(decoded_data)} bytes")
                return None
            
            self.logger.info(f"RS decoding completed: {len(data)} bytes -> {len(decoded_data)} bytes")
            return bytes(decoded_data)
            
        except Exception as e:
            self.logger.error(f"RS decoding failed: {e}")
            return None
    
    def find_sync_marker(self, data):
        """同期マーカーを検索"""
        try:
            # NumPy配列の場合はバイト列に変換
            if isinstance(data, np.ndarray):
                data = data.tobytes()
            
            # 同期マーカーの検索
            sync_pos = data.find(self.sync_marker)
            if sync_pos == -1:
                self.logger.error("Sync marker not found")
                self.logger.error(f"First 32 bytes: {data[:32].hex()}")
                return None
            
            self.logger.info(f"Sync marker found at position {sync_pos}")
            self.logger.info(f"Sync marker bytes: {data[sync_pos:sync_pos+4].hex()}")
            return sync_pos
            
        except Exception as e:
            self.logger.error(f"Error finding sync marker: {e}")
            return None
    
    def decode_cadu_frame(self, frame_data):
        """CADUフレームをデコード"""
        try:
            # 同期マーカーの検出
            sync_pos = self.find_sync_marker(frame_data)
            if sync_pos is None:
                self.logger.error("Sync marker not found")
                return None
                
            # 同期マーカー以降のデータを取得
            data_after_sync = frame_data[sync_pos + 4:]
            
            # ビット列に変換してからViterbi復号
            bit_data = np.unpackbits(np.frombuffer(data_after_sync, dtype=np.uint8))
            decoded_data = self.viterbi_decode_bits(bit_data)
            if decoded_data is None:
                self.logger.error("Viterbi decoding failed")
                return None
                
            # リードソロモン復号
            rs_decoded_data = self.apply_reed_solomon_decode(decoded_data)
            if rs_decoded_data is None:
                self.logger.error("Reed-Solomon decoding failed")
                return None
                
            # データサイズの確認
            if len(rs_decoded_data) != self.vcdu_size:
                self.logger.error(f"Decoded data size mismatch: expected {self.vcdu_size} bytes, got {len(rs_decoded_data)} bytes")
                return None
                
            # データの比較（VCDU全体を渡す）
            if not self.compare_data(rs_decoded_data):
                self.logger.error("Decoded data does not match original data")
                return None
                
            return rs_decoded_data
            
        except Exception as e:
            self.logger.error(f"CADU frame decoding failed: {str(e)}")
            return None

def self_test():
    """無雑音環境での往復一致テスト"""
    logger.info("=== Self Test: Noiseless Round-trip Test ===")
    
    try:
        # フレーム生成器の初期化
        frame_generator = CCSDSFrameGenerator()
        
        # テストデータの生成（固定シード）
        np.random.seed(42)
        test_payload = np.random.randint(0, 256, size=1107, dtype=np.uint8)
        
        # VCDU全体を生成
        vcdu_header = frame_generator.generate_vcdu_header(0)
        full_vcdu = vcdu_header + test_payload.tobytes()
        
        # 符号化処理
        encoded_data = frame_generator.apply_reed_solomon(full_vcdu)
        if encoded_data is None:
            logger.error("RS encoding failed in self test")
            return False
            
        encoded_bits = frame_generator.apply_convolutional_encoding(encoded_data)
        if encoded_bits is None:
            logger.error("Convolutional encoding failed in self test")
            return False
        
        # 復号処理
        frame_decoder = CCSDSFrameDecoder()
        frame_decoder.set_original_data(full_vcdu)
        
        # Viterbi復号（直接ビット列を渡す）
        decoded_data = frame_decoder.viterbi_decode_bits(encoded_bits)
        if decoded_data is None:
            logger.error("Viterbi decoding failed in self test")
            return False
        
        # RS復号
        rs_decoded_data = frame_decoder.apply_reed_solomon_decode(decoded_data)
        if rs_decoded_data is None:
            logger.error("RS decoding failed in self test")
            return False
        
        # データ比較
        if frame_decoder.compare_data(rs_decoded_data):
            logger.info("✓ Self test passed: 100% data match")
            return True
        else:
            logger.error("✗ Self test failed: data mismatch")
            return False
            
    except Exception as e:
        logger.error(f"Self test error: {e}")
        return False

def main():
    """メイン処理"""
    # 自己テストを実行
    if not self_test():
        logger.error("Self test failed, aborting main test")
        return 1
    
    # テストパラメータ
    sample_rate = 8e6  # 8 MSps
    symbol_rate = 1e6  # 1 Mbps
    num_frames = 5     # テストフレーム数
    
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # フレーム生成器の初期化
        frame_generator = CCSDSFrameGenerator()
        
        # BPSK変調器の初期化
        modulator = BPSKModulator(sample_rate, symbol_rate)
        
        # BPSK復調器の初期化
        demodulator = BPSKDemodulator(sample_rate, symbol_rate)
        
        # テストデータの生成（固定シードで再現性を確保）
        np.random.seed(42)
        test_payload = np.random.randint(0, 256, size=1107, dtype=np.uint8)  # ペイロードは1107バイト
        logger.info(f"Generated test payload: {len(test_payload)} bytes")
        
        # 各フレームの処理
        for i in range(num_frames):
            logger.info(f"\n=== Test Frame {i+1}/{num_frames} ===")
            
            # 送信フレームの生成（ペイロードを指定）
            frame = frame_generator.generate_cadu_frame(i, test_payload.tobytes())
            logger.info(f"Generated CADU frame: {len(frame)} bytes")
            
            # 送信フレームの保存
            tx_frame_path = os.path.join(timestamp_dir, f"tx_frame_{i}.bin")
            with open(tx_frame_path, 'wb') as f:
                f.write(frame)
            logger.info(f"Saved transmitted frame to {tx_frame_path}")
            
            # BPSK変調
            iq_samples = modulator.modulate(frame)
            logger.info(f"Modulated {len(frame)} bytes to {len(iq_samples)} IQ samples")
            logger.info(f"Generated IQ samples: {len(iq_samples)} samples")
            
            # IQサンプルの保存
            tx_iq_path = os.path.join(timestamp_dir, f"tx_iq_{i}.npy")
            np.save(tx_iq_path, iq_samples)
            logger.info(f"Saved IQ samples to {tx_iq_path}")
            
            # BPSK復調
            demodulated_bits = demodulator.demodulate(iq_samples)
            logger.info(f"Demodulated bits: {len(demodulated_bits)} bits")
            logger.info(f"First 20 bits: {demodulated_bits[:20]}")
            logger.info(f"Best confidence: {demodulator.best_confidence:.3f}")
            
            # ビット列をバイト列に変換
            received_bytes = np.packbits(demodulated_bits)
            logger.info(f"Demodulated bits: {len(demodulated_bits)} bits")
            logger.info(f"Converted to bytes: {len(received_bytes)} bytes")
            logger.info(f"Received bytes hex: {received_bytes[:32].tobytes().hex()}")
            
            # 受信フレームの保存
            rx_frame_path = os.path.join(timestamp_dir, f"rx_frame_{i}.bin")
            with open(rx_frame_path, 'wb') as f:
                f.write(received_bytes)
            logger.info(f"Saved received frame to {rx_frame_path}")
            
            # フレームの比較
            logger.info("Comparing first 32 bytes:")
            logger.info(f"Original: {frame[:32].hex()}")
            logger.info(f"Received: {received_bytes[:32].tobytes().hex()}")
            
            # ビットマッチ率の計算
            frame_bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
            bit_match = np.mean(demodulated_bits == frame_bits)
            logger.info(f"Bit match: {bit_match*100:.2f}% ({int(bit_match*len(demodulated_bits))}/{len(demodulated_bits)} bits)")
            
            # フレームデコーダの初期化と設定
            frame_decoder = CCSDSFrameDecoder()
            vcdu_header = frame_generator.generate_vcdu_header(i)
            full_vcdu = vcdu_header + test_payload.tobytes()  # VCDUヘッダー(8バイト) + データ(1107バイト) = 1115バイト
            frame_decoder.set_original_data(full_vcdu)  # VCDU全体を設定
            
            # CADUフレームのデコード
            try:
                decoded_vcdu = frame_decoder.decode_cadu_frame(received_bytes)
                if decoded_vcdu is not None:
                    logger.info(f"Successfully decoded CADU frame: {len(decoded_vcdu)} bytes")
                    # デコードされたデータの比較（VCDU全体を渡す）
                    if frame_decoder.compare_data(decoded_vcdu):
                        logger.info("✓ Data match: 100%")
                    else:
                        logger.error("✗ Data mismatch detected")
                else:
                    logger.error("Failed to decode CADU frame")
            except Exception as e:
                logger.error(f"Error during CADU frame decoding: {str(e)}")
            
            time.sleep(0.5)  # フレーム間の待機時間
            
        logger.info("\nTest completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
        sys.exit(main())