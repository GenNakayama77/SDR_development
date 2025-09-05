#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDS通信パイプラインの完全実装:
CADU生成→リードソロモン符号化→インターリーブ→畳み込み符号化→BPSK変調→PLUTO SDR送信→PLUTO SDR受信→BPSK復調→ビタビ復号→デインターリーブ→リードソロモン復号
CCSDS 131.0-B-3に準拠
"""

import numpy as np
import logging
import os
import time
from datetime import datetime
import reedsolo as rs
import matplotlib.pyplot as plt
import adi  # PLUTO SDRのPythonインターフェース
import scipy.signal as signal

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ccsds_pipeline")

class CCSDSFrameGenerator:
    """CCSDSフレーム生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.sync_marker = bytes([0x1A, 0xCF, 0xFC, 0x1D])  # CCSDS同期マーカー
        self.frame_counter = 0
        logger.info("CCSDSFrameGenerator initialized")
    
    def generate_vcdu_header(self, counter):
        """VCDUヘッダーの生成"""
        header = bytearray(8)
        header[0] = 0x00  # バージョン番号
        header[1] = 0x00  # スペースクラフトID
        header[2] = 0x00  # 仮想チャネルID
        header[3] = 0x00  # シグナリングフィールド
        header[4:8] = counter.to_bytes(4, 'big')  # フレームカウンター
        return header
    
    def generate_cadu_frame(self, data):
        """CADUフレームの生成"""
        try:
            # 同期マーカー + VCDUヘッダー + データ
            vcdu_header = self.generate_vcdu_header(self.frame_counter)
            frame = self.sync_marker + vcdu_header + data
            
            self.frame_counter += 1
            logger.info(f"Generated CADU frame: {len(frame)} bytes")
            logger.info(f"Frame counter: {self.frame_counter-1}")
            return frame
            
        except Exception as e:
            logger.error(f"Error generating CADU frame: {e}")
            return None
    
    def reset_counter(self, value=0):
        """フレームカウンターをリセット"""
        old_value = self.frame_counter
        self.frame_counter = value
        logger.info(f"Frame counter reset: {old_value} -> {value}")
        return old_value

class ReedSolomonProcessor:
    """リードソロモン符号化/復号処理クラス"""
    
    def __init__(self, interleave_depth=5):
        """初期化"""
        # リードソロモン符号のパラメータ（CCSDS 131.0-B-3に準拠）
        self.rs_n = 255  # 符号長
        self.rs_k = 223  # 情報長
        self.rs_t = 16   # 誤り訂正能力
        self.interleave_depth = interleave_depth  # インターリーブ深さ
        
        # リードソロモン符号器の初期化
        # 生成多項式: x^8 + x^7 + x^2 + x + 1
        self.rs_codec = rs.RSCodec(self.rs_n - self.rs_k, c_exp=8, prim=0x11D)
        
        logger.info(f"ReedSolomonProcessor initialized:")
        logger.info(f"  Block size (k): {self.rs_k} bytes")
        logger.info(f"  Code length (n): {self.rs_n} bytes")
        logger.info(f"  Error correction (t): {self.rs_t} bytes")
        logger.info(f"  Interleave depth: {self.interleave_depth}")
    
    def encode(self, data):
        """リードソロモン符号化"""
        try:
            # データサイズの確認と調整
            required_size = self.rs_k * self.interleave_depth
            if len(data) < required_size:
                logger.warning(f"Data size {len(data)} bytes is less than required {required_size} bytes. Padding with zeros.")
                data = data + bytearray([0] * (required_size - len(data)))
            elif len(data) > required_size:
                logger.warning(f"Data size {len(data)} bytes is greater than required {required_size} bytes. Truncating.")
                data = data[:required_size]
            
            # データをブロックに分割
            blocks = [data[i:i+self.rs_k] for i in range(0, len(data), self.rs_k)]
            if len(blocks) != self.interleave_depth:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.interleave_depth})")
                return None
            
            # 各ブロックを符号化
            encoded_blocks = []
            for i, block in enumerate(blocks):
                try:
                    encoded_block = self.rs_codec.encode(block)
                    encoded_blocks.append(encoded_block)
                    logger.info(f"RS encoding successful for block {i+1}")
                except Exception as e:
                    logger.error(f"RS encoding failed for block {i+1}: {e}")
                    return None
            
            # 符号化されたブロックを結合
            encoded_data = bytearray()
            for block in encoded_blocks:
                encoded_data.extend(block)
            
            logger.info(f"RS encoding completed: {len(data)} bytes -> {len(encoded_data)} bytes")
            return encoded_data
            
        except Exception as e:
            logger.error(f"RS encoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def decode(self, data):
        """リードソロモン復号"""
        try:
            # データサイズの確認
            required_size = self.rs_n * self.interleave_depth
            if len(data) != required_size:
                logger.error(f"Invalid data size: {len(data)} bytes (expected {required_size} bytes)")
                return None
            
            # データをブロックに分割
            blocks = [data[i:i+self.rs_n] for i in range(0, len(data), self.rs_n)]
            if len(blocks) != self.interleave_depth:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.interleave_depth})")
                return None
            
            # 各ブロックを復号
            decoded_blocks = []
            for i, block in enumerate(blocks):
                try:
                    # bytearrayをbytesに変換してからdecodeを呼び出す
                    block_bytes = bytes(block)
                    decoded_result = self.rs_codec.decode(block_bytes)
                    
                    # 復号結果の処理
                    if isinstance(decoded_result, tuple):
                        decoded_block = decoded_result[0]
                        # エラー訂正情報があれば表示
                        if len(decoded_result) > 1:
                            error_info = decoded_result[1]
                            if isinstance(error_info, list) and error_info:
                                logger.info(f"Corrected errors at positions: {error_info}")
                            elif isinstance(error_info, int) and error_info > 0:
                                logger.info(f"Corrected {error_info} errors")
                    else:
                        decoded_block = decoded_result
                    
                    # 情報部分のみを取得（パリティ部分を除く）
                    decoded_block = decoded_block[:self.rs_k]
                    decoded_blocks.append(decoded_block)
                    logger.info(f"RS decoding successful for block {i+1}")
                except Exception as e:
                    logger.error(f"RS decoding failed for block {i+1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            
            # 復号されたブロックを結合
            decoded_data = bytearray()
            for block in decoded_blocks:
                decoded_data.extend(block)
            
            logger.info(f"RS decoding completed: {len(data)} bytes -> {len(decoded_data)} bytes")
            return decoded_data
            
        except Exception as e:
            logger.error(f"RS decoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class InterleaveProcessor:
    """インターリーブ/デインターリーブ処理クラス"""
    
    def __init__(self, rs_processor):
        """初期化"""
        self.rs_processor = rs_processor
        self.block_size = rs_processor.rs_n  # RS符号化後のブロックサイズ
        self.interleave_depth = rs_processor.interleave_depth
        logger.info(f"InterleaveProcessor initialized")
        logger.info(f"  Block size: {self.block_size} bytes")
        logger.info(f"  Interleave depth: {self.interleave_depth}")
    
    def interleave(self, data):
        """インターリーブ処理"""
        try:
            # データサイズの確認
            required_size = self.block_size * self.interleave_depth
            if len(data) != required_size:
                logger.error(f"Invalid data size: {len(data)} bytes (expected {required_size} bytes)")
                return None
            
            # データをブロックに分割
            blocks = [data[i:i+self.block_size] for i in range(0, len(data), self.block_size)]
            if len(blocks) != self.interleave_depth:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.interleave_depth})")
                return None
            
            # インターリーブ
            interleaved_data = bytearray()
            for i in range(self.block_size):
                for block in blocks:
                    if i < len(block):
                        interleaved_data.append(block[i])
            
            logger.info(f"Interleaving completed: {len(data)} bytes -> {len(interleaved_data)} bytes")
            return interleaved_data
            
        except Exception as e:
            logger.error(f"Interleaving failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def deinterleave(self, data):
        """デインターリーブ処理"""
        try:
            # データサイズの確認
            required_size = self.block_size * self.interleave_depth
            if len(data) != required_size:
                logger.error(f"Invalid data size: {len(data)} bytes (expected {required_size} bytes)")
                return None
            
            # データをブロックに分割
            blocks = [bytearray(self.block_size) for _ in range(self.interleave_depth)]
            
            # デインターリーブ
            for i in range(self.block_size):
                for j in range(self.interleave_depth):
                    index = i * self.interleave_depth + j
                    if index < len(data):
                        blocks[j][i] = data[index]
            
            # ブロックを結合
            deinterleaved_data = bytearray()
            for block in blocks:
                deinterleaved_data.extend(block)
            
            logger.info(f"Deinterleaving completed: {len(data)} bytes -> {len(deinterleaved_data)} bytes")
            return deinterleaved_data
            
        except Exception as e:
            logger.error(f"Deinterleaving failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class ConvolutionalEncoder:
    """畳み込み符号化クラス（CCSDS 131.0-B-3に準拠）"""
    
    def __init__(self):
        """初期化"""
        # 生成多項式（CCSDS 131.0-B-3に準拠）
        self.g1 = 0o171  # 1 + x + x^2 + x^3 + x^6
        self.g2 = 0o133  # 1 + x + x^2 + x^4 + x^6
        self.constraint_length = 7  # K = 7
        self.rate = 1/2  # 符号化率 1/2
        self.state = 0  # 初期状態
        logger.info("ConvolutionalEncoder initialized")
    
    def encode(self, data):
        """畳み込み符号化"""
        try:
            # バイト列をビット列に変換
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            logger.info(f"Input data: {len(data)} bytes, {len(bits)} bits")
            
            # 符号化
            encoded_bits = []
            self.state = 0  # 状態をリセット
            
            # 各入力ビットの処理
            for bit in bits:
                # シフトレジスタの内容を計算
                register = (self.state << 1) | bit
                # 次の状態を計算（最上位ビットを落とす）
                self.state = register & 0x3F
                
                # G1生成多項式に対する出力
                output1 = 0
                for i in range(self.constraint_length):
                    if (self.g1 >> i) & 1:
                        output1 ^= (register >> i) & 1
                
                # G2生成多項式に対する出力
                output2 = 0
                for i in range(self.constraint_length):
                    if (self.g2 >> i) & 1:
                        output2 ^= (register >> i) & 1
                
                # 出力ビットを追加（順序に注意）
                encoded_bits.extend([output2, output1])
            
            # 終端処理（オプション）- レジスタを0に戻す
            termination_bits = []
            for _ in range(self.constraint_length - 1):
                register = (self.state << 1) | 0
                self.state = register & 0x3F
                
                output1 = 0
                for i in range(self.constraint_length):
                    if (self.g1 >> i) & 1:
                        output1 ^= (register >> i) & 1
                
                output2 = 0
                for i in range(self.constraint_length):
                    if (self.g2 >> i) & 1:
                        output2 ^= (register >> i) & 1
                
                termination_bits.extend([output2, output1])
            
            # 終端ビットを追加
            encoded_bits.extend(termination_bits)
            
            # ビット列をバイト列に変換
            # まず8の倍数長になるようにパディング
            padding_needed = (8 - len(encoded_bits) % 8) % 8
            if padding_needed > 0:
                encoded_bits.extend([0] * padding_needed)
                
            encoded_bits_np = np.array(encoded_bits, dtype=np.uint8)
            encoded_data = np.packbits(encoded_bits_np).tobytes()
            
            logger.info(f"Convolutional encoding completed: {len(data)} bytes -> {len(encoded_data)} bytes")
            logger.info(f"Input bits: {len(bits)}, Output bits: {len(encoded_bits)}")
            logger.info(f"Termination bits: {len(termination_bits)} bits")
            return encoded_data
            
        except Exception as e:
            logger.error(f"Convolutional encoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class ViterbiDecoder:
    """最適化されたビタビ復号クラス（CCSDS 131.0-B-3に準拠）"""
    
    def __init__(self, k=1, n=2, m=6):
        self.k = k  # 入力ビット数
        self.n = n  # 出力ビット数
        self.m = m  # シフトレジスタの段数
        self.num_states = 2 ** m  # 状態数
        self.state = 0  # 初期状態
        self.constraint_length = m + 1  # 拘束長 K = m + 1
        
        # CCSDS 131.0-B-3標準の生成多項式
        self.g1 = 0o171  # 1 + x + x^2 + x^3 + x^4 + x^5 + x^6
        self.g2 = 0o133  # 1 + x + x^2 + x^4 + x^6
        
        # トレリス構造の初期化 - 事前計算を利用
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)
        
        # 高速化: 事前計算されたブランチメトリックテーブル
        # 各状態での入力ビットに対する各受信シンボルでのブランチメトリック
        # 形状: [state, input_bit, received_symbol]
        self.branch_metrics = {}
        
        # トレリス構造の構築
        self._build_trellis()
        logger.info("ViterbiDecoder initialized")
        
    def _build_trellis(self):
        """トレリス構造を構築と事前計算を実行"""
        for state in range(self.num_states):
            for input_bit in range(2):
                # シフトレジスタの内容を計算
                register = (state << 1) | input_bit
                # 次の状態を計算（最上位ビットを落とす）
                self.next_state[state, input_bit] = register & 0x3F
                
                # G1生成多項式に対する出力
                output_1 = 0
                for i in range(self.constraint_length):
                    if (self.g1 >> i) & 1:
                        output_1 ^= (register >> i) & 1
                
                # G2生成多項式に対する出力
                output_2 = 0
                for i in range(self.constraint_length):
                    if (self.g2 >> i) & 1:
                        output_2 ^= (register >> i) & 1
                
                # 出力ビットの順序を修正（畳み込み符号化側と同じ順序）
                self.output_bits[state, input_bit] = [output_2, output_1]
    
    def decode(self, encoded_data, original_size=None):
        """最適化されたビタビ復号を実行"""
        try:
            # バイト列をビット列に変換
            encoded_bits = np.unpackbits(np.frombuffer(encoded_data, dtype=np.uint8))
            
            # エンコード時の終端処理による追加ビットを無視するために元のデータサイズを計算
            original_bit_length = (len(encoded_bits) // 2) - (self.constraint_length - 1)
            
            # シンボルペアに変換（高速化: 配列操作）
            symbol_pairs = encoded_bits[:len(encoded_bits) // 2 * 2].reshape(-1, 2)
            
            # 高速化のためのパラメータ
            num_symbols = len(symbol_pairs)
            
            # 性能計測開始
            start_time = datetime.now()
            
            # パスメトリックと生存パスの初期化（事前に完全に割り当て）
            path_metrics = np.full(self.num_states, np.inf)
            path_metrics[0] = 0  # 初期状態のメトリックを0に設定
            
            # 高速化: 前の状態のみを記録（入力ビットは後で計算可能）
            survivor_path = np.zeros((num_symbols, self.num_states), dtype=np.int32)
            
            # デバッグ情報
            logger.info(f"Viterbi input size: {len(encoded_data)} bytes, {len(encoded_bits)} bits")
            logger.info(f"Symbol pairs: {num_symbols}")
            logger.info(f"Expected original bits: {original_bit_length}")
            
            # 高速化: 一時配列を事前に割り当て（再利用）
            temp_metrics = np.zeros(self.num_states, dtype=np.float32)
            
            # 高速化: 状態遷移の事前計算を活用
            transitions = []
            for state in range(self.num_states):
                for input_bit in range(2):
                    next_s = self.next_state[state][input_bit]
                    out_bits = self.output_bits[state][input_bit]
                    transitions.append((state, input_bit, next_s, out_bits))
            
            # ビタビアルゴリズムの実行（前方パス）
            for t in range(num_symbols):
                # 高速化: 直接配列再利用
                temp_metrics.fill(np.inf)
                
                # 受信シンボル
                rx_symbol = symbol_pairs[t]
                
                # 高速化: 有効な状態と遷移のみを処理
                for state in range(self.num_states):
                    # 無効な状態はスキップ
                    if path_metrics[state] == np.inf:
                        continue
                        
                    # 現在の状態のメトリック
                    current_metric = path_metrics[state]
                    
                    # 状態からの各遷移を処理
                    for input_bit in range(2):
                        next_s = self.next_state[state][input_bit]
                        expected_bits = self.output_bits[state][input_bit]
                        
                        # ハミング距離の計算
                        # 高速化: 論理XORを使用してハミング距離計算
                        branch_metric = np.sum(np.logical_xor(expected_bits, rx_symbol))
                        new_metric = current_metric + branch_metric
                        
                        if new_metric < temp_metrics[next_s]:
                            temp_metrics[next_s] = new_metric
                            survivor_path[t, next_s] = state
                
                # 高速化: 参照の入れ替え（コピー不要）
                path_metrics, temp_metrics = temp_metrics, path_metrics
            
            # 性能計測用（前方パス）
            forward_time = datetime.now()
            forward_duration = (forward_time - start_time).total_seconds()
            logger.info(f"Forward path completed in {forward_duration:.3f} seconds")
            
            # トレースバック（後方パス）
            # 最小メトリックの状態から開始
            current_state = np.argmin(path_metrics)
            logger.info(f"Final state with minimum metric: {current_state}")
            
            # トレースバック結果を格納する配列（高速化: 一度に割り当て）
            # ビット配列を直接初期化し、後で反転
            decoded_bits = np.zeros(num_symbols, dtype=np.uint8)
            
            # 高速化: トレースバックのスピードアップ
            for t in range(num_symbols - 1, -1, -1):
                prev_state = survivor_path[t, current_state]
                # 前状態から現状態への遷移に必要な入力ビットを決定（高速計算）
                input_bit = 1 if self.next_state[prev_state, 0] != current_state else 0
                decoded_bits[t] = input_bit
                current_state = prev_state
            
            # 反転は不要 - 既に正しい順序でビットを格納
            # 終端処理ビットを取り除き、元のデータサイズに切り詰める
            if len(decoded_bits) > original_bit_length:
                decoded_bits = decoded_bits[:original_bit_length]
            
            # 元のバイト数を使用（指定された場合）
            target_size = original_size if original_size else (original_bit_length + 7) // 8
            logger.info(f"Target output size: {target_size} bytes")
            
            # 高速化: ビット列をバイト列に変換（切り詰め済みのビット配列）
            # まず8の倍数長になるようにパディング
            needed_bits = target_size * 8
            if len(decoded_bits) > needed_bits:
                decoded_bits = decoded_bits[:needed_bits]
            elif len(decoded_bits) < needed_bits:
                # 足りない場合はパディング - 高速: 事前割り当て
                padding = np.zeros(needed_bits - len(decoded_bits), dtype=np.uint8)
                decoded_bits = np.concatenate([decoded_bits, padding])
            
            # バイトに変換
            decoded_data = np.packbits(decoded_bits).tobytes()
            
            # 最終的なサイズ調整
            if len(decoded_data) != target_size:
                if len(decoded_data) > target_size:
                    decoded_data = decoded_data[:target_size]
                else:
                    # 足りない場合はパディング
                    decoded_data = decoded_data + bytes([0] * (target_size - len(decoded_data)))
            
            # 性能計測用（全体）
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            logger.info(f"Viterbi decoding completed: {len(encoded_data)} bytes -> {len(decoded_data)} bytes in {total_duration:.3f} seconds")
            logger.info(f"Number of decoded bits: {len(decoded_bits)} -> trimmed to {needed_bits}")
            return decoded_data
            
        except Exception as e:
            logger.error(f"Viterbi decoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

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

class PlutoSDRInterface:
    """PLUTO SDRインターフェースクラス"""
    
    def __init__(self, rx_freq=900e6, tx_freq=900e6, rx_rate=10e6, tx_rate=10e6, tx_gain=0, rx_gain=30):
        try:
            # PLUTO SDRに接続
            self.sdr = adi.Pluto()
            
            # 送信設定
            self.sdr.tx_lo = int(tx_freq)  # 周波数設定
            self.sdr.sample_rate = int(tx_rate)  # サンプリングレート
            self.sdr.tx_rf_bandwidth = int(tx_rate)  # RF帯域幅
            self.sdr.tx_hardwaregain_chan0 = tx_gain  # 送信ゲイン
            
            # 受信設定
            self.sdr.rx_lo = int(rx_freq)  # 周波数設定
            self.sdr.rx_rf_bandwidth = int(rx_rate)  # RF帯域幅
            self.sdr.gain_control_mode_chan0 = "manual"  # ゲイン制御モード
            self.sdr.rx_hardwaregain_chan0 = rx_gain  # 受信ゲイン
            
            # バッファサイズ設定
            self.sdr.rx_buffer_size = 2**18  # 受信バッファ
            
            # 設定の保存
            self.rx_rate = rx_rate
            self.tx_rate = tx_rate
            
            logger.info(f"PLUTO SDR initialized:")
            logger.info(f"  TX frequency: {tx_freq/1e6} MHz")
            logger.info(f"  RX frequency: {rx_freq/1e6} MHz")
            logger.info(f"  TX rate: {tx_rate/1e6} Msps")
            logger.info(f"  RX rate: {rx_rate/1e6} Msps")
            logger.info(f"  TX gain: {tx_gain} dB")
            logger.info(f"  RX gain: {rx_gain} dB")
            
        except Exception as e:
            logger.error(f"PLUTO SDR initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def transmit(self, i_samples, q_samples, cyclic=False):
        try:
            # 送信データの準備
            length = min(len(i_samples), len(q_samples))
            tx_data = i_samples[:length] + 1j * q_samples[:length]
            
            # 正規化（クリッピングを防ぐため）
            max_magnitude = np.max(np.abs(tx_data))
            if max_magnitude > 0:
                tx_data = tx_data * 0.8 / max_magnitude
            
            # ランプの適用
            ramp_length = min(1000, length // 10)
            if ramp_length > 0:
                ramp_up = np.linspace(0, 1, ramp_length)
                ramp_down = np.linspace(1, 0, ramp_length)
                tx_data[:ramp_length] *= ramp_up
                tx_data[-ramp_length:] *= ramp_down
            
            # 繰り返し送信モードの設定
            self.sdr.tx_cyclic_buffer = cyclic
            
            # 送信
            logger.info(f"Transmitting {length} samples...")
            self.sdr.tx(tx_data)  # ここに注意
            
            if not cyclic:
                logger.info("Transmission completed")
            else:
                logger.info("Cyclic transmission started")
                
            return True
            
        except Exception as e:
            logger.error(f"PLUTO SDR transmission failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def receive(self, num_samples=None, timeout=5.0):
        """
        PLUTO SDRで信号を受信
        Args:
            num_samples (int): 受信するサンプル数（Noneの場合はバッファサイズ）
            timeout (float): タイムアウト時間 [秒]
        Returns:
            tuple: (I信号, Q信号)
        """
        try:
            # 受信サンプル数の設定
            if num_samples is not None:
                self.sdr.rx_buffer_size = num_samples
            else:
                num_samples = self.sdr.rx_buffer_size
            
            # 受信開始時間
            start_time = time.time()
            logger.info(f"Receiving {num_samples} samples...")
            
            # 受信
            rx_data = self.sdr.rx()
            
            # 受信データの分解
            i_samples = np.real(rx_data)
            q_samples = np.imag(rx_data)
            
            # 受信信号強度
            signal_power = np.mean(np.abs(rx_data)**2)
            logger.info(f"Received {len(rx_data)} samples. Signal power: {10*np.log10(signal_power):.2f} dB")
            
            return i_samples, q_samples
            
        except Exception as e:
            logger.error(f"PLUTO SDR reception failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def close(self):
        """SDRの接続を閉じる"""
        try:
            # 繰り返し送信を停止
            self.sdr.tx_cyclic_buffer = False
            logger.info("PLUTO SDR connection closed")
        except Exception as e:
            logger.error(f"Error closing PLUTO SDR: {e}")

class SignalProcessor:
    """信号処理クラス（フィルタリング、同期などの追加処理）"""
    
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
        logger.info(f"SignalProcessor initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Symbol rate: {symbol_rate} bps")
        logger.info(f"  Samples per symbol: {self.samples_per_symbol}")
    
    def apply_rrc_filter(self, i_samples, q_samples, alpha=0.35, num_taps=101):
        """
        RRCフィルタを適用
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
            alpha (float): ロールオフ係数
            num_taps (int): フィルタタップ数
        Returns:
            tuple: (フィルタリングされたI信号, フィルタリングされたQ信号)
        """
        try:
            # RRCフィルタの設計
            rrc_taps = signal.firwin2(num_taps, 
                                     [0, 1.0/(2*self.samples_per_symbol), 1.0/self.samples_per_symbol, 1.0],
                                     [1, 1, 0, 0], 
                                     window=('kaiser', 5.0))
            
            # フィルタリング
            i_filtered = signal.filtfilt(rrc_taps, [1.0], i_samples)
            q_filtered = signal.filtfilt(rrc_taps, [1.0], q_samples)
            
            logger.info(f"Applied RRC filter with {num_taps} taps and alpha={alpha}")
            return i_filtered, q_filtered
            
        except Exception as e:
            logger.error(f"RRC filtering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return i_samples, q_samples  # エラー時は元の信号を返す
    
    def normalize_signal(self, i_samples, q_samples):
        """
        信号の正規化
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
        Returns:
            tuple: (正規化されたI信号, 正規化されたQ信号)
        """
        try:
            # I/Q信号の結合
            complex_signal = i_samples + 1j * q_samples
            
            # 信号強度の計算
            signal_power = np.mean(np.abs(complex_signal)**2)
            
            # 正規化
            if signal_power > 0:
                scale_factor = 1.0 / np.sqrt(signal_power)
                i_normalized = i_samples * scale_factor
                q_normalized = q_samples * scale_factor
                logger.info(f"Normalized signal. Original power: {10*np.log10(signal_power):.2f} dB")
                return i_normalized, q_normalized
            else:
                logger.warning("Signal power is zero, cannot normalize")
                return i_samples, q_samples
                
        except Exception as e:
            logger.error(f"Signal normalization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return i_samples, q_samples  # エラー時は元の信号を返す
    
    def detect_frame_start(self, i_samples, q_samples, sync_pattern=None):
        """
        フレーム開始位置の検出
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
            sync_pattern (np.ndarray): 同期パターン
        Returns:
            int: 検出された開始位置
        """
        try:
            # デフォルトの同期パターン（CCSDS同期マーカー 0x1ACFFC1D に対応）
            if sync_pattern is None:
                sync_bits = np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8))
                sync_pattern = 2 * sync_bits - 1  # BPSKシンボルに変換（0→-1, 1→1）
            
            # 相関計算のための信号準備
            complex_signal = i_samples + 1j * q_samples
            signal_mag = np.abs(complex_signal)
            
            # ダウンサンプリング（シンボルレートに合わせる）
            symbol_indices = np.arange(0, len(i_samples), self.samples_per_symbol)
            symbol_samples = i_samples[symbol_indices]
            
            # 硬判定（-1または1）
            hard_symbols = np.sign(symbol_samples)
            
            # 同期パターンで相関を計算
            corr = np.correlate(hard_symbols, sync_pattern, mode='valid')
            
            # 最大相関位置を検出
            max_corr_idx = np.argmax(np.abs(corr))
            max_corr_val = corr[max_corr_idx]
            
            # 元のサンプルインデックスに変換
            start_position = max_corr_idx * self.samples_per_symbol
            
            logger.info(f"Frame start detected at sample {start_position} (correlation: {max_corr_val})")
            return start_position
            
        except Exception as e:
            logger.error(f"Frame start detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0  # エラー時は先頭を返す

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

def plot_spectrum(i_samples, q_samples, fs, title):
    """スペクトルをプロット"""
    try:
        # 複素信号の作成
        complex_signal = i_samples + 1j * q_samples
        
        # FFTの計算
        n = len(complex_signal)
        spectrum = np.fft.fftshift(np.fft.fft(complex_signal) / n)
        f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
        
        # パワースペクトル密度の計算
        psd = 20 * np.log10(np.abs(spectrum) + 1e-12)
        
        # プロット
        plt.figure(figsize=(12, 6))
        plt.plot(f / 1e6, psd)
        plt.title(title)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.xlim(-fs/2e6, fs/2e6)
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error plotting spectrum: {e}")
        return None

def plot_constellation(i_samples, q_samples, title, max_points=10000):
    """コンスタレーションをプロット"""
    try:
        # サンプル数の制限
        n_samples = min(len(i_samples), max_points)
        i_subset = i_samples[:n_samples]
        q_subset = q_samples[:n_samples]
        
        # プロット
        plt.figure(figsize=(8, 8))
        plt.scatter(i_subset, q_subset, s=1, alpha=0.5)
        plt.title(title)
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        max_val = max(np.max(np.abs(i_subset)), np.max(np.abs(q_subset)))
        plt.xlim(-max_val*1.2, max_val*1.2)
        plt.ylim(-max_val*1.2, max_val*1.2)
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error plotting constellation: {e}")
        return None

class CCSPipeline:
    """CCSDS通信パイプラインクラス"""
    
    def __init__(self, interleave_depth=5, sample_rate=10e6, symbol_rate=1e6):
        """初期化"""
        self.frame_generator = CCSDSFrameGenerator()
        self.rs_processor = ReedSolomonProcessor(interleave_depth)
        self.interleave_processor = InterleaveProcessor(self.rs_processor)
        self.conv_encoder = ConvolutionalEncoder()
        self.viterbi_decoder = ViterbiDecoder()
        self.bpsk_modulator = BPSKModulator(sample_rate, symbol_rate)
        self.bpsk_demodulator = BPSKDemodulator(sample_rate, symbol_rate)
        logger.info("CCSPipeline initialized")
    
    def process_transmit(self, data):
        """送信側処理（フレーム生成→リードソロモン符号化→インターリーブ→畳み込み符号化→BPSK変調）"""
        try:
            # CADUフレームの生成
            cadu_frame = self.frame_generator.generate_cadu_frame(data)
            if cadu_frame is None:
                logger.error("CADU frame generation failed")
                return None, None, None, None
            
            # リードソロモン符号化
            rs_encoded_data = self.rs_processor.encode(cadu_frame)
            if rs_encoded_data is None:
                logger.error("Reed-Solomon encoding failed")
                return None, None, None, None
            
            # インターリーブ
            interleaved_data = self.interleave_processor.interleave(rs_encoded_data)
            if interleaved_data is None:
                logger.error("Interleaving failed")
                return None, None, None, None
            
            # 畳み込み符号化
            conv_encoded_data = self.conv_encoder.encode(interleaved_data)
            if conv_encoded_data is None:
                logger.error("Convolutional encoding failed")
                return None, None, None, None
            
            # BPSK変調
            i_samples, q_samples = self.bpsk_modulator.modulate(conv_encoded_data)
            if i_samples is None or q_samples is None:
                logger.error("BPSK modulation failed")
                return None, None, None, None
            
            logger.info(f"Transmit processing completed: {len(data)} bytes -> {len(i_samples)} samples")
            return cadu_frame, conv_encoded_data, i_samples, q_samples
            
        except Exception as e:
            logger.error(f"Transmit processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None
    
    def process_receive(self, i_samples, q_samples, interleaved_size=None):
        """受信側処理（BPSK復調→ビタビ復号→デインターリーブ→リードソロモン復号）"""
        try:
            # BPSK復調
            demodulated_data = self.bpsk_demodulator.demodulate(i_samples, q_samples)
            if demodulated_data is None:
                logger.error("BPSK demodulation failed")
                return None
            
            # ビタビ復号
            viterbi_decoded_data = self.viterbi_decoder.decode(demodulated_data, interleaved_size)
            if viterbi_decoded_data is None:
                logger.error("Viterbi decoding failed")
                return None
            
            # デインターリーブ
            deinterleaved_data = self.interleave_processor.deinterleave(viterbi_decoded_data)
            if deinterleaved_data is None:
                logger.error("Deinterleaving failed")
                return None
            
            # リードソロモン復号
            rs_decoded_data = self.rs_processor.decode(deinterleaved_data)
            if rs_decoded_data is None:
                logger.error("Reed-Solomon decoding failed")
                return None
            
            logger.info(f"Receive processing completed: {len(i_samples)} samples -> {len(rs_decoded_data)} bytes")
            return rs_decoded_data
            
        except Exception as e:
            logger.error(f"Receive processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def compare_data(original, processed):
    """データの比較"""
    try:
        if len(original) != len(processed):
            logger.error(f"Data size mismatch: original={len(original)} bytes, processed={len(processed)} bytes")
            
            # どちらが長い/短いかを判断
            if len(original) > len(processed):
                logger.error(f"Processed data is {len(original) - len(processed)} bytes shorter")
                # 比較可能な長さまでトリミング
                original = original[:len(processed)]
            else:
                logger.error(f"Processed data is {len(processed) - len(original)} bytes longer")
                # 比較可能な長さまでトリミング
                processed = processed[:len(original)]
            
            logger.warning("Comparing truncated data for diagnostic purposes")
        
        # ビット単位での比較
        match_count = sum(1 for a, b in zip(original, processed) if a == b)
        match_percentage = (match_count / len(original)) * 100
        
        # バイト単位での詳細分析（最初の数バイトのみ）
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

def sdr_loopback_test():
    """PLUTO SDRを使用したループバックテスト"""
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # パラメータ設定
        sample_rate = 2e6  # サンプリングレート [Hz]
        symbol_rate = 100e3  # シンボルレート [bps]
        center_freq = 1.0e9  # 中心周波数 [Hz]
        tx_gain = -10  # 送信ゲイン（低めに設定）
        rx_gain = 30  # 受信ゲイン
        interleave_depth = 5  # インターリーブの深さ
        
        # 通信パイプラインの初期化
        pipeline = CCSPipeline(interleave_depth=interleave_depth, sample_rate=sample_rate, symbol_rate=symbol_rate)
        
        # パラメータの取得
        rs_block_size = pipeline.rs_processor.rs_k
        interleave_depth = pipeline.rs_processor.interleave_depth
        
        # テストデータの生成
        # リードソロモンのブロックサイズに合わせる
        required_data_size = rs_block_size * interleave_depth - 12  # CADUヘッダ分を考慮
        test_data = bytearray([i % 256 for i in range(required_data_size)])  # 異なるパターンのデータ
        logger.info(f"Generated test data: {len(test_data)} bytes")
        
        # テストデータの保存
        test_data_path = os.path.join(timestamp_dir, "test_data.bin")
        with open(test_data_path, 'wb') as f:
            f.write(test_data)
        logger.info(f"Saved test data to {test_data_path}")
        
        # 送信側処理
        logger.info("=== Starting transmit processing ===")
        cadu_frame, conv_encoded_data, i_samples, q_samples = pipeline.process_transmit(test_data)
        if cadu_frame is None or conv_encoded_data is None or i_samples is None or q_samples is None:
            logger.error("Transmit processing failed")
            return 1
        
        # CADUフレームの保存
        cadu_frame_path = os.path.join(timestamp_dir, "cadu_frame.bin")
        with open(cadu_frame_path, 'wb') as f:
            f.write(cadu_frame)
        logger.info(f"Saved CADU frame to {cadu_frame_path}")
        
        # 畳み込み符号化データの保存
        conv_encoded_path = os.path.join(timestamp_dir, "conv_encoded_data.bin")
        with open(conv_encoded_path, 'wb') as f:
            f.write(conv_encoded_data)
        logger.info(f"Saved convolutional encoded data to {conv_encoded_path}")
        
        # 送信信号の処理（フィルタリングなど）
        signal_processor = SignalProcessor(sample_rate, symbol_rate)
        i_filtered, q_filtered = signal_processor.apply_rrc_filter(i_samples, q_samples)
        
        # 変調信号のプロット
        fig = plot_signals(i_filtered, q_filtered, "BPSK Modulated Signal (Filtered)")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "modulated_signal.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved plot to {plot_path}")
        
        # スペクトルのプロット
        fig = plot_spectrum(i_filtered, q_filtered, sample_rate, "BPSK Spectrum")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "tx_spectrum.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved spectrum plot to {plot_path}")
        
        # PLUTO SDRの初期化
        logger.info("=== Initializing PLUTO SDR ===")
        pluto = PlutoSDRInterface(
            rx_freq=center_freq, 
            tx_freq=center_freq, 
            rx_rate=sample_rate, 
            tx_rate=sample_rate,
            tx_gain=tx_gain,
            rx_gain=rx_gain
        )
        
        # 信号の送信
        logger.info("=== Transmitting signal ===")
        pluto.transmit(i_filtered, q_filtered, cyclic=True)
        
        # 一定時間待機
        time.sleep(1.0)
        
        # 信号の受信
        logger.info("=== Receiving signal ===")
        rx_i, rx_q = pluto.receive(num_samples=len(i_filtered) * 2)  # 余裕を持って受信
        
        # 信号の可視化
        fig = plot_signals(rx_i, rx_q, "Received Signal", max_samples=1000)
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "received_signal.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved received signal plot to {plot_path}")
        
        # コンスタレーションのプロット
        fig = plot_constellation(rx_i, rx_q, "Received Constellation")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "rx_constellation.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved constellation plot to {plot_path}")
        
        # 受信信号の処理
        rx_i_normalized, rx_q_normalized = signal_processor.normalize_signal(rx_i, rx_q)
        
        # フレーム開始位置の検出
        start_pos = signal_processor.detect_frame_start(rx_i_normalized, rx_q_normalized)
        
        # フレーム開始位置からの信号抽出
        rx_i_frame = rx_i_normalized[start_pos:start_pos + len(i_filtered)]
        rx_q_frame = rx_q_normalized[start_pos:start_pos + len(q_filtered)]
        
        # 信号の可視化（フレーム抽出後）
        fig = plot_signals(rx_i_frame, rx_q_frame, "Received Signal (Frame Detected)", max_samples=1000)
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "received_signal_frame.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved frame-detected signal plot to {plot_path}")
        
        # インターリーブ後のデータサイズを記録（ビタビ復号のパラメータとして使用）
        interleaved_size = pipeline.rs_processor.rs_n * interleave_depth
        
        # 受信側処理
        logger.info("=== Starting receive processing ===")
        received_data = pipeline.process_receive(rx_i_frame, rx_q_frame, interleaved_size)
        if received_data is None:
            logger.error("Receive processing failed")
            return 1
        
        # 受信データの保存
        received_path = os.path.join(timestamp_dir, "received_data.bin")
        with open(received_path, 'wb') as f:
            f.write(received_data)
        logger.info(f"Saved received data to {received_path}")
        
        # 元のCADUフレームと受信データの比較
        if not compare_data(cadu_frame, received_data):
            logger.error("Data comparison failed")
            return 1
        
        # SDR接続を閉じる
        pluto.close()
        
        logger.info("\nSDR loopback test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred in SDR loopback test: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """メイン処理"""
    return sdr_loopback_test()

if __name__ == '__main__':
    import sys
    sys.exit(main())