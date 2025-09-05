#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDS通信パイプラインの完全実装 (QPSK版):
CADU生成→リードソロモン符号化→インターリーブ→畳み込み符号化→QPSK変調→PLUTO SDR送信→PLUTO SDR受信→QPSK復調→ビタビ復号→デインターリーブ→リードソロモン復号
CCSDS 131.0-B-3に準拠、20 Mbpsのデータレートで20 MHz帯域幅
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
import threading
import pickle
from tqdm import tqdm  # 進捗バーのためのライブラリ

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ccsds_pipeline.log"),
        logging.StreamHandler()
    ]
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
    
    def generate_cadu_frame(self, data, counter=None):
        """CADUフレームの生成"""
        try:
            # カウンターの指定がある場合はそれを使用
            if counter is not None:
                frame_counter = counter
            else:
                frame_counter = self.frame_counter
                self.frame_counter += 1
                
            # 同期マーカー + VCDUヘッダー + データ
            vcdu_header = self.generate_vcdu_header(frame_counter)
            frame = self.sync_marker + vcdu_header + data
            
            logger.info(f"Generated CADU frame: {len(frame)} bytes, Counter: {frame_counter}")
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
    
    def get_counter_from_frame(self, frame):
        """フレームからカウンター値を取得"""
        try:
            if len(frame) < 12:  # 同期マーカー(4) + ヘッダー(8)の最小長
                return None
            # カウンターは8バイト目から4バイト
            counter_bytes = frame[8:12]
            counter = int.from_bytes(counter_bytes, 'big')
            return counter
        except Exception as e:
            logger.error(f"Error extracting counter from frame: {e}")
            return None

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
                    logger.debug(f"RS encoding successful for block {i+1}")
                except Exception as e:
                    logger.error(f"RS encoding failed for block {i+1}: {e}")
                    return None
            
            # 符号化されたブロックを結合
            encoded_data = bytearray()
            for block in encoded_blocks:
                encoded_data.extend(block)
            
            logger.debug(f"RS encoding completed: {len(data)} bytes -> {len(encoded_data)} bytes")
            return encoded_data
            
        except Exception as e:
            logger.error(f"RS encoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def decode(self, data, correct_errors=True):
        """
        リードソロモン復号
        
        Args:
            data: 復号するデータ
            correct_errors: エラー訂正を行うかどうか。Falseの場合はエラー検出のみ行う
            
        Returns:
            復号されたデータ、または検出されたエラー数（correct_errors=Falseの場合）
        """
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
            
            # エラー検出のみモードの場合
            if not correct_errors:
                total_errors = 0
                blocks_with_errors = 0
                
                for i, block in enumerate(blocks):
                    try:
                        # bytearrayをbytesに変換
                        block_bytes = bytes(block)
                        # エラー検出
                        _, errors = self.rs_codec.decode(block_bytes, nostrip=True)
                        if errors:
                            blocks_with_errors += 1
                            total_errors += len(errors) if isinstance(errors, list) else errors
                    except rs.ReedSolomonError:
                        # 訂正不可能なエラー
                        blocks_with_errors += 1
                        total_errors += self.rs_t + 1  # 最大訂正能力を超えている
                
                return {"total_errors": total_errors, "blocks_with_errors": blocks_with_errors}
            
            # 各ブロックを復号（エラー訂正モード）
            decoded_blocks = []
            error_stats = {"total_errors": 0, "blocks_with_errors": 0, "uncorrectable": 0}
            
            for i, block in enumerate(blocks):
                try:
                    # bytearrayをbytesに変換してからdecodeを呼び出す
                    block_bytes = bytes(block)
                    
                    try:
                        decoded_result = self.rs_codec.decode(block_bytes)
                        
                        # 復号結果の処理
                        if isinstance(decoded_result, tuple):
                            decoded_block = decoded_result[0]
                            # エラー訂正情報があれば統計を更新
                            if len(decoded_result) > 1:
                                error_info = decoded_result[1]
                                if isinstance(error_info, list) and error_info:
                                    logger.debug(f"Block {i+1}: Corrected {len(error_info)} errors")
                                    error_stats["total_errors"] += len(error_info)
                                    error_stats["blocks_with_errors"] += 1
                                elif isinstance(error_info, int) and error_info > 0:
                                    logger.debug(f"Block {i+1}: Corrected {error_info} errors")
                                    error_stats["total_errors"] += error_info
                                    error_stats["blocks_with_errors"] += 1
                        else:
                            decoded_block = decoded_result
                        
                        # 情報部分のみを取得（パリティ部分を除く）
                        decoded_block = decoded_block[:self.rs_k]
                        decoded_blocks.append(decoded_block)
                    
                    except rs.ReedSolomonError as rse:
                        logger.warning(f"Block {i+1}: Uncorrectable errors: {str(rse)}")
                        error_stats["uncorrectable"] += 1
                        # 訂正できない場合は元のデータ（パリティを除く）を使用
                        decoded_blocks.append(block[:self.rs_k])
                        
                except Exception as e:
                    logger.error(f"RS decoding failed for block {i+1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            
            # 復号されたブロックを結合
            decoded_data = bytearray()
            for block in decoded_blocks:
                decoded_data.extend(block)
            
            logger.debug(f"RS decoding completed: {len(data)} bytes -> {len(decoded_data)} bytes")
            logger.debug(f"RS error stats: {error_stats}")
            
            # 復号データとエラー統計を返す
            return {"data": decoded_data, "error_stats": error_stats}
            
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
            
            logger.debug(f"Interleaving completed: {len(data)} bytes -> {len(interleaved_data)} bytes")
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
            
            logger.debug(f"Deinterleaving completed: {len(data)} bytes -> {len(deinterleaved_data)} bytes")
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
            logger.debug(f"Input data: {len(data)} bytes, {len(bits)} bits")
            
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
            
            logger.debug(f"Convolutional encoding completed: {len(data)} bytes -> {len(encoded_data)} bytes")
            logger.debug(f"Input bits: {len(bits)}, Output bits: {len(encoded_bits)}")
            logger.debug(f"Termination bits: {len(termination_bits)} bits")
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
            logger.debug(f"Viterbi input size: {len(encoded_data)} bytes, {len(encoded_bits)} bits")
            logger.debug(f"Symbol pairs: {num_symbols}")
            logger.debug(f"Expected original bits: {original_bit_length}")
            
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
            logger.debug(f"Forward path completed in {forward_duration:.3f} seconds")
            
            # トレースバック（後方パス）
            # 最小メトリックの状態から開始
            current_state = np.argmin(path_metrics)
            logger.debug(f"Final state with minimum metric: {current_state}")
            
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
            logger.debug(f"Target output size: {target_size} bytes")
            
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
            logger.debug(f"Viterbi decoding completed: {len(encoded_data)} bytes -> {len(decoded_data)} bytes in {total_duration:.3f} seconds")
            logger.debug(f"Number of decoded bits: {len(decoded_bits)} -> trimmed to {needed_bits}")
            
            # メトリックも返す（エラーレートの推定に使用可能）
            min_metric = np.min(path_metrics)
            return {"data": decoded_data, "metric": min_metric}
            
        except Exception as e:
            logger.error(f"Viterbi decoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class QPSKModulator:
    """QPSK変調器クラス"""
    
    def __init__(self, sample_rate=40e6, symbol_rate=10e6):
        """
        初期化
        Args:
            sample_rate (float): サンプリングレート [Hz]
            symbol_rate (float): シンボルレート [bps]
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        logger.info(f"QPSKModulator initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Symbol rate: {symbol_rate} bps")
        logger.info(f"  Samples per symbol: {self.samples_per_symbol}")
    
    def modulate(self, data):
        """
        QPSK変調を実行
        Args:
            data (bytearray): 入力データ
        Returns:
            tuple: (I信号, Q信号)
        """
        try:
            # バイト列をビット列に変換（MSB first）
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            logger.debug(f"Input data: {len(data)} bytes, {len(bits)} bits")
            
            # デバッグ: 最初の数ビットを表示
            logger.debug(f"First 16 bits: {bits[:16]}")
            
            # ビット列をI/Qに割り当て（偶数ビット→I, 奇数ビット→Q）
            # パディングが必要な場合（ビット数が奇数の場合）
            if len(bits) % 2 != 0:
                bits = np.append(bits, 0)
                logger.debug("Added one padding bit for QPSK modulation")
                
            # I/Qビットを分離
            i_bits = bits[0::2]  # 偶数位置のビット
            q_bits = bits[1::2]  # 奇数位置のビット
            
            # 各ビットをシンボルに変換（0→-1, 1→1）
            i_symbols = 2 * i_bits.astype(np.float32) - 1
            q_symbols = 2 * q_bits.astype(np.float32) - 1
            
            # デバッグ: 最初の数シンボルを表示
            logger.debug(f"First I symbols: {i_symbols[:8]}")
            logger.debug(f"First Q symbols: {q_symbols[:8]}")
            
            # シンボルをサンプルに変換（シンボルレートからサンプリングレートに）
            i_samples = np.repeat(i_symbols, self.samples_per_symbol)
            q_samples = np.repeat(q_symbols, self.samples_per_symbol)
            
            logger.debug(f"QPSK modulation completed: {len(data)} bytes -> {len(i_samples)} samples")
            return i_samples, q_samples
            
        except Exception as e:
            logger.error(f"QPSK modulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

class QPSKDemodulator:
    """QPSK復調器クラス"""
    
    def __init__(self, sample_rate=40e6, symbol_rate=10e6):
        """
        初期化
        Args:
            sample_rate (float): サンプリングレート [Hz]
            symbol_rate (float): シンボルレート [bps]
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        logger.info(f"QPSKDemodulator initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Symbol rate: {symbol_rate} bps")
        logger.info(f"  Samples per symbol: {self.samples_per_symbol}")
    
    def demodulate(self, i_samples, q_samples):
        """
        QPSK復調を実行
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
        Returns:
            bytearray: 復調されたデータ
        """
        try:
            # シンボル判定（I信号とQ信号の符号で判定）
            # シンボル中央のサンプルを使用
            symbol_indices = np.arange(self.samples_per_symbol//2, len(i_samples), self.samples_per_symbol)
            if len(symbol_indices) == 0:
                logger.error("No symbol indices found for demodulation")
                return None
                
            i_symbol_samples = i_samples[symbol_indices]
            q_symbol_samples = q_samples[symbol_indices]
            
            # デバッグ: シンボルサンプルの最初の数値を表示
            logger.debug(f"First I symbol samples: {i_symbol_samples[:8]}")
            logger.debug(f"First Q symbol samples: {q_symbol_samples[:8]}")
            
            # シンボル判定: 0未満なら0、それ以外なら1
            i_bits = np.where(i_symbol_samples < 0, 0, 1).astype(np.uint8)
            q_bits = np.where(q_symbol_samples < 0, 0, 1).astype(np.uint8)
            
            # デバッグ: 判定後のビットの最初の数値を表示
            logger.debug(f"First I bits after decision: {i_bits[:8]}")
            logger.debug(f"First Q bits after decision: {q_bits[:8]}")
            
            # I/Qビットを交互に配置して一つのビット列に戻す
            bits = np.empty(i_bits.size + q_bits.size, dtype=np.uint8)
            bits[0::2] = i_bits  # 偶数位置にIビット
            bits[1::2] = q_bits  # 奇数位置にQビット
            
            # ビット列をバイト列に変換（MSB first）
            # packbitsはuint8型配列を期待する
            bytes_needed = (len(bits) + 7) // 8  # 必要なバイト数（切り上げ）
            bit_padding = bytes_needed * 8 - len(bits)  # 足りないビット数
            
            if bit_padding > 0:
                # 8の倍数になるようにパディング
                padded_bits = np.concatenate([bits, np.zeros(bit_padding, dtype=np.uint8)])
                logger.debug(f"Added {bit_padding} padding bits for byte alignment")
            else:
                padded_bits = bits
                
            # パッキングしてバイト列に変換
            decoded_data = np.packbits(padded_bits).tobytes()
            
            logger.debug(f"QPSK demodulation completed: {len(i_samples)} samples -> {len(decoded_data)} bytes")
            return decoded_data
            
        except Exception as e:
            logger.error(f"QPSK demodulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class PlutoSDRInterface:
    """PLUTO SDRインターフェースクラス"""
    
    def __init__(self, rx_freq=2.4e9, tx_freq=2.4e9, rx_rate=40e6, tx_rate=40e6, tx_gain=-10, rx_gain=30, tx_bandwidth=20e6, rx_bandwidth=20e6):
        """
        PLUTO SDRの初期化
        Args:
            rx_freq (float): 受信周波数 [Hz]
            tx_freq (float): 送信周波数 [Hz]
            rx_rate (float): 受信サンプリングレート [Hz]
            tx_rate (float): 送信サンプリングレート [Hz]
            tx_gain (int): 送信ゲイン [dB]
            rx_gain (int): 受信ゲイン [dB]
            tx_bandwidth (float): 送信帯域幅 [Hz]
            rx_bandwidth (float): 受信帯域幅 [Hz]
        """
        try:
            # PLUTO SDRに接続
            self.sdr = adi.Pluto()
            
            # 送信設定
            self.sdr.tx.frequency = int(tx_freq)
            self.sdr.tx.sampling_frequency = int(tx_rate)
            self.sdr.tx.gain_control_mode = "manual"
            self.sdr.tx.hardwaregain = tx_gain
            self.sdr.tx.bandwidth = int(tx_bandwidth)
            
            # 受信設定
            self.sdr.rx.frequency = int(rx_freq)
            self.sdr.rx.sampling_frequency = int(rx_rate)
            self.sdr.rx.gain_control_mode = "manual"
            self.sdr.rx.gain = rx_gain
            self.sdr.rx.bandwidth = int(rx_bandwidth)
            
            # バッファサイズ設定
            self.sdr.rx_buffer_size = 2**18  # 受信バッファを大きめに設定
            
            # 設定の保存
            self.rx_rate = rx_rate
            self.tx_rate = tx_rate
            
            logger.info(f"PLUTO SDR initialized:")
            logger.info(f"  TX frequency: {tx_freq/1e6} MHz")
            logger.info(f"  RX frequency: {rx_freq/1e6} MHz")
            logger.info(f"  TX rate: {tx_rate/1e6} Msps")
            logger.info(f"  RX rate: {rx_rate/1e6} Msps")
            logger.info(f"  TX bandwidth: {tx_bandwidth/1e6} MHz")
            logger.info(f"  RX bandwidth: {rx_bandwidth/1e6} MHz")
            logger.info(f"  TX gain: {tx_gain} dB")
            logger.info(f"  RX gain: {rx_gain} dB")
            
        except Exception as e:
            logger.error(f"PLUTO SDR initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def transmit(self, i_samples, q_samples, cyclic=False):
        """
        IQ信号をPLUTO SDRで送信
        Args:
            i_samples (np.ndarray): I信号サンプル
            q_samples (np.ndarray): Q信号サンプル
            cyclic (bool): 繰り返し送信するかどうか
        """
        try:
            # 送信データの準備
            length = min(len(i_samples), len(q_samples))
            tx_data = i_samples[:length] + 1j * q_samples[:length]
            
            # 正規化（クリッピングを防ぐため）
            max_magnitude = np.max(np.abs(tx_data))
            if max_magnitude > 0:
                tx_data = tx_data * 0.8 / max_magnitude  # 0.8を掛けて余裕を持たせる
            
            # サンプルの先頭と末尾にランプを適用（急激な変化を避ける）
            ramp_length = min(1000, length // 10)
            ramp_up = np.linspace(0, 1, ramp_length)
            ramp_down = np.linspace(1, 0, ramp_length)
            
            # ランプの適用
            tx_data[:ramp_length] *= ramp_up
            tx_data[-ramp_length:] *= ramp_down
            
            # 繰り返し送信モードの設定
            self.sdr.tx_cyclic_buffer = cyclic
            
            # 送信
            logger.info(f"Transmitting {length} samples...")
            self.sdr.tx(tx_data)
            
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
    
    def __init__(self, sample_rate=40e6, symbol_rate=10e6):
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
                
                # QPSKの場合、偶数ビットがI、奇数ビットがQに対応
                sync_i_bits = sync_bits[0::2]
                sync_q_bits = sync_bits[1::2]
                
                # BPSKシンボルに変換（0→-1, 1→1）
                sync_i = 2 * sync_i_bits - 1
                sync_q = 2 * sync_q_bits - 1
            else:
                # 既に用意された同期パターンを使用
                sync_i = sync_pattern[0]
                sync_q = sync_pattern[1]
            
            # 相関計算のための信号準備
            symbol_indices = np.arange(0, len(i_samples), self.samples_per_symbol)
            i_symbols = i_samples[symbol_indices]
            q_symbols = q_samples[symbol_indices]
            
            # 硬判定（-1または1）
            i_hard = np.sign(i_symbols)
            q_hard = np.sign(q_symbols)
            
            # I/Q両方の相関を計算し、合計
            i_corr = np.correlate(i_hard, sync_i, mode='valid')
            q_corr = np.correlate(q_hard, sync_q, mode='valid')
            total_corr = i_corr + q_corr
            
            # 最大相関位置を検出
            max_corr_idx = np.argmax(np.abs(total_corr))
            max_corr_val = total_corr[max_corr_idx]
            
            # 元のサンプルインデックスに変換
            start_position = max_corr_idx * self.samples_per_symbol
            
            logger.info(f"Frame start detected at sample {start_position} (correlation: {max_corr_val})")
            return start_position
            
        except Exception as e:
            logger.error(f"Frame start detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0  # エラー時は先頭を返す
    
    def correlate_with_preamble(self, i_samples, q_samples, i_preamble, q_preamble, threshold=0.7):
        """
        プリアンブルとの相関を計算し、フレーム開始位置とその信頼度を検出
        
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
            i_preamble (np.ndarray): Iプリアンブルシーケンス
            q_preamble (np.ndarray): Qプリアンブルシーケンス
            threshold (float): 検出閾値（0-1の範囲）
            
        Returns:
            dict: 検出結果（位置と相関値のリスト）
        """
        try:
            # 相関長さの確認
            if len(i_preamble) * self.samples_per_symbol > len(i_samples) or len(q_preamble) * self.samples_per_symbol > len(q_samples):
                logger.error("Preamble sequence is longer than input signal")
                return {"positions": [], "correlations": []}
            
            # シンボルレートにダウンサンプリング
            symbol_indices = np.arange(0, len(i_samples), self.samples_per_symbol)
            i_symbols = i_samples[symbol_indices]
            q_symbols = q_samples[symbol_indices]
            
            # 硬判定（-1または1）
            i_hard = np.sign(i_symbols)
            q_hard = np.sign(q_symbols)
            
            # I/Q両方の相関を計算し、合計
            i_corr = np.correlate(i_hard, i_preamble, mode='valid')
            q_corr = np.correlate(q_hard, q_preamble, mode='valid')
            total_corr = np.abs(i_corr) + np.abs(q_corr)
            
            # 正規化（-1〜1の範囲に）
            total_corr_normalized = total_corr / (len(i_preamble) + len(q_preamble))
            
            # 閾値以上の相関ピークを検出
            peaks, _ = signal.find_peaks(total_corr_normalized, height=threshold)
            
            # 結果を整形
            positions = []
            correlations = []
            
            for peak in peaks:
                # 元のサンプルインデックスに変換
                sample_pos = peak * self.samples_per_symbol
                positions.append(sample_pos)
                correlations.append(total_corr_normalized[peak])
            
            logger.info(f"Detected {len(positions)} potential frame starts above threshold {threshold}")
            for i, (pos, corr_val) in enumerate(zip(positions, correlations)):
                logger.info(f"  Frame {i+1}: position={pos}, correlation={corr_val:.3f}")
            
            return {
                "positions": positions,
                "correlations": correlations
            }
            
        except Exception as e:
            logger.error(f"Preamble correlation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"positions": [], "correlations": []}
    
    def apply_carrier_recovery(self, i_samples, q_samples):
        """
        キャリア周波数オフセット補正
        Args:
            i_samples (np.ndarray): I信号
            q_samples (np.ndarray): Q信号
        Returns:
            tuple: (補正されたI信号, 補正されたQ信号)
        """
        try:
            # 複素信号に変換
            complex_signal = i_samples + 1j * q_samples
            
            # QPSKの4乗スペクトルを計算（位相の曖昧性を除去）
            complex_signal_4th = complex_signal ** 4
            
            # スペクトル中の最大ピークを見つける
            N = len(complex_signal_4th)
            spectrum = np.fft.fft(complex_signal_4th, N)
            freq = np.fft.fftfreq(N, 1/self.sample_rate)
            
            # 中央付近のピークを検索（DCを除く）
            center_idx = N // 2
            window = max(N // 100, 10)  # 検索ウィンドウ（サンプル数の1%か10サンプル、大きい方）
            start_idx = center_idx - window
            end_idx = center_idx + window
            
            # 検索ウィンドウ内のピークを見つける
            peak_idx = np.argmax(np.abs(spectrum[start_idx:end_idx])) + start_idx
            peak_freq = freq[peak_idx]
            
            # 実際の周波数オフセットは4乗したものの1/4
            freq_offset = peak_freq / 4
            
            logger.info(f"Detected carrier frequency offset: {freq_offset:.2f} Hz")
            
            # 周波数オフセットを補正
            t = np.arange(len(complex_signal)) / self.sample_rate
            correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
            corrected_signal = complex_signal * correction
            
            # 複素信号をI/Qに分解
            i_corrected = np.real(corrected_signal)
            q_corrected = np.imag(corrected_signal)
            
            return i_corrected, q_corrected
            
        except Exception as e:
            logger.error(f"Carrier recovery failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return i_samples, q_samples  # エラー時は元の信号を返す

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

def plot_constellation(i_samples, q_samples, title, max_points=10000, downsample=True):
    """コンスタレーションをプロット"""
    try:
        # ダウンサンプリングオプション（シンボルポイントのみを表示）
        if downsample and hasattr(plot_constellation, 'samples_per_symbol') and plot_constellation.samples_per_symbol > 1:
            # シンボル中心のサンプルを抽出
            indices = np.arange(plot_constellation.samples_per_symbol // 2, len(i_samples), plot_constellation.samples_per_symbol)
            i_subset = i_samples[indices]
            q_subset = q_samples[indices]
        else:
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
        
        # QPSKの理想的な判定境界を表示
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        # QPSKの理想的なコンスタレーションポイントを表示
        ideal_points = np.array([1+0j, 0+1j, -1+0j, 0-1j]) * max_val * 0.7
        plt.scatter(np.real(ideal_points), np.imag(ideal_points), color='r', marker='x', s=100)
        
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error plotting constellation: {e}")
        return None

# コンスタレーションプロット用にサンプル数/シンボルを保存
plot_constellation.samples_per_symbol = 4  # デフォルト値

def plot_packet_stats(packet_results, title="Packet Reception Statistics"):
    """パケットの成功率などをプロット"""
    try:
        # データ整形
        packet_numbers = [result["packet_num"] for result in packet_results]
        success = [1 if result["success"] else 0 for result in packet_results]
        rs_errors = [result.get("rs_errors", 0) for result in packet_results]
        snr_values = [result.get("snr", 0) for result in packet_results]
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 成功/失敗プロット
        ax1.bar(packet_numbers, success, color='green', alpha=0.7, label='Success')
        ax1.bar(packet_numbers, [1-s for s in success], bottom=success, color='red', alpha=0.7, label='Failure')
        ax1.set_ylabel('Packet Status')
        ax1.set_title(f'{title} - {sum(success)}/{len(packet_numbers)} Packets Successfully Received ({sum(success)/len(packet_numbers)*100:.1f}%)')
        ax1.legend()
        ax1.grid(True)
        
        # SNRとエラープロット
        ax2.plot(packet_numbers, snr_values, 'bo-', label='SNR (dB)', alpha=0.7)
        ax2.set_xlabel('Packet Number')
        ax2.set_ylabel('SNR (dB)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True)
        
        ax3 = ax2.twinx()
        ax3.plot(packet_numbers, rs_errors, 'ro-', label='RS Errors', alpha=0.7)
        ax3.set_ylabel('RS Errors', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error plotting packet statistics: {e}")
        return None

class CCSPipeline:
    """CCSDS通信パイプラインクラス"""
    
    def __init__(self, interleave_depth=5, sample_rate=40e6, symbol_rate=10e6, modulation="qpsk"):
        """
        初期化
        Args:
            interleave_depth (int): インターリーブの深さ
            sample_rate (float): サンプリングレート [Hz]
            symbol_rate (float): シンボルレート [bps]
            modulation (str): 変調方式 ("bpsk" または "qpsk")
        """
        self.frame_generator = CCSDSFrameGenerator()
        self.rs_processor = ReedSolomonProcessor(interleave_depth)
        self.interleave_processor = InterleaveProcessor(self.rs_processor)
        self.conv_encoder = ConvolutionalEncoder()
        self.viterbi_decoder = ViterbiDecoder()
        
        # 変調方式に基づいてモジュレータとデモジュレータを選択
        self.modulation = modulation.lower()
        if self.modulation == "qpsk":
            self.modulator = QPSKModulator(sample_rate, symbol_rate)
            self.demodulator = QPSKDemodulator(sample_rate, symbol_rate)
            logger.info(f"Using QPSK modulation at {symbol_rate/1e6} Msps (data rate: {symbol_rate*2/1e6} Mbps)")
        else:
            # デフォルトはBPSK
            self.modulator = BPSKModulator(sample_rate, symbol_rate)
            self.demodulator = BPSKDemodulator(sample_rate, symbol_rate)
            logger.info(f"Using BPSK modulation at {symbol_rate/1e6} Msps (data rate: {symbol_rate/1e6} Mbps)")
            
        # コンスタレーションプロット用の設定を更新
        plot_constellation.samples_per_symbol = int(sample_rate / symbol_rate)
            
        logger.info("CCSPipeline initialized")
    
    def process_transmit(self, data, counter=None):
        """送信側処理（フレーム生成→リードソロモン符号化→インターリーブ→畳み込み符号化→変調）"""
        try:
            # CADUフレームの生成
            cadu_frame = self.frame_generator.generate_cadu_frame(data, counter)
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
            
            # 変調（BPSKまたはQPSK）
            i_samples, q_samples = self.modulator.modulate(conv_encoded_data)
            if i_samples is None or q_samples is None:
                logger.error(f"{self.modulation.upper()} modulation failed")
                return None, None, None, None
            
            logger.info(f"Transmit processing completed: {len(data)} bytes -> {len(i_samples)} samples")
            return cadu_frame, conv_encoded_data, i_samples, q_samples
            
        except Exception as e:
            logger.error(f"Transmit processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None
    
    def process_receive(self, i_samples, q_samples, interleaved_size=None):
        """受信側処理（復調→ビタビ復号→デインターリーブ→リードソロモン復号）"""
        try:
            # 復調（BPSKまたはQPSK）
            demodulated_data = self.demodulator.demodulate(i_samples, q_samples)
            if demodulated_data is None:
                logger.error(f"{self.modulation.upper()} demodulation failed")
                return None
            
            # ビタビ復号
            viterbi_result = self.viterbi_decoder.decode(demodulated_data, interleaved_size)
            if viterbi_result is None:
                logger.error("Viterbi decoding failed")
                return None
                
            viterbi_decoded_data = viterbi_result["data"]
            
            # デインターリーブ
            deinterleaved_data = self.interleave_processor.deinterleave(viterbi_decoded_data)
            if deinterleaved_data is None:
                logger.error("Deinterleaving failed")
                return None
            
            # リードソロモン復号
            rs_result = self.rs_processor.decode(deinterleaved_data)
            if rs_result is None:
                logger.error("Reed-Solomon decoding failed")
                return None
                
            # 復号データとエラー統計を取得
            received_data = rs_result["data"]
            error_stats = rs_result["error_stats"]
            
            logger.info(f"Receive processing completed: {len(i_samples)} samples -> {len(received_data)} bytes")
            logger.info(f"RS error stats: {error_stats}")
            
            return {
                "data": received_data,
                "rs_error_stats": error_stats,
                "viterbi_metric": viterbi_result["metric"]
            }
            
        except Exception as e:
            logger.error(f"Receive processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def calculate_snr(i_samples, q_samples):
    """信号対雑音比(SNR)を計算"""
    try:
        # 複素信号に変換
        signal = i_samples + 1j * q_samples
        
        # 信号電力の計算
        signal_power = np.mean(np.abs(signal)**2)
        
        # 雑音の推定
        # 方法1: 高周波成分を雑音として扱う
        fft_signal = np.fft.fft(signal)
        n = len(fft_signal)
        # 高周波50%を雑音として扱う
        fft_signal_center = np.fft.fftshift(fft_signal)
        noise_region = np.concatenate([fft_signal_center[:n//4], fft_signal_center[3*n//4:]])
        noise_power = np.mean(np.abs(noise_region)**2)
        
        # SNRの計算
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # 無限大の代わりに大きな値
            
        return snr
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        return 0

def compare_data(original, processed):
    """データの比較"""
    try:
        if len(original) != len(processed):
            logger.warning(f"Data size mismatch: original={len(original)} bytes, processed={len(processed)} bytes")
            
            # どちらが長い/短いかを判断
            if len(original) > len(processed):
                logger.warning(f"Processed data is {len(original) - len(processed)} bytes shorter")
                # 比較可能な長さまでトリミング
                original = original[:len(processed)]
            else:
                logger.warning(f"Processed data is {len(processed) - len(original)} bytes longer")
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
            logger.debug(f"Byte {i}: Original: {orig_bits} ({orig_byte:02x}), Processed: {proc_bits} ({proc_byte:02x})")
        
        if match_percentage < 100:
            logger.warning(f"Data mismatch: {match_percentage:.2f}% match")
            # 不一致位置を表示
            mismatch_positions = [(i, a, b) for i, (a, b) in enumerate(zip(original, processed)) if a != b]
            logger.debug(f"Total mismatches: {len(mismatch_positions)}")
            # 最初の10個の不一致位置を表示
            for pos, orig, proc in mismatch_positions[:10]:
                logger.debug(f"Position {pos}: original={orig:02x}, processed={proc:02x}")
            return False, match_percentage
        
        logger.info(f"Data match: 100% ({match_count}/{len(original)} bytes)")
        return True, 100.0
        
    except Exception as e:
        logger.error(f"Error comparing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0

def sdr_qpsk_test():
    """PLUTO SDRを使用したQPSKシングルパケットテスト"""
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # パラメータ設定
        sample_rate = 40e6  # サンプリングレート [Hz]
        symbol_rate = 10e6  # シンボルレート [bps] -> QPSKで20 Mbps
        center_freq = 5.84e9  # 中心周波数 [Hz]
        tx_gain = -15  # 送信ゲイン
        rx_gain = 30  # 受信ゲイン
        bandwidth = 20e6  # 帯域幅 [Hz]
        interleave_depth = 5  # インターリーブの深さ
        
        # 通信パイプラインの初期化（QPSK指定）
        pipeline = CCSPipeline(interleave_depth=interleave_depth, sample_rate=sample_rate, symbol_rate=symbol_rate, modulation="qpsk")
        
        # パラメータの取得
        rs_block_size = pipeline.rs_processor.rs_k
        interleave_depth = pipeline.rs_processor.interleave_depth
        
        # テストデータの生成
        # リードソロモンのブロックサイズに合わせる
        required_data_size = rs_block_size * interleave_depth - 12  # CADUヘッダ分を考慮
        # 様々なビットパターンを含むデータを生成
        test_data = bytearray([i % 256 for i in range(required_data_size)])
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
        fig = plot_signals(i_filtered, q_filtered, "QPSK Modulated Signal (Filtered)")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "modulated_signal.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved plot to {plot_path}")
        
        # スペクトルのプロット
        fig = plot_spectrum(i_filtered, q_filtered, sample_rate, "QPSK Spectrum")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "tx_spectrum.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved spectrum plot to {plot_path}")
        
        # コンスタレーションのプロット
        fig = plot_constellation(i_filtered, q_filtered, "QPSK Constellation")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "tx_constellation.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved constellation plot to {plot_path}")
        
        # PLUTO SDRの初期化
        logger.info("=== Initializing PLUTO SDR ===")
        pluto = PlutoSDRInterface(
            rx_freq=center_freq, 
            tx_freq=center_freq, 
            rx_rate=sample_rate, 
            tx_rate=sample_rate,
            tx_gain=tx_gain,
            rx_gain=rx_gain,
            tx_bandwidth=bandwidth,
            rx_bandwidth=bandwidth
        )
        
        # 信号の送信
        logger.info("=== Transmitting signal ===")
        pluto.transmit(i_filtered, q_filtered, cyclic=False)
        
        # 一定時間待機
        time.sleep(0.5)
        
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
        
        # 受信信号スペクトルのプロット
        fig = plot_spectrum(rx_i, rx_q, sample_rate, "Received QPSK Spectrum")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "rx_spectrum.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved received spectrum plot to {plot_path}")
        
        # コンスタレーションのプロット
        fig = plot_constellation(rx_i, rx_q, "Received QPSK Constellation")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "rx_constellation.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved constellation plot to {plot_path}")
        
        # 信号処理（正規化とキャリア回復）
        rx_i_normalized, rx_q_normalized = signal_processor.normalize_signal(rx_i, rx_q)
        rx_i_corrected, rx_q_corrected = signal_processor.apply_carrier_recovery(rx_i_normalized, rx_q_normalized)
        
        # 処理後コンスタレーションのプロット
        fig = plot_constellation(rx_i_corrected, rx_q_corrected, "Processed QPSK Constellation")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "rx_processed_constellation.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved processed constellation plot to {plot_path}")
        
        # フレーム開始位置の検出（QPSK用にプリアンブルを準備）
        sync_bits = np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8))
        sync_i = 2 * sync_bits[0::2] - 1  # 偶数ビット→I
        sync_q = 2 * sync_bits[1::2] - 1  # 奇数ビット→Q
        
        # プリアンブルとの相関
        corr_result = signal_processor.correlate_with_preamble(rx_i_corrected, rx_q_corrected, sync_i, sync_q)
        
        # フレーム開始位置が検出できなかった場合
        if not corr_result["positions"]:
            logger.error("No frame start detected")
            return 1
        
        # 最も相関の高い位置を使用
        best_idx = np.argmax(np.abs(corr_result["correlations"]))
        start_pos = corr_result["positions"][best_idx]
        
        # フレーム開始位置からの信号抽出
        max_samples = min(len(rx_i_corrected) - start_pos, len(i_filtered))
        rx_i_frame = rx_i_corrected[start_pos:start_pos + max_samples]
        rx_q_frame = rx_q_corrected[start_pos:start_pos + max_samples]
        
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
        receive_result = pipeline.process_receive(rx_i_frame, rx_q_frame, interleaved_size)
        if receive_result is None:
            logger.error("Receive processing failed")
            return 1
        
        received_data = receive_result["data"]
        rs_error_stats = receive_result["rs_error_stats"]
        viterbi_metric = receive_result["viterbi_metric"]
        
        # 受信データの保存
        received_path = os.path.join(timestamp_dir, "received_data.bin")
        with open(received_path, 'wb') as f:
            f.write(received_data)
        logger.info(f"Saved received data to {received_path}")
        
        # 元のCADUフレームと受信データの比較
        success, match_percentage = compare_data(cadu_frame, received_data)
        if not success:
            logger.warning(f"Data comparison result: {match_percentage:.2f}% match")
        else:
            logger.info(f"Data comparison successful: 100% match")
        
        # エラー統計の表示
        logger.info(f"Reed-Solomon error statistics:")
        logger.info(f"  Total errors: {rs_error_stats['total_errors']}")
        logger.info(f"  Blocks with errors: {rs_error_stats['blocks_with_errors']}/{interleave_depth}")
        logger.info(f"  Uncorrectable blocks: {rs_error_stats['uncorrectable']}")
        
        # SDR接続を閉じる
        pluto.close()
        
        logger.info("\nQPSK SDR test completed")
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error occurred in QPSK SDR test: {e}")
        import traceback
        traceback.print_exc()
        return 1

def sdr_qpsk_multi_packet_test(num_packets=100):
    """PLUTO SDRを使用したQPSK複数パケット送受信テスト"""
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # パラメータ設定
        sample_rate = 40e6  # サンプリングレート [Hz]
        symbol_rate = 10e6  # シンボルレート [bps] -> QPSKで20 Mbps
        center_freq = 2.4e9  # 中心周波数 [Hz]
        tx_gain = -15  # 送信ゲイン
        rx_gain = 30  # 受信ゲイン
        bandwidth = 20e6  # 帯域幅 [Hz]
        interleave_depth = 5  # インターリーブの深さ
        
        # 通信パイプラインの初期化
        def sdr_qpsk_multi_packet_test(num_packets=100):
    """PLUTO SDRを使用したQPSK複数パケット送受信テスト"""
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # パラメータ設定
        sample_rate = 40e6  # サンプリングレート [Hz]
        symbol_rate = 10e6  # シンボルレート [bps] -> QPSKで20 Mbps
        center_freq = 2.4e9  # 中心周波数 [Hz]
        tx_gain = -15  # 送信ゲイン
        rx_gain = 30  # 受信ゲイン
        bandwidth = 20e6  # 帯域幅 [Hz]
        interleave_depth = 5  # インターリーブの深さ
        
        # 通信パイプラインの初期化（QPSK指定）
        pipeline = CCSPipeline(interleave_depth=interleave_depth, sample_rate=sample_rate, symbol_rate=symbol_rate, modulation="qpsk")
        
        # パラメータの取得
        rs_block_size = pipeline.rs_processor.rs_k
        interleave_depth = pipeline.rs_processor.interleave_depth
        
        # PLUTO SDRの初期化
        logger.info("=== Initializing PLUTO SDR ===")
        pluto = PlutoSDRInterface(
            rx_freq=center_freq, 
            tx_freq=center_freq, 
            rx_rate=sample_rate, 
            tx_rate=sample_rate,
            tx_gain=tx_gain,
            rx_gain=rx_gain,
            tx_bandwidth=bandwidth,
            rx_bandwidth=bandwidth
        )
        
        # 信号処理クラスの初期化
        signal_processor = SignalProcessor(sample_rate, symbol_rate)
        
        # テスト結果の保存用配列
        results = []
        
        # パケット数のカウンター
        total_packets = num_packets
        successful_packets = 0
        
        # プリアンブルシーケンス（同期マーカーをQPSK変調したもの）の準備
        sync_bits = np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8))
        sync_i_bits = sync_bits[0::2]  # 偶数ビット→I
        sync_q_bits = sync_bits[1::2]  # 奇数ビット→Q
        sync_i = 2 * sync_i_bits - 1  # BPSKシンボルに変換（0→-1, 1→1）
        sync_q = 2 * sync_q_bits - 1  # BPSKシンボルに変換（0→-1, 1→1）
        
        logger.info(f"=== Starting {total_packets} packet QPSK test ===")
        
        try:
            # 各パケットの送受信を実行
            for packet_num in tqdm(range(total_packets), desc="Processing packets"):
                # テストデータの生成（パケット番号を埋め込む）
                required_data_size = rs_block_size * interleave_depth - 12  # CADUヘッダ分を考慮
                test_data = bytearray([(packet_num + i) % 256 for i in range(required_data_size)])  # パケット番号でデータを埋める
                
                # 送信側処理
                logger.info(f"--- Processing packet {packet_num+1}/{total_packets} ---")
                cadu_frame, conv_encoded_data, i_samples, q_samples = pipeline.process_transmit(test_data, packet_num)
                if cadu_frame is None or conv_encoded_data is None or i_samples is None or q_samples is None:
                    logger.error(f"Packet {packet_num+1}: Transmit processing failed")
                    results.append({
                        "packet_num": packet_num,
                        "success": False,
                        "error": "Transmit processing failed",
                    })
                    continue
                
                # 送信信号の処理（フィルタリングなど）
                i_filtered, q_filtered = signal_processor.apply_rrc_filter(i_samples, q_samples)
                
                # パケット間のガード時間（空白）を追加
                guard_samples = int(sample_rate * 0.0002)  # 200マイクロ秒のガード時間
                i_with_guard = np.concatenate([np.zeros(guard_samples), i_filtered, np.zeros(guard_samples)])
                q_with_guard = np.concatenate([np.zeros(guard_samples), q_filtered, np.zeros(guard_samples)])
                
                # 送信
                logger.info(f"Packet {packet_num+1}: Transmitting...")
                if not pluto.transmit(i_with_guard, q_with_guard, cyclic=False):
                    logger.error(f"Packet {packet_num+1}: Transmission failed")
                    results.append({
                        "packet_num": packet_num,
                        "success": False,
                        "error": "Transmission failed",
                    })
                    continue
                
                # 送信と受信の間に少し待機（バッファの処理時間）
                wait_time = len(i_with_guard) / sample_rate * 1.5  # 送信時間の1.5倍
                time.sleep(wait_time)
                
                # 受信（送信信号の長さの1.5倍のバッファで受信）
                logger.info(f"Packet {packet_num+1}: Receiving...")
                rx_i, rx_q = pluto.receive(num_samples=int(len(i_with_guard) * 1.5))
                if rx_i is None or rx_q is None:
                    logger.error(f"Packet {packet_num+1}: Reception failed")
                    results.append({
                        "packet_num": packet_num,
                        "success": False,
                        "error": "Reception failed",
                    })
                    continue
                
                # SNRの計算
                snr = calculate_snr(rx_i, rx_q)
                logger.info(f"Packet {packet_num+1}: Estimated SNR: {snr:.2f} dB")
                
                # 信号処理（正規化とキャリア回復）
                rx_i_normalized, rx_q_normalized = signal_processor.normalize_signal(rx_i, rx_q)
                rx_i_corrected, rx_q_corrected = signal_processor.apply_carrier_recovery(rx_i_normalized, rx_q_normalized)
                
                # プリアンブルとの相関を計算
                corr_result = signal_processor.correlate_with_preamble(rx_i_corrected, rx_q_corrected, sync_i, sync_q)
                
                # フレーム開始位置が検出できなかった場合
                if not corr_result["positions"]:
                    logger.error(f"Packet {packet_num+1}: No frame start detected")
                    results.append({
                        "packet_num": packet_num,
                        "success": False,
                        "error": "No frame start detected",
                        "snr": snr
                    })
                    continue
                
                # 最も相関の高い位置を使用
                best_idx = np.argmax(np.abs(corr_result["correlations"]))
                start_pos = corr_result["positions"][best_idx]
                
                # フレーム開始位置からの信号抽出
                max_samples = min(len(rx_i_corrected) - start_pos, len(i_filtered))
                rx_i_frame = rx_i_corrected[start_pos:start_pos + max_samples]
                rx_q_frame = rx_q_corrected[start_pos:start_pos + max_samples]
                
                # 10パケットごとにコンスタレーションプロットを保存
                if packet_num % 10 == 0:
                    fig = plot_constellation(rx_i_frame, rx_q_frame, f"Packet {packet_num+1} QPSK Constellation")
                    if fig is not None:
                        plot_path = os.path.join(timestamp_dir, f"rx_constellation_packet_{packet_num+1}.png")
                        fig.savefig(plot_path)
                        plt.close(fig)
                        logger.info(f"Saved constellation plot to {plot_path}")
                
                # インターリーブ後のデータサイズを記録（ビタビ復号のパラメータとして使用）
                interleaved_size = pipeline.rs_processor.rs_n * interleave_depth
                
                # 受信側処理
                logger.info(f"Packet {packet_num+1}: Decoding...")
                receive_result = pipeline.process_receive(rx_i_frame, rx_q_frame, interleaved_size)
                if receive_result is None:
                    logger.error(f"Packet {packet_num+1}: Receive processing failed")
                    results.append({
                        "packet_num": packet_num,
                        "success": False,
                        "error": "Receive processing failed",
                        "snr": snr
                    })
                    continue
                
                received_data = receive_result["data"]
                rs_error_stats = receive_result["rs_error_stats"]
                viterbi_metric = receive_result["viterbi_metric"]
                
                # カウンター値の確認
                received_counter = pipeline.frame_generator.get_counter_from_frame(received_data)
                if received_counter != packet_num:
                    logger.warning(f"Packet {packet_num+1}: Counter mismatch - expected {packet_num}, got {received_counter}")
                
                # 元のCADUフレームと受信データの比較
                success, match_percentage = compare_data(cadu_frame, received_data)
                if success:
                    logger.info(f"Packet {packet_num+1}: Success ({match_percentage:.1f}% match)")
                    successful_packets += 1
                else:
                    logger.warning(f"Packet {packet_num+1}: Data mismatch ({match_percentage:.1f}% match)")
                
                # 結果を保存
                results.append({
                    "packet_num": packet_num,
                    "success": success,
                    "match_percentage": match_percentage,
                    "rs_errors": rs_error_stats["total_errors"],
                    "rs_blocks_with_errors": rs_error_stats["blocks_with_errors"],
                    "rs_uncorrectable": rs_error_stats["uncorrectable"],
                    "viterbi_metric": viterbi_metric,
                    "snr": snr,
                    "counter": received_counter
                })
                
                # 20パケットごとに中間結果を保存
                if (packet_num + 1) % 20 == 0 or packet_num == total_packets - 1:
                    # 結果を保存
                    results_file = os.path.join(timestamp_dir, f"packet_results_{packet_num+1}.pkl")
                    with open(results_file, "wb") as f:
                        pickle.dump(results, f)
                    logger.info(f"Saved intermediate results to {results_file}")
                    
                    # 統計の表示
                    current_success_rate = successful_packets / (packet_num + 1) * 100
                    logger.info(f"Current statistics: {successful_packets}/{packet_num+1} packets successful ({current_success_rate:.1f}%)")
        
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        
        # SDR接続を閉じる
        pluto.close()
        
        # 最終結果の保存
        results_file = os.path.join(timestamp_dir, "packet_results_final.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved final results to {results_file}")
        
        # 結果の統計
        success_rate = successful_packets / len(results) * 100
        logger.info(f"Test completed: {successful_packets}/{len(results)} packets successful ({success_rate:.1f}%)")
        
        # 統計プロット
        fig = plot_packet_stats(results, "QPSK 20 Mbps Packet Statistics")
        if fig is not None:
            plot_path = os.path.join(timestamp_dir, "packet_statistics.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved statistics plot to {plot_path}")
        
        return {
            "total_packets": len(results),
            "successful_packets": successful_packets,
            "success_rate": success_rate,
            "results_dir": timestamp_dir
        }
        
    except Exception as e:
        logger.error(f"Error occurred in QPSK multi-packet test: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_throughput(packet_size, num_packets, total_time_seconds):
    """スループットを計算する"""
    total_bits = packet_size * 8 * num_packets
    throughput_bps = total_bits / total_time_seconds
    return throughput_bps

def qpsk_throughput_test(packet_size_bytes=10000, num_packets=100):
    """QPSKスループットテスト"""
    
    # 出力ディレクトリの作成
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S_throughput")
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"スループットテスト: データファイルの保存先ディレクトリを作成しました: {timestamp_dir}")
    
    try:
        # パラメータ設定
        sample_rate = 40e6  # サンプリングレート [Hz]
        symbol_rate = 10e6  # シンボルレート [bps] -> QPSKで20 Mbps
        center_freq = 5.84e9  # 中心周波数 [Hz]
        tx_gain = -15  # 送信ゲイン
        rx_gain = 30  # 受信ゲイン
        bandwidth = 20e6  # 帯域幅 [Hz]
        
        # インターリーブの深さを調整（大きなパケットに対応）
        interleave_depth = max(5, (packet_size_bytes + 500) // 1000)
        
        # 通信パイプラインの初期化（QPSK指定）
        pipeline = CCSPipeline(interleave_depth=interleave_depth, sample_rate=sample_rate, symbol_rate=symbol_rate, modulation="qpsk")
        
        # PLUTO SDRの初期化
        logger.info("=== Initializing PLUTO SDR for throughput test ===")
        pluto = PlutoSDRInterface(
            rx_freq=center_freq, 
            tx_freq=center_freq, 
            rx_rate=sample_rate, 
            tx_rate=sample_rate,
            tx_gain=tx_gain,
            rx_gain=rx_gain,
            tx_bandwidth=bandwidth,
            rx_bandwidth=bandwidth
        )
        
        # 信号処理クラスの初期化
        signal_processor = SignalProcessor(sample_rate, symbol_rate)
        
        # テストデータの生成
        test_data = bytearray([i % 256 for i in range(packet_size_bytes)])
        logger.info(f"Generated throughput test data: {len(test_data)} bytes")
        
        # 同期マーカー（プリアンブル）をQPSK用に準備
        sync_bits = np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8))
        sync_i_bits = sync_bits[0::2]  # 偶数ビット→I
        sync_q_bits = sync_bits[1::2]  # 奇数ビット→Q
        sync_i = 2 * sync_i_bits - 1  # BPSKシンボルに変換
        sync_q = 2 * sync_q_bits - 1  # BPSKシンボルに変換
        
        # スループットテスト開始
        logger.info(f"=== Starting QPSK throughput test with {num_packets} packets of {packet_size_bytes} bytes each ===")
        
        # 開始時間の記録
        test_start_time = time.time()
        successful_packets = 0
        
        # パケット処理ループ
        for packet_num in tqdm(range(num_packets), desc="Throughput test"):
            packet_start_time = time.time()
            
            # CADUフレームの生成と送信処理
            cadu_frame, _, i_samples, q_samples = pipeline.process_transmit(test_data, packet_num)
            if cadu_frame is None or i_samples is None or q_samples is None:
                logger.error(f"Packet {packet_num+1}: Transmission processing failed")
                continue
            
            # 送信
            if not pluto.transmit(i_samples, q_samples, cyclic=False):
                logger.error(f"Packet {packet_num+1}: Transmission failed")
                continue
            
            # 送信処理時間を記録
            tx_time = time.time() - packet_start_time
            
            # 受信開始時間
            rx_start_time = time.time()
            
            # 受信
            rx_i, rx_q = pluto.receive(num_samples=len(i_samples) * 2)
            if rx_i is None or rx_q is None:
                logger.error(f"Packet {packet_num+1}: Reception failed")
                continue
            
            # 受信処理時間を記録
            rx_time = time.time() - rx_start_time
            
            # 信号処理（正規化とキャリア回復）
            rx_i_normalized, rx_q_normalized = signal_processor.normalize_signal(rx_i, rx_q)
            rx_i_corrected, rx_q_corrected = signal_processor.apply_carrier_recovery(rx_i_normalized, rx_q_normalized)
            
            # プリアンブル検出
            corr_result = signal_processor.correlate_with_preamble(rx_i_corrected, rx_q_corrected, sync_i, sync_q)
            
            # フレーム開始位置が検出できなかった場合
            if not corr_result["positions"]:
                logger.error(f"Packet {packet_num+1}: No frame start detected")
                continue
            
            # 最も相関の高い位置を使用
            best_idx = np.argmax(np.abs(corr_result["correlations"]))
            start_pos = corr_result["positions"][best_idx]
            
            # フレーム開始位置からの信号抽出
            max_samples = min(len(rx_i_corrected) - start_pos, len(i_samples))
            rx_i_frame = rx_i_corrected[start_pos:start_pos + max_samples]
            rx_q_frame = rx_q_corrected[start_pos:start_pos + max_samples]
            
            # 復号処理
            receive_result = pipeline.process_receive(rx_i_frame, rx_q_frame)
            if receive_result is None:
                logger.error(f"Packet {packet_num+1}: Receive processing failed")
                continue
            
            received_data = receive_result["data"]
            
            # カウンター値の確認
            received_counter = pipeline.frame_generator.get_counter_from_frame(received_data)
            if received_counter != packet_num:
                logger.warning(f"Packet {packet_num+1}: Counter mismatch - expected {packet_num}, got {received_counter}")
            
            # パケット処理の総時間を記録
            packet_time = time.time() - packet_start_time
            
            # パケット成功とカウント
            successful_packets += 1
            
            # 処理時間のログ記録
            logger.info(f"Packet {packet_num+1}: TX={tx_time:.3f}s, RX={rx_time:.3f}s, Total={packet_time:.3f}s")
        
        # テスト終了時間と総時間の計算
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        
        # スループットの計算
        throughput_bps = calculate_throughput(packet_size_bytes, successful_packets, total_test_time)
        throughput_mbps = throughput_bps / 1e6
        
        # 結果の表示
        logger.info("\n=== Throughput Test Results ===")
        logger.info(f"Total time: {total_test_time:.2f} seconds")
        logger.info(f"Successful packets: {successful_packets}/{num_packets} ({successful_packets/num_packets*100:.1f}%)")
        logger.info(f"Packet size: {packet_size_bytes} bytes")
        logger.info(f"Throughput: {throughput_mbps:.2f} Mbps")
        
        # SDR接続を閉じる
        pluto.close()
        
        # 結果を返す
        return {
            "total_packets": num_packets,
            "successful_packets": successful_packets,
            "packet_size_bytes": packet_size_bytes,
            "total_time_seconds": total_test_time,
            "throughput_bps": throughput_bps,
            "throughput_mbps": throughput_mbps,
            "success_rate": successful_packets / num_packets * 100
        }
        
    except Exception as e:
        logger.error(f"Error occurred in throughput test: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン処理"""
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description='PLUTO SDR QPSK 20Mbps テスト')
    parser.add_argument('--test', choices=['single', 'multi', 'throughput'], default='single',
                      help='実行するテストの種類 (single: 単一パケット, multi: 複数パケット, throughput: スループット)')
    parser.add_argument('--packets', type=int, default=100,
                      help='マルチパケットテストで送信するパケット数')
    parser.add_argument('--packet-size', type=int, default=10000,
                      help='スループットテストで使用するパケットサイズ（バイト）')
    args = parser.parse_args()
    
    if args.test == 'single':
        # シングルパケットテスト
        logger.info("シングルパケットQPSKテストを開始します")
        return sdr_qpsk_test()
    elif args.test == 'multi':
        # マルチパケットテスト
        logger.info(f"{args.packets}パケットのQPSKテストを開始します")
        return sdr_qpsk_multi_packet_test(num_packets=args.packets)
    elif args.test == 'throughput':
        # スループットテスト
        logger.info(f"QPSKスループットテストを開始します（パケットサイズ: {args.packet_size}バイト, パケット数: {args.packets}）")
        return qpsk_throughput_test(packet_size_bytes=args.packet_size, num_packets=args.packets)
    else:
        logger.error(f"不明なテスト種類: {args.test}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())