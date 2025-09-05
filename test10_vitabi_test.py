#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDSフレームの畳み込み符号化とビタビ復号のテスト
CCSDS 131.0-B-3に準拠
"""

import numpy as np
import logging
import os
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vitabi_test")

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
            logger.info(f"Frame counter: {self.frame_counter}")
            return frame
            
        except Exception as e:
            logger.error(f"Error generating CADU frame: {e}")
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
            logger.info(f"First 32 bytes: {encoded_data[:32].hex()}")
            return encoded_data
            
        except Exception as e:
            logger.error(f"Convolutional encoding failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class ViterbiDecoder:
    """ビタビ復号クラス（CCSDS 131.0-B-3に準拠）"""
    
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
        
        # トレリス構造の初期化
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output_bits = np.zeros((self.num_states, 2, 2), dtype=np.int32)
        
        # トレリス構造の構築
        self._build_trellis()
        logger.info("ViterbiDecoder initialized")
        
    def _build_trellis(self):
        """トレリス構造を構築"""
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
    
    def decode(self, encoded_data, original_cadu_size=None):
        """ビタビ復号を実行"""
        try:
            # バイト列をビット列に変換（すべてのビットを正しく処理）
            encoded_bits = np.unpackbits(np.frombuffer(encoded_data, dtype=np.uint8))
            
            # エンコード時の終端処理による追加ビットを無視するために元のデータサイズを計算
            # CADUフレームのビット数（畳み込み符号化前）
            # 終端処理とパディングを考慮
            original_bit_length = (len(encoded_bits) // 2) - (self.constraint_length - 1)
            
            # シンボルペアに変換（2ビットずつグループ化）
            symbol_pairs = []
            for i in range(0, len(encoded_bits), 2):
                if i+1 < len(encoded_bits):
                    # エンコーダの出力順序と合わせる
                    symbol_pairs.append([encoded_bits[i], encoded_bits[i+1]])
            
            symbol_pairs = np.array(symbol_pairs, dtype=np.int32)
            
            # パスメトリックと生存パスの初期化
            path_metrics = np.full(self.num_states, float('inf'))
            path_metrics[0] = 0  # 初期状態のメトリックを0に設定
            
            # 生存パスの保存用配列（各時点、各状態に対して前の状態と入力ビットを記録）
            survivor_path = np.zeros((len(symbol_pairs), self.num_states, 2), dtype=np.int32)
            
            # デバッグ情報
            logger.info(f"Input size: {len(encoded_data)} bytes, {len(encoded_bits)} bits")
            logger.info(f"Symbol pairs: {len(symbol_pairs)}")
            logger.info(f"Expected original bits: {original_bit_length}")
            
            # ビタビアルゴリズムの実行（前方パス）
            for t in range(len(symbol_pairs)):
                received_symbol = symbol_pairs[t]
                new_metrics = np.full(self.num_states, float('inf'))
                
                for state in range(self.num_states):
                    if path_metrics[state] == float('inf'):
                        continue  # 無効な状態はスキップ
                    
                    for input_bit in range(2):
                        next_state = self.next_state[state, input_bit]
                        expected_output = self.output_bits[state, input_bit]
                        
                        # ハミング距離の計算
                        branch_metric = np.sum(np.abs(expected_output - received_symbol))
                        new_metric = path_metrics[state] + branch_metric
                        
                        if new_metric < new_metrics[next_state]:
                            new_metrics[next_state] = new_metric
                            # 前の状態と入力ビットを記録
                            survivor_path[t, next_state, 0] = state
                            survivor_path[t, next_state, 1] = input_bit
                
                path_metrics = new_metrics.copy()
            
            # トレースバック（後方パス）
            # 最小メトリックの状態から開始
            current_state = np.argmin(path_metrics)
            logger.info(f"Final state with minimum metric: {current_state}")
            
            # トレースバック結果を格納する配列
            decoded_bits = []
            
            # トレースバック
            for t in range(len(symbol_pairs)-1, -1, -1):
                # 現在の状態に対する前の状態と入力ビット
                prev_state = survivor_path[t, current_state, 0]
                input_bit = survivor_path[t, current_state, 1]
                
                # 入力ビットを追加
                decoded_bits.append(input_bit)
                
                # 前の状態に移動
                current_state = prev_state
            
            # 反転して正しい順序に戻す
            decoded_bits.reverse()
            
            # 終端処理ビットを取り除き、元のデータサイズに切り詰める
            if len(decoded_bits) > original_bit_length:
                decoded_bits = decoded_bits[:original_bit_length]
            
            # 元のCADUフレームのバイト数を使用（指定された場合）
            target_size = original_cadu_size if original_cadu_size else (original_bit_length + 7) // 8
            logger.info(f"Target output size: {target_size} bytes")
            
            # ビット配列をnumpy配列に変換
            decoded_bits_np = np.array(decoded_bits, dtype=np.uint8)
            
            # まず適切な長さに切り詰め (8の倍数のビット数)
            needed_bits = target_size * 8
            if len(decoded_bits_np) > needed_bits:
                decoded_bits_np = decoded_bits_np[:needed_bits]
            elif len(decoded_bits_np) < needed_bits:
                # 足りない場合はパディング
                padding = np.zeros(needed_bits - len(decoded_bits_np), dtype=np.uint8)
                decoded_bits_np = np.concatenate([decoded_bits_np, padding])
            
            # バイトに変換
            decoded_data = np.packbits(decoded_bits_np).tobytes()
            
            # 最終的なサイズ調整
            if len(decoded_data) != target_size:
                if len(decoded_data) > target_size:
                    decoded_data = decoded_data[:target_size]
                else:
                    # これは通常発生しないはず
                    logger.warning(f"Unexpected decoded data size: {len(decoded_data)} (expected {target_size})")
                    decoded_data = decoded_data + bytes([0] * (target_size - len(decoded_data)))
            
            logger.info(f"Viterbi decoding completed: {len(encoded_data)} bytes -> {len(decoded_data)} bytes")
            logger.info(f"Number of decoded bits: {len(decoded_bits)} -> trimmed to {needed_bits}")
            logger.info(f"First 32 bytes: {decoded_data[:32].hex()}")
            
            return decoded_data
            
        except Exception as e:
            logger.error(f"Viterbi decoding failed: {e}")
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
        # フレーム生成器の初期化
        frame_generator = CCSDSFrameGenerator()
        
        # 畳み込み符号化器の初期化
        encoder = ConvolutionalEncoder()
        
        # ビタビ復号器の初期化
        decoder = ViterbiDecoder()
        
        # テストデータの生成（1115バイト）
        test_data_size = 1115
        test_data = bytearray([0x55] * test_data_size)  # パターンデータ
        logger.info(f"Generated test data: {len(test_data)} bytes")
        
        # CADUフレームの生成
        cadu_frame = frame_generator.generate_cadu_frame(test_data)
        if cadu_frame is None:
            logger.error("CADU frame generation failed")
            return 1
        
        # CADUフレームの保存
        cadu_frame_path = os.path.join(timestamp_dir, "cadu_frame.bin")
        with open(cadu_frame_path, 'wb') as f:
            f.write(cadu_frame)
        logger.info(f"Saved CADU frame to {cadu_frame_path}")
        
        # CADUフレームサイズを記録
        cadu_frame_size = len(cadu_frame)
        
        # 畳み込み符号化
        encoded_data = encoder.encode(cadu_frame)
        if encoded_data is None:
            logger.error("Convolutional encoding failed")
            return 1
        
        # 畳み込み符号化データの保存
        encoded_path = os.path.join(timestamp_dir, "encoded_data.bin")
        with open(encoded_path, 'wb') as f:
            f.write(encoded_data)
        logger.info(f"Saved encoded data to {encoded_path}")
        
        # ビタビ復号（元のCADUフレームサイズを指定）
        decoded_data = decoder.decode(encoded_data, cadu_frame_size)
        if decoded_data is None:
            logger.error("Viterbi decoding failed")
            return 1
        
        # ビタビ復号データの保存
        decoded_path = os.path.join(timestamp_dir, "decoded_data.bin")
        with open(decoded_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved decoded data to {decoded_path}")
        
        # データの比較
        if not compare_data(cadu_frame, decoded_data):
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