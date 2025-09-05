#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDSフレームのインターリーブ/デインターリーブとリードソロモン符号化/復号のテスト
CCSDS 131.0-B-3に準拠
"""

import numpy as np
import logging
import os
from datetime import datetime
import reedsolo as rs

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("interleave_test")

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

class InterleaveTest:
    """インターリーブ/デインターリーブとリードソロモン符号化/復号のテストクラス"""
    
    def __init__(self):
        """初期化"""
        # リードソロモン符号のパラメータ（CCSDS 131.0-B-3に準拠）
        self.rs_n = 255  # 符号長
        self.rs_k = 223  # 情報長
        self.rs_t = 16   # 誤り訂正能力
        self.interleave_depth = 5  # インターリーブ深さ
        
        # テストデータサイズの計算
        self.block_size = self.rs_k  # 情報長
        self.num_blocks = self.interleave_depth  # インターリーブ深さ
        self.total_size = self.block_size * self.num_blocks
        
        # リードソロモン符号器の初期化
        # 生成多項式: x^8 + x^7 + x^2 + x + 1
        self.rs_codec = rs.RSCodec(self.rs_n - self.rs_k, c_exp=8, prim=0x11D)
        
        logger.info(f"InterleaveTest initialized:")
        logger.info(f"  Block size (k): {self.block_size} bytes")
        logger.info(f"  Num blocks: {self.num_blocks}")
        logger.info(f"  Total size: {self.total_size} bytes")
        logger.info(f"  Code length (n): {self.rs_n} bytes")
        logger.info(f"  Error correction (t): {self.rs_t} bytes")
    
    def generate_test_data(self):
        """テストデータの生成"""
        # 各ブロックを異なる値で初期化
        test_data = bytearray()
        for i in range(self.num_blocks):
            block = bytearray([i + 1] * self.block_size)  # ブロックごとに異なる値を使用
            test_data.extend(block)
        
        logger.info(f"Generated test data: {len(test_data)} bytes")
        logger.info(f"First 32 bytes: {test_data[:32].hex()}")
        return test_data
    
    def apply_reed_solomon_encode(self, data):
        """リードソロモン符号化"""
        try:
            # データをブロックに分割
            blocks = [data[i:i+self.block_size] for i in range(0, len(data), self.block_size)]
            if len(blocks) != self.num_blocks:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.num_blocks})")
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
            logger.info(f"First 32 bytes: {encoded_data[:32].hex()}")
            return encoded_data
            
        except Exception as e:
            logger.error(f"RS encoding failed: {e}")
            return None
    
    def apply_reed_solomon_decode(self, data):
        """リードソロモン復号"""
        try:
            # データをブロックに分割
            blocks = [data[i:i+self.rs_n] for i in range(0, len(data), self.rs_n)]
            if len(blocks) != self.num_blocks:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.num_blocks})")
                return None
            
            # 各ブロックを復号
            decoded_blocks = []
            for i, block in enumerate(blocks):
                try:
                    # デバッグ情報の出力
                    logger.info(f"\nProcessing block {i+1}:")
                    logger.info(f"Block type: {type(block)}")
                    logger.info(f"Block length: {len(block)}")
                    logger.info(f"Block first 16 bytes: {block[:16].hex()}")
                    
                    # bytearrayをbytesに変換してからdecodeを呼び出す
                    block_bytes = bytes(block)
                    logger.info(f"Converted to bytes type: {type(block_bytes)}")
                    logger.info(f"Converted to bytes length: {len(block_bytes)}")
                    logger.info(f"Converted to bytes first 16 bytes: {block_bytes[:16].hex()}")
                    
                    decoded_result = self.rs_codec.decode(block_bytes)
                    
                    # 復号結果の処理
                    if isinstance(decoded_result, tuple):
                        decoded_block = decoded_result[0]
                        # エラー訂正情報があれば表示
                        if len(decoded_result) > 1:
                            error_info = decoded_result[1]
                            if isinstance(error_info, list):
                                logger.info(f"Corrected errors at positions: {error_info}")
                            elif isinstance(error_info, int):
                                logger.info(f"Corrected {error_info} errors")
                    else:
                        decoded_block = decoded_result
                    
                    decoded_blocks.append(decoded_block)
                    logger.info(f"RS decoding successful for block {i+1}")
                except Exception as e:
                    logger.error(f"RS decoding failed for block {i+1}: {e}")
                    logger.error(f"Block {i+1} details:")
                    logger.error(f"  Type: {type(block)}")
                    logger.error(f"  Length: {len(block)}")
                    logger.error(f"  First 16 bytes: {block[:16].hex()}")
                    return None
            
            # 復号されたブロックを結合
            decoded_data = bytearray()
            for block in decoded_blocks:
                decoded_data.extend(block)
            
            logger.info(f"RS decoding completed: {len(data)} bytes -> {len(decoded_data)} bytes")
            logger.info(f"First 32 bytes: {decoded_data[:32].hex()}")
            return decoded_data
            
        except Exception as e:
            logger.error(f"RS decoding failed: {e}")
            return None
    
    def interleave(self, data):
        """インターリーブ処理"""
        try:
            # 入力データをブロックに分割（RS符号化後のブロックサイズを使用）
            block_size = self.rs_n  # RS符号化後のブロックサイズ（255バイト）
            blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
            if len(blocks) != self.num_blocks:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.num_blocks})")
                return None
            
            # インターリーブ
            interleaved_data = bytearray()
            for i in range(block_size):
                for block in blocks:
                    if i < len(block):
                        interleaved_data.append(block[i])
            
            logger.info(f"Interleaved data: {len(interleaved_data)} bytes")
            logger.info(f"First 32 bytes: {interleaved_data[:32].hex()}")
            return interleaved_data
            
        except Exception as e:
            logger.error(f"Interleaving failed: {e}")
            return None
    
    def deinterleave(self, data):
        """デインターリーブ処理"""
        try:
            # 入力データをブロックに分割（RS符号化後のブロックサイズを使用）
            block_size = self.rs_n  # RS符号化後のブロックサイズ（255バイト）
            blocks = [bytearray(block_size) for _ in range(self.num_blocks)]
            
            # デインターリーブ
            for i in range(block_size):
                for j in range(self.num_blocks):
                    blocks[j][i] = data[i * self.num_blocks + j]
            
            # ブロックを結合
            deinterleaved_data = bytearray()
            for block in blocks:
                deinterleaved_data.extend(block)
            
            logger.info(f"Deinterleaved data: {len(deinterleaved_data)} bytes")
            logger.info(f"First 32 bytes: {deinterleaved_data[:32].hex()}")
            return deinterleaved_data
            
        except Exception as e:
            logger.error(f"Deinterleaving failed: {e}")
            return None
    
    def compare_data(self, original, processed):
        """データの比較"""
        try:
            # 復号結果がタプルの場合は最初の要素を使用
            if isinstance(processed, tuple):
                processed = processed[0]
            
            if len(original) != len(processed):
                logger.error(f"Data size mismatch: original={len(original)} bytes, processed={len(processed)} bytes")
                return False
            
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
        
        # テストクラスの初期化
        test = InterleaveTest()
        
        # テストデータの生成
        test_data = test.generate_test_data()
        
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
        
        # テストデータの保存
        test_data_path = os.path.join(timestamp_dir, "test_data.bin")
        with open(test_data_path, 'wb') as f:
            f.write(test_data)
        logger.info(f"Saved test data to {test_data_path}")
        
        # リードソロモン符号化
        rs_encoded_data = test.apply_reed_solomon_encode(test_data)
        if rs_encoded_data is None:
            logger.error("RS encoding failed")
            return 1
        
        # リードソロモン符号化データの保存
        rs_encoded_path = os.path.join(timestamp_dir, "rs_encoded_data.bin")
        with open(rs_encoded_path, 'wb') as f:
            f.write(rs_encoded_data)
        logger.info(f"Saved RS encoded data to {rs_encoded_path}")
        
        # インターリーブ
        interleaved_data = test.interleave(rs_encoded_data)
        if interleaved_data is None:
            logger.error("Interleaving failed")
            return 1
        
        # インターリーブデータの保存
        interleaved_path = os.path.join(timestamp_dir, "interleaved_data.bin")
        with open(interleaved_path, 'wb') as f:
            f.write(interleaved_data)
        logger.info(f"Saved interleaved data to {interleaved_path}")
        
        # デインターリーブ
        deinterleaved_data = test.deinterleave(interleaved_data)
        if deinterleaved_data is None:
            logger.error("Deinterleaving failed")
            return 1
        
        # デインターリーブデータの保存
        deinterleaved_path = os.path.join(timestamp_dir, "deinterleaved_data.bin")
        with open(deinterleaved_path, 'wb') as f:
            f.write(deinterleaved_data)
        logger.info(f"Saved deinterleaved data to {deinterleaved_path}")
        
        # リードソロモン復号
        rs_decoded_data = test.apply_reed_solomon_decode(deinterleaved_data)
        if rs_decoded_data is None:
            logger.error("RS decoding failed")
            return 1
        
        # リードソロモン復号データの保存
        rs_decoded_path = os.path.join(timestamp_dir, "rs_decoded_data.bin")
        with open(rs_decoded_path, 'wb') as f:
            f.write(rs_decoded_data)
        logger.info(f"Saved RS decoded data to {rs_decoded_path}")
        
        # データの比較
        if not test.compare_data(test_data, rs_decoded_data):
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