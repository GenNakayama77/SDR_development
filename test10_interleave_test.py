#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDSフレームのインターリーブ/デインターリーブテスト
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
    """インターリーブ/デインターリーブテストクラス"""
    
    def __init__(self):
        """初期化"""
        # リードソロモン符号のパラメータ
        self.rs_n = 255  # 符号長
        self.rs_k = 223  # 情報長
        self.interleave_depth = 5  # インターリーブ深さ
        
        # テストデータサイズの計算
        self.block_size = self.rs_k  # 情報長
        self.num_blocks = self.interleave_depth  # インターリーブ深さ
        self.total_size = self.block_size * self.num_blocks
        
        logger.info(f"InterleaveTest initialized. Block size: {self.block_size}, Num blocks: {self.num_blocks}")
    
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
    
    def interleave(self, data):
        """インターリーブ処理"""
        try:
            # 入力データをブロックに分割
            blocks = [data[i:i+self.block_size] for i in range(0, len(data), self.block_size)]
            if len(blocks) != self.num_blocks:
                logger.error(f"Invalid number of blocks: {len(blocks)} (expected {self.num_blocks})")
                return None
            
            # インターリーブ
            interleaved_data = bytearray()
            for i in range(self.block_size):
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
            # 入力データをブロックに分割
            blocks = [bytearray(self.block_size) for _ in range(self.num_blocks)]
            
            # デインターリーブ
            for i in range(self.block_size):
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
        
        # インターリーブ
        interleaved_data = test.interleave(test_data)
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
        
        # データの比較
        if not test.compare_data(test_data, deinterleaved_data):
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