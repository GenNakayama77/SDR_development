#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCSDSフレームのリードソロモン符号化/復号のテスト
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
logger = logging.getLogger("reedsolo_test")

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

class ReedSolomonTest:
    """リードソロモン符号化/復号のテストクラス"""
    
    def __init__(self):
        """初期化"""
        # リードソロモン符号のパラメータ（CCSDS 131.0-B-3に準拠）
        self.rs_n = 255  # 符号長
        self.rs_k = 223  # 情報長
        self.rs_t = 16   # 誤り訂正能力
        
        # リードソロモン符号器の初期化
        # 生成多項式: x^8 + x^7 + x^2 + x + 1
        self.rs_codec = rs.RSCodec(self.rs_n - self.rs_k, c_exp=8, prim=0x11D)
        
        logger.info(f"ReedSolomonTest initialized:")
        logger.info(f"  Block size (k): {self.rs_k} bytes")
        logger.info(f"  Code length (n): {self.rs_n} bytes")
        logger.info(f"  Error correction (t): {self.rs_t} bytes")
    
    def generate_test_data(self):
        """テストデータの生成"""
        # テストデータを生成（223バイト）
        # パターンを明確にするため、0x55の代わりに0-222の連番を使用
        test_data = bytearray(range(self.rs_k))
        
        logger.info(f"Generated test data: {len(test_data)} bytes")
        logger.info(f"First 32 bytes: {test_data[:32].hex()}")
        return test_data
    
    def apply_reed_solomon_encode(self, data):
        """リードソロモン符号化"""
        try:
            # データを符号化
            encoded_data = self.rs_codec.encode(data)
            
            logger.info(f"RS encoding completed: {len(data)} bytes -> {len(encoded_data)} bytes")
            logger.info(f"First 32 bytes: {encoded_data[:32].hex()}")
            return encoded_data
            
        except Exception as e:
            logger.error(f"RS encoding failed: {e}")
            return None
    
    def apply_reed_solomon_decode(self, data):
        """リードソロモン復号"""
        try:
            # データを復号
            decoded_result = self.rs_codec.decode(data)
            
            # 復号結果の処理
            if isinstance(decoded_result, tuple):
                # タプルの場合は最初の要素（復号データ）のみ使用
                decoded_data = decoded_result[0]
                # エラー訂正情報があれば表示
                if len(decoded_result) > 1:
                    error_info = decoded_result[1]
                    if isinstance(error_info, list):
                        logger.info(f"Corrected errors at positions: {error_info}")
                    elif isinstance(error_info, int):
                        logger.info(f"Corrected {error_info} errors")
            else:
                # タプルでない場合はそのまま使用
                decoded_data = decoded_result
            
            logger.info(f"RS decoding completed: {len(data)} bytes -> {len(decoded_data)} bytes")
            logger.info(f"First 32 bytes: {decoded_data[:32].hex()}")
            return decoded_data
            
        except Exception as e:
            logger.error(f"RS decoding failed: {e}")
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
            
            # データパターンの検証
            for i in range(len(original)):
                if original[i] != i:
                    logger.error(f"Invalid data pattern at position {i}: expected {i}, got {original[i]}")
                    return False
            
            logger.info(f"Data match: 100% ({match_count}/{len(original)} bytes)")
            logger.info("Data pattern verification passed")
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
        test = ReedSolomonTest()
        
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
        
        # リードソロモン復号
        rs_decoded_data = test.apply_reed_solomon_decode(rs_encoded_data)
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