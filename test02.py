#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版PlutoSDR ループバックテスト
CCSDS準拠のデータフレーム送受信テスト - バッファサイズを最適化し、同期検出を改善

通信経路:
PCでダミーデータを生成 → 符号化 → BPSK変調 → PlutoSDRの送信ポート → 
同軸ケーブルでループバック → PlutoSDRの受信ポート → PC → 復調 → 復号 → 
データ比較

PlutoSDRの送信ポートと受信ポートを同軸ケーブルで接続する必要があります
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
MAX_TEST_ITERATIONS = 15  # 15回のループで終了
test_results = []  # テスト結果を保存


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
        
        logger.info(f"Generated CADU frame, {len(cadu)} bytes, Counter: {self.vcdu_counter-1}")
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
        
        # ビットストリームの継続分析用データバッファ
        self.bit_buffer = np.array([], dtype=np.uint8)
        
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
    
    def find_sync_marker_in_bits(self, bits):
        """ビットストリーム内の同期マーカーを検索（ビット単位）"""
        # 同期マーカーをビット配列に変換
        sync_bits = np.unpackbits(np.frombuffer(self.sync_marker, dtype=np.uint8))
        sync_len = len(sync_bits)
        
        if len(bits) < sync_len:
            return -1
        
        # スライディングウィンドウを使って検索
        for i in range(len(bits) - sync_len + 1):
            if np.array_equal(bits[i:i+sync_len], sync_bits):
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
        # 簡易実装：ここでは実際の復号は行わず、エラーがないと仮定して元のデータを返す
        logger.info("Reed-Solomon decoding would happen here in a real implementation")
        
        # 実際の実装では、インターリーブ処理を元に戻し、RSコーデックを使用して復号する
        return True, 0  # 成功、修正エラー数
    
    def append_to_bit_buffer(self, bits):
        """ビットバッファに新しいビットを追加"""
        self.bit_buffer = np.append(self.bit_buffer, bits)
        
        # バッファが大きくなりすぎないように制限（最大2フレーム分程度）
        max_buffer_size = self.cadu_size * 8 * 2
        if len(self.bit_buffer) > max_buffer_size:
            self.bit_buffer = self.bit_buffer[-max_buffer_size:]
    
    def parse_from_bit_buffer(self):
        """ビットバッファからCADUフレームを解析"""
        # 同期マーカーをビット単位で検索
        sync_pos = self.find_sync_marker_in_bits(self.bit_buffer)
        if sync_pos == -1:
            return None
        
        # 同期マーカーが見つかった場合、その位置からCADUサイズ分のビットを取得
        cadu_bits = self.bit_buffer[sync_pos:sync_pos + self.cadu_size * 8]
        if len(cadu_bits) < self.cadu_size * 8:
            return None
        
        # ビットをバイトに変換
        cadu_bytes = self.bits_to_bytes(cadu_bits)
        
        # CADUフレームを解析
        frame_info = self.parse_cadu_frame(cadu_bytes)
        if frame_info is None:
            return None
        
        # 解析成功したら、処理済みのビットを削除
        self.bit_buffer = self.bit_buffer[sync_pos + self.cadu_size * 8:]
        
        return frame_info
    
    def parse_cadu_frame(self, cadu_data):
        """CADUフレームを解析"""
        if len(cadu_data) < self.cadu_size:
            logger.error(f"CADU data too short: {len(cadu_data)} < {self.cadu_size}")
            return None
        
        # 同期マーカーを検証
        if cadu_data[:len(self.sync_marker)] != self.sync_marker:
            logger.error("Invalid sync marker")
            return None
        
        # VCDUヘッダーを解析
        header_info = self.decode_vcdu_header(cadu_data[len(self.sync_marker):len(self.sync_marker)+12])
        if header_info is None:
            return None
        
        # VCDUデータを抽出
        vcdu_data = cadu_data[len(self.sync_marker):len(self.sync_marker)+self.vcdu_size]
        
        # RS符号を抽出
        rs_ecc = cadu_data[len(self.sync_marker)+self.vcdu_size:]
        
        # RS復号を試みる
        success, errors = self.apply_reed_solomon_decode(rs_ecc, vcdu_data)
        
        return {
            'version': header_info['version'],
            'scid': header_info['scid'],
            'vcid': header_info['vcid'],
            'counter': header_info['counter'],
            'payload': vcdu_data,
            'rs_success': success,
            'rs_errors': errors
        }


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


class LoopbackTest:
    """PlutoSDRを使用したループバックテストクラス"""
    
    def __init__(self, args):
        """ループバックテストの初期化"""
        self.args = args
        self.frame_gen = CCSDSFrameGenerator()
        self.frame_parser = CCSDSFrameParser()
        self.modulator = BPSKModulator(args.sample_rate, args.bit_rate)
        
        # 送受信キュー
        self.tx_queue = queue.Queue()
        self.rx_queue = queue.Queue()
        
        # 最近の送信フレームを保持するデータ構造
        self.sent_frames = deque(maxlen=20)  # 最新の20フレームを保持
        
        # スレッド終了フラグ
        self.stop_event = threading.Event()
        
        # 受信データを保存する連続バッファ
        self.rx_bit_buffer = []
        
        # SDR初期化
        try:
            self.sdr = adi.Pluto(args.device_uri)
            
            # 送信設定
            self.sdr.tx_lo = int(args.freq)
            self.sdr.tx_rf_bandwidth = int(args.sample_rate)
            self.sdr.tx_hardwaregain_chan0 = int(args.tx_gain)
            self.sdr.tx_cyclic_buffer = False
            
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
            logger.info(f"Tx gain: {args.tx_gain} dB, Rx gain: {args.rx_gain} dB")
            logger.info(f"Buffer size: {args.buffer_size} samples")
        except Exception as e:
            logger.error(f"Failed to initialize SDR: {e}")
            sys.exit(1)
    
    def generate_frame(self):
        """テストフレームを生成して送信キューに追加"""
        for i in range(MAX_TEST_ITERATIONS):
            if self.stop_event.is_set():
                break
                
            # フレーム生成
            cadu_frame = self.frame_gen.generate_cadu_frame()
            
            # フレームをキューに追加
            self.tx_queue.put(cadu_frame)
            
            # 送信フレーム履歴に追加
            self.sent_frames.append(cadu_frame)
            
            # 適度な間隔を空ける
            time.sleep(0.5)  # 間隔を広げて受信側の処理時間を確保
        
        # すべてのフレームが送信されるのを待機
        time.sleep(2)
        
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
                    preamble = np.ones(1000, dtype=complex) * 2**14  # 明確な開始マーカー
                    postamble = -np.ones(1000, dtype=complex) * 2**14  # 明確な終了マーカー
                    
                    # 結合
                    tx_samples_with_amble = np.concatenate([preamble, tx_samples, postamble])
                    
                    # SDRへの送信
                    self.sdr.tx(tx_samples_with_amble)
                    logger.info(f"Transmitted frame with counter: {self.frame_gen.vcdu_counter-1}")
                    
                    # キュー処理完了を通知
                    self.tx_queue.task_done()
                    
                    # 送信後に少し待機（受信側の処理時間を確保）
                    time.sleep(0.2)
                else:
                    # キューが空の場合は少し待機
                    time.sleep(0.01)
            
            except queue.Empty:
                # タイムアウトした場合は再試行
                continue
            except Exception as e:
                logger.error(f"Transmission error: {e}")
                time.sleep(0.1)
        
        logger.info("Transmitter thread completed")
    
    def receive_thread(self):
        """受信スレッド"""
        logger.info("Starting receiver thread")
        
        while not self.stop_event.is_set():
            try:
                # サンプルを受信
                rx_samples = self.sdr.rx()
                if rx_samples is None:
                    continue
                
                logger.info(f"Received {len(rx_samples)} samples")
                
                # BPSK復調
                bits = self.modulator.demodulate(rx_samples)
                if bits is None:
                    continue
                
                logger.info(f"Demodulated {len(rx_samples)} IQ samples to {len(bits)} bits")
                
                # ビットバッファに追加
                self.frame_parser.append_to_bit_buffer(bits)
                
                # フレーム解析を試みる
                while True:
                    frame_info = self.frame_parser.parse_from_bit_buffer()
                    if frame_info is None:
                        break
                    
                    # フレーム情報をログ出力
                    logger.info(f"Found CADU frame: Counter={frame_info['counter']}, "
                              f"Version={frame_info['version']}, "
                              f"SCID={frame_info['scid']}, "
                              f"VCID={frame_info['vcid']}")
                    
                    # フレームをキューに追加
                    self.rx_queue.put(frame_info)
                    
                    # テスト結果を記録
                    self.compare_frames(frame_info)
                
            except Exception as e:
                logger.error(f"Error in receive thread: {str(e)}")
                continue
            
        logger.info("Receiver thread completed")
    
    def compare_frames(self, rx_frame_info):
        """送信フレームと受信フレームを比較"""
        # 受信したVCIDとカウンター値を取得
        rx_vcid = rx_frame_info['header']['vcid']
        rx_counter = rx_frame_info['header']['counter']
        
        # 送信履歴から一致するフレームを探す
        for sent_frame in list(self.sent_frames):
            # 送信フレームからヘッダー情報を抽出
            header_start = len(self.frame_gen.sync_marker)
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
                
                # 90%以上一致した場合は成功とみなす
                if match_percent > 90:
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
        logger.info("Starting loopback test")
        
        # 受信スレッドを開始（先に受信スレッドを開始することで、送信が始まる前にリスニングを開始）
        rx_thread = threading.Thread(target=self.receive_thread)
        rx_thread.start()
        
        # 少し待機して受信スレッドが起動するのを待つ
        time.sleep(0.5)
        
        # 送信スレッドを開始
        tx_thread = threading.Thread(target=self.transmit_thread)
        tx_thread.start()
        
        # フレーム生成スレッドを開始
        gen_thread = threading.Thread(target=self.generate_frame)
        gen_thread.start()
        
        try:
            # すべてのスレッドが終了するのを待機
            gen_thread.join()
            tx_thread.join()
            # 受信スレッドにさらに時間を与える（最後のフレームを処理する時間）
            time.sleep(2)
            self.stop_event.set()
            rx_thread.join()
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            self.stop_event.set()
            gen_thread.join()
            tx_thread.join()
            rx_thread.join()
        
        # テスト結果を表示
        self.print_test_results()
        
        # リソースのクリーンアップ
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
                header_start = len(self.frame_gen.sync_marker)
                sent_vcid = frame[header_start + 1] & 0x3F
                sent_counter = int.from_bytes(frame[header_start + 2:header_start + 5], byteorder='big')
                logger.info(f"  VCID={sent_vcid}, Counter={sent_counter}")
        
        logger.info("=================================")
    
    def close(self):
        """リソースのクリーンアップ"""
        logger.info("Closing SDR resources")


def main():
    """メイン関数"""
    import argparse
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Improved PlutoSDR CCSDS loopback test')
    parser.add_argument('--freq', type=float, default=5.84e9,
                      help='Center frequency in Hz (default: 5.84 GHz)')
    parser.add_argument('--sample-rate', type=float, default=10e6,
                      help='Sample rate in Hz (default: 10 MSps)')
    parser.add_argument('--bit-rate', type=float, default=1e6,
                      help='Bit rate in bps (default: 1 Mbps)')
    parser.add_argument('--tx-gain', type=float, default=-5,
                      help='TX gain in dB (default: -5 dB)')
    parser.add_argument('--rx-gain', type=float, default=60,
                      help='RX gain in dB (default: 60 dB)')
    parser.add_argument('--buffer-size', type=int, default=131072,
                      help='RX buffer size (default: 131072, approximately 2^17)')
    parser.add_argument('--device-uri', type=str, default='ip:192.168.2.1',
                      help='SDR device URI (default: ip:192.168.2.1)')
    
    args = parser.parse_args()
    
    try:
        # ループバックテストを実行
        test = LoopbackTest(args)
        test.run_test()
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())