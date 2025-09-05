#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlutoSDRループバックテスト - 1Mbps用
CCSDS準拠のフレームを送信・受信して整合性を確認します。
"""

import argparse
from test03_100kbps import LoopbackTest
import logging
import sys

# ログ出力の初期化
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("origamisat2_loopback_1mbps")

def main():
    parser = argparse.ArgumentParser(description='1 Mbps PlutoSDR CCSDS loopback test')
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

    args = parser.parse_args()

    try:
        test = LoopbackTest(args)
        test.run_test()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
