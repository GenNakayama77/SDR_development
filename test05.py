def demodulate(self, iq_samples):
    """IQサンプルをBPSK復調してビット配列に変換"""
    # 正規化
    normalized = iq_samples / np.max(np.abs(iq_samples))
    
    # プリアンブル検出（より長いプリアンブルを使用）
    correlation = np.correlate(normalized.real, np.ones(4000), mode='valid')
    sync_start = np.argmax(correlation)
    
    if correlation[sync_start] < 0.7:  # 閾値を調整
        logger.warning(f"Weak sync marker detected: {correlation[sync_start]:.3f}")
        return None
        
    # ポストアンブル検出
    post_correlation = np.correlate(normalized.real[sync_start:], -np.ones(4000), mode='valid')
    sync_end = np.argmax(post_correlation)
    
    if post_correlation[sync_end] < 0.7:  # 閾値を調整
        logger.warning(f"Weak postamble detected: {post_correlation[sync_end]:.3f}")
        return None
        
    # データ部分の抽出（プリアンブルとポストアンブルを除く）
    data_samples = normalized[sync_start+4000:sync_start+sync_end]
    
    # ダウンサンプリングとビット判定（改善版）
    bits = []
    for i in range(0, len(data_samples), self.samples_per_symbol):
        # シンボル期間の中央付近のサンプルを使用
        symbol_samples = data_samples[i:i+self.samples_per_symbol]
        if len(symbol_samples) < self.samples_per_symbol:
            break
            
        # 複数サンプルの平均を取ってビット判定
        symbol_avg = np.mean(symbol_samples.real)
        
        if abs(symbol_avg) < 0.3:  # 閾値を調整
            logger.warning(f"Ambiguous bit detected: {symbol_avg:.3f}")
            
        bits.append(1 if symbol_avg > 0 else 0)
        
    return np.array(bits, dtype=np.uint8) 