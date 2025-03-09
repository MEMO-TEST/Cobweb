from sklearn.preprocessing import MinMaxScaler

def set_normalized_scaler(bounds):
    feature_min = bounds[0]
    feature_max = bounds[1]
    
    # 创建 MinMaxScaler 对象，并设置 feature_range 参数来指定归一化的范围
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  # 默认 feature_range 是 (0, 1)

    # 使用 fit 方法将自定义的最小值和最大值设置到 scaler 中
    scaler.fit([feature_min, feature_max])
    
    return scaler