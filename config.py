"""
MA20趋势跟踪策略配置文件
包含所有参数配置和交易品种设置
"""

import os
from typing import Dict, Any

# 基础配置
BASE_CONFIG = {
    # 数据配置
    'data_source': 'akshare',  # 'tushare' 或 'akshare'
    'tushare_token': os.getenv('TUSHARE_TOKEN', ''),  # 从环境变量获取
    'data_cache_dir': 'data/cache',
    
    # 策略参数
    'ma_period': 20,  # MA20周期
    'max_loss_pct': 0.06,  # 最大止损容忍度6%
    'force_stop_pct': 0.03,  # 强制止损3%
    
    # 交易品种配置
    'instruments': {
        'RB0': {  # 螺纹钢主连
            'name': '螺纹钢主连',
            'exchange': 'SHF',
            'commission': 0.0003,  # 万分之三
            'margin_rate': 0.10,  # 保证金10%
            'contract_multiplier': 10,  # 合约乘数
            'slippage': 0.001,  # 滑点0.1%
        },
        'CU0': {  # 铜主连
            'name': '铜主连',
            'exchange': 'SHF',
            'commission': 0.00005,  # 万分之0.5
            'margin_rate': 0.08,  # 保证金8%
            'contract_multiplier': 5,
            'slippage': 0.001,
        },
        'IF0': {  # 沪深300主连
            'name': '沪深300主连',
            'exchange': 'CFFEX',
            'commission': 0.000023,  # 万分之0.23
            'margin_rate': 0.12,  # 保证金12%
            'contract_multiplier': 300,
            'slippage': 0.001,
        }
    },
    
    # 回测配置
    'backtest': {
        'start_date': '2024-01-01',
        'end_date': '2025-12-31',
        'initial_capital': 100000,  # 初始资金
        'risk_per_trade': 0.02,  # 每笔交易风险2%
        'max_position_size': 0.8,  # 最大仓位80%
    },
    
    # 日志配置
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/strategy.log',
    }
}

# K线合成配置
RESAMPLE_CONFIG = {
    'target_period': '2D',  # 2日K线
    'aggregation_rules': {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum',
    }
}

# 性能分析配置
ANALYSIS_CONFIG = {
    'indicators': [
        'total_return',
        'annual_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'trade_count',
        'avg_holding_days'
    ],
    'visualization': {
        'equity_curve': True,
        'drawdown_chart': True,
        'trade_distribution': True,
        'monthly_returns': True,
    }
}

def get_config(section: str = None) -> Dict[str, Any]:
    """获取配置
    
    Args:
        section: 配置段名称，如果为None返回所有配置
        
    Returns:
        配置字典
    """
    if section is None:
        return BASE_CONFIG
    return BASE_CONFIG.get(section, {})

def get_instrument_config(symbol: str) -> Dict[str, Any]:
    """获取特定品种配置
    
    Args:
        symbol: 品种代码
        
    Returns:
        品种配置字典
    """
    return BASE_CONFIG['instruments'].get(symbol, {})

def validate_config() -> bool:
    """验证配置有效性
    
    Returns:
        配置是否有效
    """
    # 检查Tushare token
    if BASE_CONFIG['data_source'] == 'tushare' and not BASE_CONFIG['tushare_token']:
        print("警告: Tushare token未设置，请设置环境变量TUSHARE_TOKEN")
        return False
    
    # 检查参数范围
    if BASE_CONFIG['ma_period'] <= 0:
        print("错误: MA周期必须大于0")
        return False
    
    if BASE_CONFIG['max_loss_pct'] <= 0 or BASE_CONFIG['force_stop_pct'] <= 0:
        print("错误: 止损比例必须大于0")
        return False
    
    return True