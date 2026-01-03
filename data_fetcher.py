"""
MA20趋势跟踪策略 - 数据获取模块
支持tushare和akshare数据源
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from config import get_config, get_instrument_config

# 设置日志
log_config = get_config('logging')
logging.basicConfig(
    level=getattr(logging, log_config.get('level', 'INFO')),
    format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """期货数据获取器"""
    
    def __init__(self, data_source: str = 'tushare'):
        """初始化数据获取器
        
        Args:
            data_source: 数据源 ('tushare' 或 'akshare')
        """
        self.data_source = data_source
        self.config = get_config()
        
        # 初始化数据源
        if data_source == 'tushare' and TUSHARE_AVAILABLE:
            token = self.config['tushare_token']
            if not token:
                raise ValueError("Tushare token未设置，请设置环境变量TUSHARE_TOKEN")
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("Tushare数据源初始化成功")
        elif data_source == 'akshare' and AKSHARE_AVAILABLE:
            logger.info("Akshare数据源初始化成功")
        else:
            raise ValueError(f"数据源 {data_source} 不可用")
    
    def fetch_futures_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取期货历史数据
        
        Args:
            symbol: 期货品种代码 (如 'RB0', 'CU0')
            start_date: 开始日期 (格式: '2020-01-01')
            end_date: 结束日期 (格式: '2024-12-31')
            
        Returns:
            DataFrame包含期货数据
        """
        logger.info(f"获取 {symbol} 数据，时间范围: {start_date} 至 {end_date}")
        
        if self.data_source == 'tushare':
            return self._fetch_from_tushare(symbol, start_date, end_date)
        elif self.data_source == 'akshare':
            return self._fetch_from_akshare(symbol, start_date, end_date)
        else:
            raise ValueError(f"不支持的数据源: {self.data_source}")
    
    def _fetch_from_tushare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Tushare获取期货数据"""
        try:
            # 转换日期格式
            start_date_str = start_date.replace('-', '')
            end_date_str = end_date.replace('-', '')
            
            # 获取品种配置
            instrument_config = get_instrument_config(symbol)
            
            # 获取期货日线数据
            df = self.pro.fut_daily(
                ts_code=f"{symbol}.SHF",  # 假设都是上期所品种
                start_date=start_date_str,
                end_date=end_date_str,
                fields='ts_code,trade_date,open,high,low,close,vol,oi'
            )
            
            if df.empty:
                logger.warning(f"Tushare未找到 {symbol} 的数据")
                # 尝试其他交易所
                for exchange in ['DCE', 'CZCE', 'CFFEX']:
                    df = self.pro.fut_daily(
                        ts_code=f"{symbol}.{exchange}",
                        start_date=start_date_str,
                        end_date=end_date_str,
                        fields='ts_code,trade_date,open,high,low,close,vol,oi'
                    )
                    if not df.empty:
                        logger.info(f"在 {exchange} 找到 {symbol} 数据")
                        break
            
            if df.empty:
                raise ValueError(f"未找到 {symbol} 的数据")
            
            # 数据预处理
            df = self._process_tushare_data(df)
            logger.info(f"成功获取 {len(df)} 条Tushare数据")
            return df
            
        except Exception as e:
            logger.error(f"Tushare数据获取失败: {e}")
            raise
    
    def _fetch_from_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Akshare获取期货数据"""
        try:
            # 获取期货主连数据
            df = ak.futures_zh_daily_sina(symbol=symbol)
            
            if df.empty:
                raise ValueError(f"Akshare未找到 {symbol} 的数据")
            
            # 数据预处理
            df = self._process_akshare_data(df)
            
            # 按日期过滤
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            logger.info(f"成功获取 {len(df)} 条Akshare数据")
            return df
            
        except Exception as e:
            logger.error(f"Akshare数据获取失败: {e}")
            raise
    
    def _process_tushare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理Tushare数据"""
        # 重命名列
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume',
            'oi': 'open_interest'
        })
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 转换价格数据类型
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 转换成交量
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0)
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 数据验证
        self._validate_price_data(df)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
    
    def _process_akshare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理Akshare数据"""
        # 重命名列
        df = df.rename(columns={
            'date': 'date',
            'volume': 'volume'
        })
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 转换价格数据类型
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 转换成交量
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # 添加持仓量（Akshare可能没有）
        if 'open_interest' not in df.columns:
            df['open_interest'] = 0
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 数据验证
        self._validate_price_data(df)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
    
    def _validate_price_data(self, df: pd.DataFrame) -> None:
        """验证价格数据的有效性"""
        # 检查缺失值
        if df[['open', 'high', 'low', 'close']].isnull().any().any():
            logger.warning("价格数据中存在缺失值")
        
        # 检查价格逻辑
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        
        if invalid_high.any():
            logger.warning(f"发现 {invalid_high.sum()} 条数据high价格异常")
        
        if invalid_low.any():
            logger.warning(f"发现 {invalid_low.sum()} 条数据low价格异常")
        
        # 检查价格跳跃（单日涨跌幅超过20%）
        df['price_change'] = df['close'].pct_change().abs()
        extreme_changes = df['price_change'] > 0.2
        
        if extreme_changes.any():
            logger.warning(f"发现 {extreme_changes.sum()} 条数据单日涨跌幅超过20%")
    
    def save_data(self, df: pd.DataFrame, symbol: str, data_dir: str = 'data') -> str:
        """保存数据到本地文件
        
        Args:
            df: 数据DataFrame
            symbol: 品种代码
            data_dir: 保存目录
            
        Returns:
            保存文件路径
        """
        # 创建目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 生成文件名
        start_date = df['date'].min().strftime('%Y%m%d')
        end_date = df['date'].max().strftime('%Y%m%d')
        filename = f"{symbol}_{start_date}_{end_date}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # 保存数据
        df.to_csv(filepath, index=False)
        logger.info(f"数据已保存到: {filepath}")
        
        return filepath
    
    def load_cached_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """从缓存文件加载数据
        
        Args:
            filepath: 缓存文件路径
            
        Returns:
            数据DataFrame，如果文件不存在返回None
        """
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"从缓存加载数据: {len(df)} 条记录")
            return df
        return None


def test_data_fetcher():
    """测试数据获取功能"""
    print("测试数据获取器...")
    
    # 测试Tushare
    if TUSHARE_AVAILABLE:
        try:
            fetcher = DataFetcher('tushare')
            df = fetcher.fetch_futures_data('RB0', '2023-01-01', '2023-01-31')
            print(f"Tushare数据获取成功: {len(df)} 条记录")
            print(df.head())
        except Exception as e:
            print(f"Tushare测试失败: {e}")
    
    # 测试Akshare
    if AKSHARE_AVAILABLE:
        try:
            fetcher = DataFetcher('akshare')
            df = fetcher.fetch_futures_data('RB0', '2023-01-01', '2023-01-31')
            print(f"Akshare数据获取成功: {len(df)} 条记录")
            print(df.head())
        except Exception as e:
            print(f"Akshare测试失败: {e}")


if __name__ == "__main__":
    test_data_fetcher()