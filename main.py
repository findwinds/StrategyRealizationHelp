"""
MA20趋势跟踪策略 - 主程序
整合所有模块，提供完整的策略运行功能
"""

import pandas as pd
import numpy as np
import logging
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# 导入策略模块
from data_fetcher import DataFetcher
from data_processor import DataProcessor
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from backtest_engine import BacktestEngine
from performance_analyzer import PerformanceAnalyzer, PerformanceVisualizer
from config import get_config, validate_config, get_instrument_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MA20TrendFollowingStrategy:
    """MA20趋势跟踪策略主类"""
    
    def __init__(self, symbol: str = 'RB0', data_source: str = 'akshare'):
        """初始化策略
        
        Args:
            symbol: 交易品种代码
            data_source: 数据源 ('tushare' 或 'akshare')
        """
        self.symbol = symbol
        self.data_source = data_source
        self.config = get_config()
        
        # 初始化各模块
        self.data_fetcher = DataFetcher(data_source)
        self.data_processor = DataProcessor()
        self.signal_generator = SignalGenerator(ma_period=self.config['ma_period'])
        self.risk_manager = RiskManager()
        self.backtest_engine = BacktestEngine(symbol)
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = PerformanceVisualizer()
        
        logger.info(f"MA20趋势跟踪策略初始化完成，品种: {symbol}, 数据源: {data_source}")
    
    def prepare_data(self, start_date: str, end_date: str, 
                    cache_dir: str = 'data/cache') -> pd.DataFrame:
        """准备策略数据
        
        Args:
            start_date: 开始日期 (格式: '2020-01-01')
            end_date: 结束日期 (格式: '2024-12-31')
            cache_dir: 缓存目录
            
        Returns:
            完整的策略数据DataFrame
        """
        logger.info(f"准备数据: {start_date} 至 {end_date}")
        
        # 1. 获取原始数据
        try:
            raw_data = self.data_fetcher.fetch_futures_data(self.symbol, start_date, end_date)
            logger.info(f"获取原始数据: {len(raw_data)} 条记录")
        except Exception as e:
            logger.error(f"数据获取失败: {e}")
            raise
        
        # 2. 保存原始数据缓存
        cache_path = os.path.join(cache_dir, f"{self.symbol}_raw_data.csv")
        os.makedirs(cache_dir, exist_ok=True)
        self.data_fetcher.save_data(raw_data, self.symbol, cache_dir)
        
        # 3. 合成2日K线
        try:
            data_2day = self.data_processor.create_2day_kline(raw_data)
            logger.info(f"合成2日K线: {len(data_2day)} 条记录")
        except Exception as e:
            logger.error(f"2日K线合成失败: {e}")
            raise
        
        # 4. 准备策略数据（计算MA和特征）
        try:
            strategy_data = self.data_processor.prepare_strategy_data(data_2day, self.config['ma_period'])
            logger.info(f"策略数据准备完成: {len(strategy_data)} 条有效记录")
        except Exception as e:
            logger.error(f"策略数据准备失败: {e}")
            raise
        
        # 5. 生成交易信号
        try:
            signals_data = self.signal_generator.generate_signals(strategy_data)
            logger.info(f"信号生成完成")
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            raise
        
        # 6. 数据摘要
        summary = self.data_processor.get_data_summary(signals_data)
        logger.info(f"数据摘要: {summary}")
        
        return signals_data
    
    def run_backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """运行回测
        
        Args:
            data: 策略数据
            initial_capital: 初始资金
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始回测，初始资金: {initial_capital}")
        
        try:
            # 运行回测
            results = self.backtest_engine.run_backtest(data, initial_capital)
            
            # 打印回测报告
            self.backtest_engine.print_backtest_report(results)
            
            logger.info("回测完成")
            return results
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            raise
    
    def analyze_performance(self, backtest_results: Dict[str, Any]) -> str:
        """分析绩效
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            绩效分析报告
        """
        logger.info("开始绩效分析...")
        
        try:
            # 提取交易数据
            trades = backtest_results.get('strategy_data', {}).get('trades', [])
            
            if not trades:
                logger.warning("没有交易数据，无法进行分析")
                return "无交易数据"
            
            # 转换为DataFrame
            trades_df = pd.DataFrame(trades)
            
            # 生成绩效报告
            report = self.performance_analyzer.generate_performance_report(trades_df)
            
            logger.info("绩效分析完成")
            return report
            
        except Exception as e:
            logger.error(f"绩效分析失败: {e}")
            raise
    
    def visualize_results(self, backtest_results: Dict[str, Any], save_dir: str = 'results'):
        """可视化结果
        
        Args:
            backtest_results: 回测结果
            save_dir: 保存目录
        """
        logger.info("开始结果可视化...")
        
        try:
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 提取交易数据
            trades = backtest_results.get('strategy_data', {}).get('trades', [])
            if not trades:
                logger.warning("没有交易数据，无法生成图表")
                return
            
            trades_df = pd.DataFrame(trades)
            
            # 生成各种图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. 交易分布图
            dist_path = os.path.join(save_dir, f"trade_distribution_{timestamp}.png")
            self.visualizer.trade_distribution(trades_df, dist_path)
            
            # 2. 月度表现热力图
            heatmap_path = os.path.join(save_dir, f"monthly_heatmap_{timestamp}.png")
            self.visualizer.monthly_performance_heatmap(trades_df, heatmap_path)
            
            logger.info(f"图表已保存到: {save_dir}")
            
        except Exception as e:
            logger.error(f"结果可视化失败: {e}")
            raise
    
    def run_complete_strategy(self, start_date: str = '2020-01-01', 
                            end_date: str = '2024-12-31',
                            initial_capital: float = 100000,
                            save_results: bool = True) -> Dict[str, Any]:
        """运行完整策略
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            save_results: 是否保存结果
            
        Returns:
            完整结果字典
        """
        logger.info(f"运行完整策略: {self.symbol} ({start_date} 至 {end_date})")
        
        try:
            # 1. 准备数据
            data = self.prepare_data(start_date, end_date)
            
            # 2. 运行回测
            backtest_results = self.run_backtest(data, initial_capital)
            
            # 3. 绩效分析
            performance_report = self.analyze_performance(backtest_results)
            
            # 4. 结果可视化
            if save_results:
                self.visualize_results(backtest_results)
            
            # 5. 保存完整结果
            complete_results = {
                'symbol': self.symbol,
                'data_source': self.data_source,
                'time_range': {'start': start_date, 'end': end_date},
                'initial_capital': initial_capital,
                'data_summary': self.data_processor.get_data_summary(data),
                'backtest_results': backtest_results,
                'performance_report': performance_report,
                'timestamp': datetime.now().isoformat()
            }
            
            if save_results:
                self._save_complete_results(complete_results)
            
            logger.info("完整策略运行完成")
            return complete_results
            
        except Exception as e:
            logger.error(f"完整策略运行失败: {e}")
            raise
    
    def _save_complete_results(self, results: Dict[str, Any]):
        """保存完整结果
        
        Args:
            results: 完整结果字典
        """
        try:
            # 创建结果目录
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = results['symbol']
            filename = f"strategy_results_{symbol}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # 保存为JSON（需要转换一些数据类型）
            import json
            
            # 转换DataFrame为字典
            results_copy = results.copy()
            if 'data_summary' in results_copy and isinstance(results_copy['data_summary'], dict):
                # 转换日期对象为字符串
                for key, value in results_copy['data_summary'].items():
                    if isinstance(value, pd.Timestamp):
                        results_copy['data_summary'][key] = value.isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"完整结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def generate_strategy_report(self, results: Dict[str, Any]) -> str:
        """生成策略报告
        
        Args:
            results: 策略结果
            
        Returns:
            格式化报告字符串
        """
        report = []
        
        # 基本信息
        report.append("=" * 60)
        report.append("           MA20趋势跟踪策略完整报告")
        report.append("=" * 60)
        
        symbol = results['symbol']
        time_range = results['time_range']
        initial_capital = results['initial_capital']
        
        report.append(f"\n【基本信息】")
        report.append(f"交易品种: {symbol}")
        report.append(f"时间范围: {time_range['start']} 至 {time_range['end']}")
        report.append(f"初始资金: {initial_capital:,.2f} CNY")
        report.append(f"数据源: {results['data_source']}")
        
        # 数据摘要
        data_summary = results.get('data_summary', {})
        if data_summary:
            report.append(f"\n【数据摘要】")
            report.append(f"总记录数: {data_summary.get('total_records', 0)}")
            report.append(f"交易日: {data_summary.get('date_range', {}).get('trading_days', 0)}")
            
            price_stats = data_summary.get('price_stats', {})
            if price_stats:
                report.append(f"价格区间: {price_stats.get('lowest', 0):.2f} - {price_stats.get('highest', 0):.2f}")
        
        # 回测结果
        backtest_results = results.get('backtest_results', {})
        if backtest_results:
            basic_info = backtest_results.get('basic_info', {})
            report.append(f"\n【回测结果】")
            report.append(f"最终资产: {basic_info.get('final_value', 0):,.2f} CNY")
            report.append(f"总收益率: {basic_info.get('total_return', 0)*100:+.2f}%")
            report.append(f"交易次数: {basic_info.get('total_trades', 0)}")
            
            # 风险指标
            risk_metrics = backtest_results.get('risk_metrics', {})
            if risk_metrics:
                report.append(f"最大回撤: {risk_metrics.get('max_drawdown_pct', 0):.2f}%")
                report.append(f"夏普比率: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            
            # 交易指标
            trade_metrics = backtest_results.get('trade_metrics', {})
            if trade_metrics:
                report.append(f"胜率: {trade_metrics.get('win_rate_pct', 0):.2f}%")
                report.append(f"盈亏比: {trade_metrics.get('profit_factor', 0):.2f}")
        
        # 绩效报告
        performance_report = results.get('performance_report', '')
        if performance_report:
            report.append(f"\n【详细绩效分析】")
            report.append(performance_report)
        
        report.append(f"\n【生成时间】")
        report.append(f"报告生成时间: {results.get('timestamp', '未知')}")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MA20趋势跟踪策略')
    parser.add_argument('--symbol', type=str, default='RB0', 
                       help='交易品种代码 (默认: RB0)')
    parser.add_argument('--data-source', type=str, default='akshare',
                       choices=['tushare', 'akshare'], help='数据源 (默认: akshare)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='开始日期 (默认: 2020-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='结束日期 (默认: 2024-12-31)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='初始资金 (默认: 100000)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    parser.add_argument('--test', action='store_true',
                       help='运行测试模式')
    
    args = parser.parse_args()
    
    # 验证配置
    if not validate_config():
        logger.error("配置验证失败，请检查配置")
        return
    
    try:
        if args.test:
            # 测试模式
            logger.info("运行测试模式...")
            from test_strategy import run_comprehensive_tests
            success = run_comprehensive_tests()
            if success:
                logger.info("所有测试通过!")
            else:
                logger.error("部分测试失败!")
        else:
            # 正常运行策略
            logger.info("运行MA20趋势跟踪策略...")
            
            # 创建策略实例
            strategy = MA20TrendFollowingStrategy(
                symbol=args.symbol,
                data_source=args.data_source
            )
            
            # 运行完整策略
            results = strategy.run_complete_strategy(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                save_results=not args.no_save
            )
            
            # 生成最终报告
            final_report = strategy.generate_strategy_report(results)
            print("\n" + final_report)
            
            logger.info("策略运行完成!")
            
    except Exception as e:
        logger.error(f"策略运行失败: {e}")
        raise


if __name__ == "__main__":
    main()