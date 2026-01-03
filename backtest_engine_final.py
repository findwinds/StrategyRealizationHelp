"""
MA20趋势跟踪策略 - 最终修复版
解决Backtrader参数传递问题
"""

import backtrader as bt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from config import get_config, get_instrument_config
from signal_generator import SignalGenerator, SignalType
from risk_manager import RiskManager, PositionSide

# 设置日志
logger = logging.getLogger(__name__)


class MA20StrategyFinal(bt.Strategy):
    """MA20趋势跟踪策略 - 最终修复版"""
    
    params = (
        ('ma_period', 20),
        ('max_loss_pct', 0.06),
        ('force_stop_pct', 0.03),
        ('risk_per_trade', 0.02),
        ('symbol', 'RB0'),
        ('commission', 0.0003),
        ('margin_rate', 0.10),
        ('contract_multiplier', 10),
        ('slippage', 0.001),
        ('printlog', True),
    )
    
    def __init__(self):
        """初始化策略"""
        # 技术指标
        self.ma20 = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.ma_period
        )
        
        # 策略组件
        self.signal_generator = SignalGenerator(ma_period=self.p.ma_period)
        self.risk_manager = RiskManager()
        
        # 交易状态
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.position_size = None
        self.position_side = PositionSide.NONE
        self.prev_extreme = None  # 前一根K线的极值价格
        self.extreme_price = None  # 当前持仓期间的极值价格
        
        # 记录交易历史
        self.trades = []
        self.signals = []
        
        # 移动止损标志
        self.stop_moved_to_breakeven = False
        
        logger.info(f"MA20策略初始化完成，周期: {self.p.ma_period}")
    
    def next(self):
        """每个K线的策略逻辑"""
        # 记录前一根K线的极值价格
        if len(self.data) > 1:
            self.prev_extreme = {
                'high': self.data.high[-1],
                'low': self.data.low[-1]
            }
        
        # 如果有未完成的订单，等待
        if self.order:
            return
        
        # 检查当前持仓
        if self.position:
            self._check_exit_conditions()
        else:
            self._check_entry_conditions()
    
    def _check_entry_conditions(self):
        """检查进场条件"""
        current_price = self.data.close[0]
        current_open = self.data.open[0]
        ma_value = self.ma20[0]
        
        # 确保有足够的历史数据
        if len(self.data) < self.p.ma_period + 1:
            return
        
        # 检查做多信号
        if current_price > ma_value and current_price > current_open:
            if self.prev_extreme:
                self._enter_long_position()
        
        # 检查做空信号
        elif current_price < ma_value and current_price < current_open:
            if self.prev_extreme:
                self._enter_short_position()
    
    def _enter_long_position(self):
        """进入做多仓位"""
        logger.info(f"做多信号触发，价格: {self.data.close[0]:.2f}")
        
        # 计算止损
        stop_result = self.risk_manager.calculate_stop_loss(
            entry_price=self.data.close[0],
            prev_extreme=self.prev_extreme['low'],
            direction=PositionSide.LONG
        )
        
        # 计算仓位大小
        capital = self.broker.getvalue()
        position_result = self.risk_manager.calculate_position_size(
            capital=capital,
            entry_price=self.data.close[0],
            stop_price=stop_result.stop_price,
            margin_rate=self.p.margin_rate,
            contract_multiplier=self.p.contract_multiplier
        )
        
        # 下单
        self.order = self.buy(size=position_result.position_size)
        
        # 记录状态
        self.entry_price = self.data.close[0]
        self.stop_price = stop_result.stop_price
        self.position_size = position_result.position_size
        self.position_side = PositionSide.LONG
        self.extreme_price = self.data.high[0]  # 记录极值价格
        self.stop_moved_to_breakeven = False
        
        # 记录信号
        self.signals.append({
            'date': self.data.datetime.date(0),
            'type': 'BUY',
            'price': self.data.close[0],
            'size': position_result.position_size,
            'stop_price': stop_result.stop_price,
            'risk_amount': position_result.risk_amount
        })
        
        self.log(f"做多开仓: 价格={self.data.close[0]:.2f}, 数量={position_result.position_size}, "
                f"止损={stop_result.stop_price:.2f}")
    
    def _enter_short_position(self):
        """进入做空仓位"""
        logger.info(f"做空信号触发，价格: {self.data.close[0]:.2f}")
        
        # 计算止损
        stop_result = self.risk_manager.calculate_stop_loss(
            entry_price=self.data.close[0],
            prev_extreme=self.prev_extreme['high'],
            direction=PositionSide.SHORT
        )
        
        # 计算仓位大小
        capital = self.broker.getvalue()
        position_result = self.risk_manager.calculate_position_size(
            capital=capital,
            entry_price=self.data.close[0],
            stop_price=stop_result.stop_price,
            margin_rate=self.p.margin_rate,
            contract_multiplier=self.p.contract_multiplier
        )
        
        # 下单
        self.order = self.sell(size=position_result.position_size)
        
        # 记录状态
        self.entry_price = self.data.close[0]
        self.stop_price = stop_result.stop_price
        self.position_size = position_result.position_size
        self.position_side = PositionSide.SHORT
        self.extreme_price = self.data.low[0]  # 记录极值价格
        self.stop_moved_to_breakeven = False
        
        # 记录信号
        self.signals.append({
            'date': self.data.datetime.date(0),
            'type': 'SELL',
            'price': self.data.close[0],
            'size': position_result.position_size,
            'stop_price': stop_result.stop_price,
            'risk_amount': position_result.risk_amount
        })
        
        self.log(f"做空开仓: 价格={self.data.close[0]:.2f}, 数量={position_result.position_size}, "
                f"止损={stop_result.stop_price:.2f}")
    
    def _check_exit_conditions(self):
        """检查出场条件"""
        current_price = self.data.close[0]
        current_open = self.data.open[0]
        
        # 更新极值价格
        if self.position_side == PositionSide.LONG:
            self.extreme_price = max(self.extreme_price, self.data.high[0])
        else:
            self.extreme_price = min(self.extreme_price, self.data.low[0])
        
        # 路径A: 浮亏时
        if self._is_losing_position():
            # 1. 价格触及止损位
            if self._check_stop_loss_hit():
                self._close_position("止损触发")
                return
            
            # 2. K线颜色反转
            if self._check_kline_reversal():
                self._close_position("K线反转（浮亏）")
                return
        
        # 路径B: 浮盈时
        else:
            # 移动止损至成本价（保本）
            if not self.stop_moved_to_breakeven:
                self._move_stop_to_breakeven()
            
            # K线颜色反转
            if self._check_kline_reversal():
                self._close_position("K线反转（浮盈）")
                return
    
    def _is_losing_position(self) -> bool:
        """判断是否为亏损仓位"""
        if self.position_side == PositionSide.LONG:
            return self.data.close[0] < self.entry_price
        else:
            return self.data.close[0] > self.entry_price
    
    def _check_stop_loss_hit(self) -> bool:
        """检查是否触发止损"""
        if self.position_side == PositionSide.LONG:
            return self.data.low[0] <= self.stop_price
        else:
            return self.data.high[0] >= self.stop_price
    
    def _check_kline_reversal(self) -> bool:
        """检查K线颜色反转"""
        current_close = self.data.close[0]
        current_open = self.data.open[0]
        
        if self.position_side == PositionSide.LONG:
            # 做多时收阴线
            return current_close < current_open
        else:
            # 做空时收阳线
            return current_close > current_open
    
    def _move_stop_to_breakeven(self):
        """移动止损至成本价"""
        self.stop_price = self.entry_price
        self.stop_moved_to_breakeven = True
        self.log(f"移动止损至成本价: {self.entry_price:.2f}")
    
    def _close_position(self, reason: str):
        """平仓"""
        if self.position_side == PositionSide.LONG:
            self.order = self.sell(size=self.position_size)
            action = "平多"
        else:
            self.order = self.buy(size=self.position_size)
            action = "平空"
        
        # 记录交易
        exit_price = self.data.close[0]
        pnl = self._calculate_pnl(exit_price)
        
        self.trades.append({
            'entry_date': self.signals[-1]['date'],
            'exit_date': self.data.datetime.date(0),
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position_size': self.position_size,
            'position_side': self.position_side.name,
            'pnl': pnl,
            'reason': reason,
            'holding_days': len(self.data) - self.signals[-1].get('bar_idx', len(self.data))
        })
        
        self.log(f"{action}: 价格={exit_price:.2f}, 原因={reason}, 盈亏={pnl:.2f}")
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """计算盈亏"""
        if self.position_side == PositionSide.LONG:
            return (exit_price - self.entry_price) * self.position_size * self.p.contract_multiplier
        else:
            return (self.entry_price - exit_price) * self.position_size * self.p.contract_multiplier
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入成交: 价格={order.executed.price:.2f}, 数量={order.executed.size}")
            else:
                self.log(f"卖出成交: 价格={order.executed.price:.2f}, 数量={order.executed.size}")
            
            self.order = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单失败: {order.status}")
            self.order = None
    
    def notify_trade(self, trade):
        """交易状态通知"""
        if trade.isclosed:
            self.log(f"交易关闭: 毛利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}")
    
    def log(self, txt, dt=None):
        """日志函数"""
        dt = dt or self.data.datetime.date(0)
        if self.p.printlog:
            logger.info(f'{dt.isoformat()} {txt}')
    
    def stop(self):
        """策略结束"""
        self.log(f"策略结束，最终资产: {self.broker.getvalue():.2f}")


class BacktestEngineFinal:
    """最终修复版回测引擎"""
    
    def __init__(self, symbol: str = 'RB0'):
        """初始化回测引擎"""
        self.symbol = symbol
        self.config = get_config()
        self.instrument_config = get_instrument_config(symbol)
        self.cerebro = None
        self.results = None
        
        logger.info(f"回测引擎初始化完成，品种: {symbol}")
    
    def prepare_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """准备回测数据"""
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        # 创建数据feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # 使用索引作为日期
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # 如果没有持仓量数据
        )
        
        return data
    
    def setup_cerebro(self, df: pd.DataFrame, initial_capital: float = 100000):
        """设置回测引擎"""
        self.cerebro = bt.Cerebro()
        
        # 添加数据
        data = self.prepare_data(df)
        self.cerebro.adddata(data)
        
        # 添加策略 - 使用最终修复版策略
        self.cerebro.addstrategy(
            MA20StrategyFinal,
            ma_period=self.config['ma_period'],
            max_loss_pct=self.config['max_loss_pct'],
            force_stop_pct=self.config['force_stop_pct'],
            risk_per_trade=self.config['backtest']['risk_per_trade'],
            symbol=self.symbol,
            **self.instrument_config
        )
        
        # 设置初始资金
        self.cerebro.broker.setcash(initial_capital)
        
        # 设置手续费
        self.cerebro.broker.setcommission(
            commission=self.instrument_config['commission'],
            margin=self.instrument_config['margin_rate'],
            mult=self.instrument_config['contract_multiplier']
        )
        
        # 设置滑点
        self.cerebro.broker.set_slippage_perc(
            perc=self.instrument_config['slippage']
        )
        
        # 添加分析器
        self._add_analyzers()
        
        logger.info(f"回测引擎设置完成，初始资金: {initial_capital}")
    
    def _add_analyzers(self):
        """添加分析器"""
        # 收益率分析
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # 夏普比率
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        # 最大回撤
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # 交易分析
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 时间序列收益
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        # SQN (System Quality Number)
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """运行回测"""
        logger.info("开始运行回测...")
        
        # 设置回测引擎
        self.setup_cerebro(df, initial_capital)
        
        # 运行回测
        self.results = self.cerebro.run()
        
        # 提取结果
        results = self._extract_results()
        
        logger.info("回测运行完成")
        return results
    
    def _extract_results(self) -> Dict[str, Any]:
        """提取回测结果"""
        if not self.results:
            return {}
        
        strat = self.results[0]
        
        # 基本收益指标
        final_value = self.cerebro.broker.getvalue()
        initial_capital = self.cerebro.broker.startingcash
        total_return = (final_value - initial_capital) / initial_capital
        
        # 获取分析器结果
        returns_analyzer = strat.analyzers.returns.get_analysis()
        sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
        trades_analyzer = strat.analyzers.trades.get_analysis()
        
        # 交易统计
        total_trades = trades_analyzer.total.total
        won_trades = trades_analyzer.won.total if hasattr(trades_analyzer.won, 'total') else 0
        lost_trades = trades_analyzer.lost.total if hasattr(trades_analyzer.lost, 'total') else 0
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏统计
        pnl_won = trades_analyzer.won.pnl.total if hasattr(trades_analyzer.won, 'pnl') else 0
        pnl_lost = trades_analyzer.lost.pnl.total if hasattr(trades_analyzer.lost, 'pnl') else 0
        profit_factor = abs(pnl_won / pnl_lost) if pnl_lost != 0 else float('inf')
        
        results = {
            'basic_info': {
                'symbol': self.symbol,
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': total_trades,
            },
            'return_metrics': {
                'total_return_pct': total_return * 100,
                'annual_return_pct': returns_analyzer.get('rnorm100', 0),
                'avg_return_pct': returns_analyzer.get('ravg', 0) * 100,
            },
            'risk_metrics': {
                'max_drawdown_pct': drawdown_analyzer.max.drawdown,
                'max_drawdown_period': drawdown_analyzer.max.len,
                'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0),
            },
            'trade_metrics': {
                'win_rate_pct': win_rate * 100,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'profit_factor': profit_factor,
                'avg_win': trades_analyzer.won.pnl.average if hasattr(trades_analyzer.won, 'pnl') else 0,
                'avg_loss': trades_analyzer.lost.pnl.average if hasattr(trades_analyzer.lost, 'pnl') else 0,
            },
            'strategy_data': {
                'trades': strat.trades,
                'signals': strat.signals,
                'ma_values': list(strat.ma20.array),
            }
        }
        
        return results
    
    def print_backtest_report(self, results: Dict[str, Any]):
        """打印回测报告"""
        if not results:
            print("没有回测结果")
            return
        
        print("\n" + "="*50)
        print("           回 测 报 告")
        print("="*50)
        
        # 基本信息
        basic = results['basic_info']
        print(f"品种: {basic['symbol']}")
        print(f"初始资金: {basic['initial_capital']:,.2f} CNY")
        print(f"最终资产: {basic['final_value']:,.2f} CNY")
        print(f"总收益率: {basic['total_return']*100:+.2f}%")
        print(f"总交易次数: {basic['total_trades']}")
        
        # 收益指标
        returns = results['return_metrics']
        print(f"\n收益指标:")
        print(f"  年化收益率: {returns['annual_return_pct']:+.2f}%")
        print(f"  平均收益率: {returns['avg_return_pct']:+.2f}%")
        
        # 风险指标
        risk = results['risk_metrics']
        print(f"\n风险指标:")
        print(f"  最大回撤: {risk['max_drawdown_pct']:+.2f}%")
        print(f"  回撤期: {risk['max_drawdown_period']} 天")
        print(f"  夏普比率: {risk['sharpe_ratio']:.2f}")
        
        # 交易指标
        trade = results['trade_metrics']
        print(f"\n交易指标:")
        print(f"  胜率: {trade['win_rate_pct']:.2f}%")
        print(f"  盈利交易: {trade['won_trades']}")
        print(f"  亏损交易: {trade['lost_trades']}")
        print(f"  盈亏比: {trade['profit_factor']:.2f}")
        print(f"  平均盈利: {trade['avg_win']:.2f}")
        print(f"  平均亏损: {trade['avg_loss']:.2f}")
        
        print("="*50)


if __name__ == "__main__":
    # 测试最终修复版
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    print("测试最终修复版MA20趋势跟踪策略...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='2D')
    n = len(dates)
    
    # 生成价格数据（趋势+随机波动）
    trend = np.linspace(4000, 4500, n)
    noise = np.random.normal(0, 50, n)
    prices = trend + noise
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.normal(0, 20, n),
        'high': prices + np.random.uniform(0, 100, n),
        'low': prices - np.random.uniform(0, 100, n),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n)
    })
    
    # 确保价格逻辑正确
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        test_data.loc[i, 'high'] = max(row['high'], row['open'], row['close'])
        test_data.loc[i, 'low'] = min(row['low'], row['open'], row['close'])
    
    # 测试最终修复版回测引擎
    engine = BacktestEngineFinal('RB0')
    results = engine.run_backtest(test_data, initial_capital=100000)
    
    # 打印报告
    engine.print_backtest_report(results)
    
    print("\n最终修复版回测引擎测试完成!")
    print(f"策略在测试期间实现了 {results['return_metrics']['total_return_pct']:+.2f}% 的收益率")
    print(f"共进行了 {results['basic_info']['total_trades']} 笔交易，胜率 {results['trade_metrics']['win_rate_pct']:.2f}%")