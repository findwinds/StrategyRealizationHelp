"""
MA20趋势跟踪策略 - 可视化工具
生成丰富的图表展示策略表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import os
import logging

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyVisualizer:
    """策略可视化工具"""
    
    def __init__(self, figsize=(15, 10)):
        """初始化可视化工具
        
        Args:
            figsize: 图表大小
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'neutral': '#7f7f7f',
            'background': '#f8f9fa'
        }
    
    def create_comprehensive_report(self, data: pd.DataFrame, trades: pd.DataFrame, 
                                   backtest_results: dict, save_dir: str = 'results'):
        """创建综合报告
        
        Args:
            data: 价格数据
            trades: 交易记录
            backtest_results: 回测结果
            save_dir: 保存目录
        """
        logger.info("创建综合可视化报告...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成各种图表
        self.plot_equity_curve(data, trades, backtest_results, save_dir, timestamp)
        self.plot_price_chart_with_signals(data, trades, save_dir, timestamp)
        self.plot_trade_distribution(trades, save_dir, timestamp)
        self.plot_monthly_performance(trades, save_dir, timestamp)
        self.plot_drawdown_analysis(data, trades, save_dir, timestamp)
        self.plot_trade_timing_analysis(trades, save_dir, timestamp)
        
        logger.info(f"可视化报告已保存到: {save_dir}")
    
    def plot_equity_curve(self, data: pd.DataFrame, trades: pd.DataFrame, 
                         backtest_results: dict, save_dir: str, timestamp: str):
        """绘制权益曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 计算权益曲线
        equity_curve = self._calculate_equity_curve(trades, backtest_results['initial_capital'])
        
        # 主图：权益曲线
        ax1.plot(equity_curve.index, equity_curve.values, 
                color=self.colors['primary'], linewidth=2, label='权益曲线')
        ax1.axhline(y=backtest_results['initial_capital'], 
                   color=self.colors['neutral'], linestyle='--', alpha=0.7, 
                   label='初始资金线')
        
        # 添加最高点和最低点标记
        max_equity = equity_curve.max()
        min_equity = equity_curve.min()
        max_date = equity_curve.idxmax()
        min_date = equity_curve.idxmin()
        
        ax1.scatter(max_date, max_equity, color=self.colors['success'], s=100, 
                   marker='^', label=f'最高点: {max_equity:,.0f}')
        ax1.scatter(min_date, min_equity, color=self.colors['danger'], s=100, 
                   marker='v', label=f'最低点: {min_equity:,.0f}')
        
        ax1.set_title('权益曲线与回撤分析', fontsize=16, fontweight='bold')
        ax1.set_ylabel('资金 (CNY)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 副图：回撤
        drawdown = self._calculate_drawdown(equity_curve)
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color=self.colors['danger'], alpha=0.3, label='回撤')
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'equity_curve_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"权益曲线图已保存: {save_path}")
    
    def plot_price_chart_with_signals(self, data: pd.DataFrame, trades: pd.DataFrame, 
                                     save_dir: str, timestamp: str):
        """绘制价格图表和交易信号"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 主图：价格和MA20
        ax1.plot(data.index, data['close'], color=self.colors['primary'], 
                linewidth=1.5, label='收盘价')
        ax1.plot(data.index, data['ma20'], color=self.colors['secondary'], 
                linewidth=1.5, label='MA20')
        
        # 添加交易信号
        buy_signals = trades[trades['type'] == 'BUY']
        sell_signals = trades[trades['type'] == 'SELL']
        
        # 买入信号
        for _, trade in buy_signals.iterrows():
            if trade['date'] in data.index:
                ax1.scatter(trade['date'], trade['price'], 
                           color=self.colors['success'], s=100, marker='^', 
                           label='买入信号' if _ == buy_signals.index[0] else "")
        
        # 卖出信号
        for _, trade in sell_signals.iterrows():
            if trade['date'] in data.index:
                ax1.scatter(trade['date'], trade['price'], 
                           color=self.colors['danger'], s=100, marker='v', 
                           label='卖出信号' if _ == sell_signals.index[0] else "")
        
        ax1.set_title('价格走势与交易信号', fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格 (CNY)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 副图：成交量
        ax2.bar(data.index, data['volume'], color=self.colors['neutral'], alpha=0.7)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'price_signals_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"价格信号图已保存: {save_path}")
    
    def plot_trade_distribution(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        """绘制交易分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 盈亏分布直方图
        ax1 = axes[0, 0]
        pnls = trades[trades['pnl'].notna()]['pnl']
        ax1.hist(pnls, bins=20, alpha=0.7, color=self.colors['primary'])
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=pnls.mean(), color=self.colors['secondary'], 
                   linestyle='--', label=f'平均: {pnls.mean():.0f}')
        ax1.set_title('盈亏分布', fontweight='bold')
        ax1.set_xlabel('盈亏 (CNY)')
        ax1.set_ylabel('频次')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 盈亏散点图（按时间）
        ax2 = axes[0, 1]
        trades_with_pnl = trades[trades['pnl'].notna()]
        colors = [self.colors['success'] if pnl > 0 else self.colors['danger'] 
                 for pnl in trades_with_pnl['pnl']]
        ax2.scatter(trades_with_pnl.index, trades_with_pnl['pnl'], 
                   c=colors, alpha=0.7, s=50)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('盈亏时间序列', fontweight='bold')
        ax2.set_xlabel('交易序号')
        ax2.set_ylabel('盈亏 (CNY)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 连续盈亏分析
        ax3 = axes[1, 0]
        consecutive_wins = self._calculate_consecutive_wins_losses(pnls)
        win_lengths = [len(streak) for streak in consecutive_wins['wins']]
        loss_lengths = [len(streak) for streak in consecutive_wins['losses']]
        
        if win_lengths:
            ax3.hist(win_lengths, bins=range(1, max(win_lengths)+2), 
                    alpha=0.7, color=self.colors['success'], label='连胜')
        if loss_lengths:
            ax3.hist(loss_lengths, bins=range(1, max(loss_lengths)+2), 
                    alpha=0.7, color=self.colors['danger'], label='连亏')
        ax3.set_title('连续盈亏次数分布', fontweight='bold')
        ax3.set_xlabel('连续次数')
        ax3.set_ylabel('出现频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 盈亏幅度分析
        ax4 = axes[1, 1]
        win_pnls = pnls[pnls > 0]
        loss_pnls = pnls[pnls < 0]
        
        if len(win_pnls) > 0:
            ax4.boxplot([win_pnls, loss_pnls], 
                       labels=['盈利', '亏损'], 
                       patch_artist=True,
                       boxprops=dict(facecolor=self.colors['primary'], alpha=0.7))
        ax4.set_title('盈亏幅度箱线图', fontweight='bold')
        ax4.set_ylabel('盈亏 (CNY)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'trade_distribution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交易分布图已保存: {save_path}")
    
    def plot_monthly_performance(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        """绘制月度表现热力图"""
        # 准备月度数据
        trades_with_pnl = trades[trades['pnl'].notna()].copy()
        trades_with_pnl['month'] = trades_with_pnl['date'].dt.to_period('M')
        
        monthly_stats = trades_with_pnl.groupby('month').agg({
            'pnl': ['sum', 'count', 'mean'],
            'date': 'first'
        }).round(2)
        
        monthly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'first_date']
        monthly_stats['win_rate'] = trades_with_pnl.groupby('month').apply(
            lambda x: (x['pnl'] > 0).mean() * 100
        ).round(1)
        
        # 创建热力图数据
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 月度盈亏热力图
        monthly_pnl = monthly_stats['total_pnl'].values.reshape(-1, 1)
        im1 = ax1.imshow(monthly_pnl, cmap='RdYlGn', aspect='auto')
        ax1.set_title('月度盈亏热力图', fontsize=14, fontweight='bold')
        ax1.set_ylabel('月份')
        ax1.set_yticks(range(len(monthly_stats)))
        ax1.set_yticklabels([str(period) for period in monthly_stats.index])
        
        # 添加数值标签
        for i in range(len(monthly_stats)):
            ax1.text(0, i, f'{monthly_stats.iloc[i]["total_pnl"]:,.0f}', 
                    ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='盈亏 (CNY)')
        
        # 月度胜率
        monthly_wr = monthly_stats['win_rate'].values.reshape(-1, 1)
        im2 = ax2.imshow(monthly_wr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_title('月度胜率热力图', fontsize=14, fontweight='bold')
        ax2.set_ylabel('月份')
        ax2.set_yticks(range(len(monthly_stats)))
        ax2.set_yticklabels([str(period) for period in monthly_stats.index])
        
        # 添加数值标签
        for i in range(len(monthly_stats)):
            ax2.text(0, i, f'{monthly_stats.iloc[i]["win_rate"]:.1f}%', 
                    ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='胜率 (%)')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'monthly_heatmap_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"月度热力图已保存: {save_path}")
    
    def plot_drawdown_analysis(self, data: pd.DataFrame, trades: pd.DataFrame, 
                               save_dir: str, timestamp: str):
        """绘制回撤分析图"""
        # 计算权益曲线和回撤
        equity_curve = self._calculate_equity_curve(trades, 100000)  # 假设初始资金10万
        drawdown = self._calculate_drawdown(equity_curve)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 回撤时间序列
        ax1 = axes[0, 0]
        ax1.fill_between(drawdown.index, drawdown.values, 0, 
                        color=self.colors['danger'], alpha=0.3)
        ax1.plot(drawdown.index, drawdown.values, color=self.colors['danger'], linewidth=1)
        ax1.set_title('回撤时间序列', fontweight='bold')
        ax1.set_ylabel('回撤 (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤分布
        ax2 = axes[0, 1]
        ax2.hist(drawdown[drawdown < 0], bins=30, alpha=0.7, 
                color=self.colors['danger'])
        ax2.axvline(x=drawdown.min(), color='red', linestyle='--', 
                   label=f'最大回撤: {drawdown.min():.2f}%')
        ax2.axvline(x=drawdown.mean(), color=self.colors['secondary'], 
                   linestyle='--', label=f'平均回撤: {drawdown.mean():.2f}%')
        ax2.set_title('回撤分布', fontweight='bold')
        ax2.set_xlabel('回撤 (%)')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回撤恢复时间
        ax3 = axes[1, 0]
        recovery_times = self._calculate_recovery_times(equity_curve, drawdown)
        if recovery_times:
            ax3.hist(recovery_times, bins=15, alpha=0.7, 
                    color=self.colors['primary'])
            ax3.set_title('回撤恢复时间分布', fontweight='bold')
            ax3.set_xlabel('恢复时间 (天)')
            ax3.set_ylabel('频次')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无回撤数据', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
        
        # 4. 回撤与收益关系
        ax4 = axes[1, 1]
        monthly_returns = equity_curve.resample('M').last().pct_change() * 100
        monthly_drawdowns = drawdown.resample('M').min()
        
        if len(monthly_returns) > 1 and len(monthly_drawdowns) > 1:
            ax4.scatter(monthly_drawdowns, monthly_returns, alpha=0.7, 
                       color=self.colors['primary'])
            ax4.set_title('月度回撤与收益关系', fontweight='bold')
            ax4.set_xlabel('月度最大回撤 (%)')
            ax4.set_ylabel('月度收益 (%)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '数据不足', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'drawdown_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"回撤分析图已保存: {save_path}")
    
    def plot_trade_timing_analysis(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        """绘制交易时机分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 交易时间分布（小时）
        ax1 = axes[0, 0]
        trades_with_time = trades.copy()
        trades_with_time['hour'] = trades_with_time['date'].dt.hour
        hour_counts = trades_with_time['hour'].value_counts().sort_index()
        
        ax1.bar(hour_counts.index, hour_counts.values, 
               color=self.colors['primary'], alpha=0.7)
        ax1.set_title('交易时间分布（小时）', fontweight='bold')
        ax1.set_xlabel('小时')
        ax1.set_ylabel('交易次数')
        ax1.grid(True, alpha=0.3)
        
        # 2. 交易星期分布
        ax2 = axes[0, 1]
        trades_with_time['weekday'] = trades_with_time['date'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = trades_with_time['weekday'].value_counts().reindex(weekday_order, fill_value=0)
        
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        ax2.bar(range(len(weekday_counts)), weekday_counts.values, 
               color=self.colors['secondary'], alpha=0.7)
        ax2.set_title('交易星期分布', fontweight='bold')
        ax2.set_xlabel('星期')
        ax2.set_ylabel('交易次数')
        ax2.set_xticks(range(len(weekday_names)))
        ax2.set_xticklabels(weekday_names)
        ax2.grid(True, alpha=0.3)
        
        # 3. 持仓时间分布
        ax3 = axes[1, 0]
        if 'holding_days' in trades.columns:
            holding_days = trades[trades['holding_days'].notna()]['holding_days']
            ax3.hist(holding_days, bins=20, alpha=0.7, color=self.colors['success'])
            ax3.axvline(x=holding_days.mean(), color='red', linestyle='--', 
                       label=f'平均: {holding_days.mean():.1f}天')
            ax3.set_title('持仓时间分布', fontweight='bold')
            ax3.set_xlabel('持仓天数')
            ax3.set_ylabel('频次')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无持仓时间数据', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
        
        # 4. 交易频率时间序列
        ax4 = axes[1, 1]
        trades_with_date = trades.copy()
        trades_with_date['date_only'] = trades_with_date['date'].dt.date
        daily_trades = trades_with_date.groupby('date_only').size()
        
        if len(daily_trades) > 1:
            ax4.plot(daily_trades.index, daily_trades.values, 
                    color=self.colors['primary'], linewidth=1.5)
            ax4.fill_between(daily_trades.index, daily_trades.values, 
                           alpha=0.3, color=self.colors['primary'])
            ax4.set_title('日交易频率', fontweight='bold')
            ax4.set_xlabel('日期')
            ax4.set_ylabel('交易次数')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '数据不足', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'trade_timing_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交易时机分析图已保存: {save_path}")
    
    def _calculate_equity_curve(self, trades: pd.DataFrame, initial_capital: float) -> pd.Series:
        """计算权益曲线"""
        equity_curve = []
        current_capital = initial_capital
        
        for _, trade in trades.iterrows():
            if 'pnl' in trade and pd.notna(trade['pnl']):
                current_capital += trade['pnl']
            equity_curve.append(current_capital)
        
        # 创建时间序列
        dates = trades[trades['pnl'].notna()]['date'].values
        return pd.Series(equity_curve, index=pd.to_datetime(dates))
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """计算回撤"""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        return drawdown
    
    def _calculate_consecutive_wins_losses(self, pnls: pd.Series) -> dict:
        """计算连续盈亏"""
        wins = []
        losses = []
        current_streak = []
        
        for pnl in pnls:
            if pnl > 0:
                if current_streak and current_streak[0] <= 0:
                    if current_streak[0] < 0:
                        losses.append(current_streak)
                    current_streak = []
                current_streak.append(pnl)
            elif pnl < 0:
                if current_streak and current_streak[0] > 0:
                    if current_streak[0] > 0:
                        wins.append(current_streak)
                    current_streak = []
                current_streak.append(pnl)
        
        # 处理最后一个streak
        if current_streak:
            if current_streak[0] > 0:
                wins.append(current_streak)
            else:
                losses.append(current_streak)
        
        return {'wins': wins, 'losses': losses}
    
    def _calculate_recovery_times(self, equity_curve: pd.Series, drawdown: pd.Series) -> list:
        """计算回撤恢复时间"""
        recovery_times = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd == 0 and in_drawdown and start_date:
                in_drawdown = False
                recovery_time = (date - start_date).days
                recovery_times.append(recovery_time)
                start_date = None
        
        return recovery_times


def create_visualization_from_backtest_results():
    """从回测结果创建可视化"""
    import glob
    import json
    
    # 找到最新的回测结果
    result_files = glob.glob('results/backtest_report_*.txt')
    trade_files = glob.glob('results/trades_*.csv')
    
    if not result_files or not trade_files:
        logger.error("未找到回测结果文件")
        return
    
    # 使用最新的文件
    latest_trade_file = max(trade_files, key=os.path.getctime)
    logger.info(f"使用交易文件: {latest_trade_file}")
    
    # 读取交易数据
    trades_df = pd.read_csv(latest_trade_file)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # 读取原始数据（假设存在）
    data_files = glob.glob('data/cache/*.csv')
    if data_files:
        latest_data_file = max(data_files, key=os.path.getctime)
        data_df = pd.read_csv(latest_data_file)
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df = data_df.set_index('date')
    else:
        logger.warning("未找到原始数据文件，使用模拟数据")
        # 创建模拟数据
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        prices = 4000 + np.cumsum(np.random.normal(0, 50, len(dates)))
        data_df = pd.DataFrame({
            'close': prices,
            'ma20': pd.Series(prices).rolling(20).mean(),
            'volume': np.random.randint(10000, 100000, len(dates))
        }, index=dates)
    
    # 创建可视化
    visualizer = StrategyVisualizer()
    
    # 模拟回测结果
    backtest_results = {
        'initial_capital': 100000,
        'final_capital': 67032.34,  # 从之前的回测结果
        'total_return': -0.3297,
        'total_trades': len(trades_df[trades_df['pnl'].notna()])
    }
    
    visualizer.create_comprehensive_report(data_df, trades_df, backtest_results)
    
    logger.info("可视化报告生成完成!")


if __name__ == "__main__":
    create_visualization_from_backtest_results()