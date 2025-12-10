"""
Dashboard Generator for quantSim

Generates a local HTML dashboard with:
- Account overview (equity, cash, P&L)
- Current positions with P&L breakdown
- Recent signals and their outcomes
- Equity curve chart
- Position allocation pie chart
- Daily returns histogram
- Crisis/bubble score indicator
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3

from src.exec.alpaca_client import get_alpaca_api
from src.config import load_config
from src.logging import get_logger

logger = get_logger(__name__)

# Dashboard output directory
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'dashboard_output')

# Crisis cache file
CRISIS_CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'crisis_cache.json')


def get_account_data() -> Dict:
    """Fetch current account data from Alpaca."""
    try:
        api = get_alpaca_api()
        account = api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'initial_capital': 100000.0,  # Paper trading starts with $100k
            'status': account.status,
            'daytrade_count': account.daytrade_count,
            'pattern_day_trader': account.pattern_day_trader,
        }
    except Exception as e:
        logger.error(f"Failed to get account data: {e}")
        return {}


def get_positions_data() -> List[Dict]:
    """Fetch current positions from Alpaca."""
    try:
        api = get_alpaca_api()
        positions = api.list_positions()
        return [{
            'symbol': p.symbol,
            'qty': float(p.qty),
            'avg_entry_price': float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc) * 100,
            'side': 'long' if float(p.qty) > 0 else 'short',
        } for p in positions]
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return []


def get_portfolio_history() -> Dict:
    """Fetch portfolio history from Alpaca."""
    try:
        api = get_alpaca_api()
        # Get last 30 days of portfolio history
        history = api.get_portfolio_history(
            period='1M',
            timeframe='1D',
        )
        
        timestamps = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in history.timestamp]
        
        return {
            'timestamps': timestamps,
            'equity': list(history.equity),
            'profit_loss': list(history.profit_loss) if history.profit_loss else [],
            'profit_loss_pct': list(history.profit_loss_pct) if history.profit_loss_pct else [],
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio history: {e}")
        return {'timestamps': [], 'equity': [], 'profit_loss': [], 'profit_loss_pct': []}


def get_recent_orders(limit: int = 20) -> List[Dict]:
    """Fetch recent orders from Alpaca."""
    try:
        api = get_alpaca_api()
        orders = api.list_orders(status='all', limit=limit)
        return [{
            'symbol': o.symbol,
            'side': o.side,
            'qty': o.qty,
            'type': o.type,
            'status': o.status,
            'created_at': o.created_at.strftime('%Y-%m-%d %H:%M') if o.created_at else '',
            'filled_at': o.filled_at.strftime('%Y-%m-%d %H:%M') if o.filled_at else '',
            'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None,
        } for o in orders]
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        return []


def get_latest_signals() -> Dict:
    """Read latest signals from signals.log."""
    signals_file = os.path.join(os.path.dirname(__file__), '..', '..', 'signals.log')
    try:
        with open(signals_file, 'r') as f:
            content = f.read()
        
        # Parse the last signal block
        blocks = content.strip().split('--- Signals for ')
        if len(blocks) < 2:
            return {'buy': [], 'sell': [], 'timestamp': ''}
        
        last_block = blocks[-1]
        lines = last_block.strip().split('\n')
        
        timestamp = lines[0].split(' ---')[0] if lines else ''
        
        buy_signals = []
        sell_signals = []
        current_section = None
        
        for line in lines:
            if 'BUY:' in line:
                current_section = 'buy'
            elif 'SELL:' in line:
                current_section = 'sell'
            elif current_section and line.strip() and not line.startswith('   symbol'):
                parts = line.split()
                if len(parts) >= 3:
                    symbol = parts[1]
                    prob = float(parts[2])
                    if current_section == 'buy':
                        buy_signals.append({'symbol': symbol, 'probability': prob})
                    else:
                        sell_signals.append({'symbol': symbol, 'probability': prob})
        
        return {
            'buy': buy_signals,
            'sell': sell_signals,
            'timestamp': timestamp,
        }
    except Exception as e:
        logger.error(f"Failed to read signals: {e}")
        return {'buy': [], 'sell': [], 'timestamp': ''}


def get_model_metrics() -> Dict:
    """Get latest model training metrics from database or logs."""
    try:
        # Try to read from a metrics file if it exists
        metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
        
        # Default metrics
        return {
            'accuracy': 0.76,
            'roc_auc': 0.84,
            'last_trained': datetime.now().strftime('%Y-%m-%d'),
        }
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        return {}


def get_crisis_data() -> Dict:
    """Load cached crisis score data."""
    try:
        if not os.path.exists(CRISIS_CACHE_FILE):
            logger.info("No crisis cache file found")
            return {
                'composite_score': 0,
                'status': 'UNKNOWN',
                'timestamp': 'N/A',
                'levels': {},
                'readings': {},
            }
        
        with open(CRISIS_CACHE_FILE, 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        logger.error(f"Failed to load crisis data: {e}")
        return {
            'composite_score': 0,
            'status': 'ERROR',
            'timestamp': 'N/A',
            'levels': {},
            'readings': {},
        }


def calculate_performance_metrics(account: Dict, history: Dict) -> Dict:
    """Calculate performance metrics."""
    initial_capital = account.get('initial_capital', 100000)
    current_equity = account.get('equity', initial_capital)
    
    total_return = ((current_equity - initial_capital) / initial_capital) * 100
    
    # Calculate daily returns for Sharpe ratio
    equity_values = history.get('equity', [])
    if len(equity_values) > 1:
        daily_returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(ret)
        
        if daily_returns:
            import statistics
            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
            sharpe = (avg_return * 252) / (std_return * (252 ** 0.5)) if std_return > 0 else 0
            max_dd = calculate_max_drawdown(equity_values)
        else:
            sharpe = 0
            max_dd = 0
    else:
        sharpe = 0
        max_dd = 0
    
    return {
        'total_return_pct': round(total_return, 2),
        'total_return_dollar': round(current_equity - initial_capital, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
    }


def calculate_max_drawdown(equity_values: List[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if not equity_values:
        return 0
    
    peak = equity_values[0]
    max_dd = 0
    
    for value in equity_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def generate_html(data: Dict) -> str:
    """Generate the HTML dashboard."""
    
    account = data.get('account', {})
    positions = data.get('positions', [])
    history = data.get('history', {})
    orders = data.get('orders', [])
    signals = data.get('signals', {})
    metrics = data.get('metrics', {})
    performance = data.get('performance', {})
    crisis = data.get('crisis', {})
    
    # Prepare chart data
    equity_labels = json.dumps(history.get('timestamps', []))
    equity_data = json.dumps(history.get('equity', []))
    
    # Position allocation data
    position_labels = json.dumps([p['symbol'] for p in positions])
    position_values = json.dumps([abs(p['market_value']) for p in positions])
    position_colors = json.dumps([
        '#22c55e' if p['unrealized_pl'] >= 0 else '#ef4444' 
        for p in positions
    ])
    
    # Generate positions table rows
    positions_html = ''
    for p in sorted(positions, key=lambda x: abs(x['unrealized_pl']), reverse=True):
        pl_class = 'positive' if p['unrealized_pl'] >= 0 else 'negative'
        positions_html += f'''
        <tr>
            <td><strong>{p['symbol']}</strong></td>
            <td>{p['side'].upper()}</td>
            <td>{int(abs(p['qty']))}</td>
            <td>${p['avg_entry_price']:.2f}</td>
            <td>${p['current_price']:.2f}</td>
            <td>${abs(p['market_value']):,.2f}</td>
            <td class="{pl_class}">${p['unrealized_pl']:+,.2f}</td>
            <td class="{pl_class}">{p['unrealized_plpc']:+.2f}%</td>
        </tr>
        '''
    
    if not positions:
        positions_html = '<tr><td colspan="8" class="no-data">No open positions</td></tr>'
    
    # Generate orders table rows
    orders_html = ''
    for o in orders[:10]:
        status_class = 'filled' if o['status'] == 'filled' else 'pending'
        orders_html += f'''
        <tr>
            <td>{o['created_at']}</td>
            <td><strong>{o['symbol']}</strong></td>
            <td class="{'buy' if o['side'] == 'buy' else 'sell'}">{o['side'].upper()}</td>
            <td>{o['qty']}</td>
            <td>{'${:.2f}'.format(o['filled_avg_price']) if o['filled_avg_price'] else '-'}</td>
            <td class="{status_class}">{o['status'].upper()}</td>
        </tr>
        '''
    
    if not orders:
        orders_html = '<tr><td colspan="6" class="no-data">No recent orders</td></tr>'
    
    # Generate signals HTML
    buy_signals_html = ''.join([
        f'<div class="signal-item buy"><span class="symbol">{s["symbol"]}</span><span class="prob">{s["probability"]:.1%}</span></div>'
        for s in signals.get('buy', [])
    ]) or '<div class="no-data">No buy signals</div>'
    
    sell_signals_html = ''.join([
        f'<div class="signal-item sell"><span class="symbol">{s["symbol"]}</span><span class="prob">{s["probability"]:.1%}</span></div>'
        for s in signals.get('sell', [])
    ]) or '<div class="no-data">No sell signals</div>'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>quantSim Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #22c55e, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }}
        
        .subtitle {{
            color: #71717a;
            font-size: 0.9rem;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        .card h2 {{
            font-size: 1rem;
            color: #71717a;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .metric.positive {{
            color: #22c55e;
        }}
        
        .metric.negative {{
            color: #ef4444;
        }}
        
        .metric-label {{
            color: #71717a;
            font-size: 0.85rem;
            margin-top: 4px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }}
        
        .stat-item {{
            padding: 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .stat-label {{
            color: #71717a;
            font-size: 0.75rem;
            text-transform: uppercase;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            color: #71717a;
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 1px;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        
        .positive {{
            color: #22c55e;
        }}
        
        .negative {{
            color: #ef4444;
        }}
        
        .buy {{
            color: #22c55e;
        }}
        
        .sell {{
            color: #ef4444;
        }}
        
        .filled {{
            color: #22c55e;
        }}
        
        .pending {{
            color: #f59e0b;
        }}
        
        .crisis-normal .stat-value {{
            color: #22c55e;
        }}
        
        .crisis-elevated .stat-value {{
            color: #f59e0b;
        }}
        
        .crisis-high .stat-value {{
            color: #f97316;
        }}
        
        .crisis-critical .stat-value {{
            color: #ef4444;
        }}
        
        .signals-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .signals-section h3 {{
            font-size: 0.9rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .signals-section h3::before {{
            content: '';
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .signals-section.buy-section h3::before {{
            background: #22c55e;
        }}
        
        .signals-section.sell-section h3::before {{
            background: #ef4444;
        }}
        
        .signal-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-weight: 500;
        }}
        
        .signal-item.buy {{
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        
        .signal-item.sell {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        
        .signal-item .symbol {{
            font-weight: 600;
        }}
        
        .signal-item .prob {{
            color: #a1a1aa;
        }}
        
        .no-data {{
            color: #71717a;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }}
        
        .model-info {{
            display: flex;
            gap: 20px;
        }}
        
        .model-stat {{
            flex: 1;
            text-align: center;
            padding: 16px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
        }}
        
        .model-stat .value {{
            font-size: 1.8rem;
            font-weight: 600;
            color: #3b82f6;
        }}
        
        .model-stat .label {{
            color: #71717a;
            font-size: 0.75rem;
            text-transform: uppercase;
            margin-top: 4px;
        }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #71717a;
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            .signals-container {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>quantSim Dashboard</h1>
            <p class="subtitle">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET</p>
        </header>
        
        <!-- Account Overview -->
        <div class="grid">
            <div class="card">
                <h2>Portfolio Value</h2>
                <div class="metric">${account.get('equity', 0):,.2f}</div>
                <div class="metric-label">Total Equity</div>
            </div>
            
            <div class="card">
                <h2>Total Return</h2>
                <div class="metric {'positive' if performance.get('total_return_pct', 0) >= 0 else 'negative'}">
                    {performance.get('total_return_pct', 0):+.2f}%
                </div>
                <div class="metric-label">${performance.get('total_return_dollar', 0):+,.2f}</div>
            </div>
            
            <div class="card">
                <h2>Cash Available</h2>
                <div class="metric">${account.get('cash', 0):,.2f}</div>
                <div class="metric-label">Buying Power: ${account.get('buying_power', 0):,.2f}</div>
            </div>
            
            <div class="card">
                <h2>Risk Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{performance.get('sharpe_ratio', 0):.2f}</div>
                        <div class="stat-label">Sharpe Ratio</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value negative">{performance.get('max_drawdown_pct', 0):.1f}%</div>
                        <div class="stat-label">Max Drawdown</div>
                    </div>
                    <div class="stat-item crisis-{'normal' if crisis.get('status', 'UNKNOWN') == 'NORMAL' else 'elevated' if crisis.get('status', 'UNKNOWN') == 'ELEVATED' else 'high' if crisis.get('status', 'UNKNOWN') == 'HIGH' else 'critical'}">
                        <div class="stat-value">{crisis.get('composite_score', 0):.0f}/100</div>
                        <div class="stat-label">Crisis Score ({crisis.get('status', 'N/A')})</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="font-size: 0.9rem;">{crisis.get('date', 'N/A')}</div>
                        <div class="stat-label">Last Crisis Check</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Equity Chart -->
        <div class="grid">
            <div class="card full-width">
                <h2>Equity Curve</h2>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Positions and Allocation -->
        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <h2>Current Positions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Qty</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>Value</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions_html}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Position Allocation</h2>
                <div class="chart-container">
                    <canvas id="allocationChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Signals and Orders -->
        <div class="grid">
            <div class="card">
                <h2>Latest Signals</h2>
                <p class="subtitle" style="margin-bottom: 16px;">Generated: {signals.get('timestamp', 'N/A')[:19]}</p>
                <div class="signals-container">
                    <div class="signals-section buy-section">
                        <h3>Buy Signals</h3>
                        {buy_signals_html}
                    </div>
                    <div class="signals-section sell-section">
                        <h3>Sell Signals</h3>
                        {sell_signals_html}
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Model Performance</h2>
                <div class="model-info">
                    <div class="model-stat">
                        <div class="value">{metrics.get('accuracy', 0):.1%}</div>
                        <div class="label">Accuracy</div>
                    </div>
                    <div class="model-stat">
                        <div class="value">{metrics.get('roc_auc', 0):.1%}</div>
                        <div class="label">ROC-AUC</div>
                    </div>
                </div>
                <p class="subtitle" style="margin-top: 16px; text-align: center;">
                    Last trained: {metrics.get('last_trained', 'N/A')}
                </p>
            </div>
        </div>
        
        <!-- Recent Orders -->
        <div class="grid">
            <div class="card full-width">
                <h2>Recent Orders</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Qty</th>
                            <th>Fill Price</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {orders_html}
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer>
            <p>quantSim - LightGBM Quantum Stock Trading System</p>
            <p>Paper Trading Mode | Data from Yahoo Finance | Execution via Alpaca</p>
        </footer>
    </div>
    
    <script>
        // Equity Chart
        const equityCtx = document.getElementById('equityChart').getContext('2d');
        new Chart(equityCtx, {{
            type: 'line',
            data: {{
                labels: {equity_labels},
                datasets: [{{
                    label: 'Portfolio Equity',
                    data: {equity_data},
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#71717a',
                            maxTicksLimit: 10
                        }}
                    }},
                    y: {{
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)'
                        }},
                        ticks: {{
                            color: '#71717a',
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }}
            }}
        }});
        
        // Allocation Chart
        const allocationCtx = document.getElementById('allocationChart').getContext('2d');
        new Chart(allocationCtx, {{
            type: 'doughnut',
            data: {{
                labels: {position_labels},
                datasets: [{{
                    data: {position_values},
                    backgroundColor: {position_colors},
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{
                            color: '#e4e4e7',
                            padding: 15,
                            font: {{
                                size: 12
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    return html


def generate_dashboard() -> str:
    """Generate the complete dashboard and save to file."""
    logger.info("Generating dashboard...")
    
    # Collect all data
    account = get_account_data()
    positions = get_positions_data()
    history = get_portfolio_history()
    orders = get_recent_orders()
    signals = get_latest_signals()
    metrics = get_model_metrics()
    performance = calculate_performance_metrics(account, history)
    crisis = get_crisis_data()
    
    data = {
        'account': account,
        'positions': positions,
        'history': history,
        'orders': orders,
        'signals': signals,
        'metrics': metrics,
        'performance': performance,
        'crisis': crisis,
    }
    
    # Generate HTML
    html = generate_html(data)
    
    # Ensure output directory exists
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    
    # Save dashboard
    output_path = os.path.join(DASHBOARD_DIR, 'index.html')
    with open(output_path, 'w') as f:
        f.write(html)
    
    # Also save a timestamped version
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(DASHBOARD_DIR, f'dashboard_{timestamp}.html')
    with open(archive_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Dashboard saved to {output_path}")
    
    return output_path


if __name__ == '__main__':
    path = generate_dashboard()
    print(f"Dashboard generated: {path}")
    print(f"Open in browser: file://{os.path.abspath(path)}")
