from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
from apscheduler.schedulers.background import BackgroundScheduler
from src.pipeline.pre_pipeline import StockDataDownloader, StockDataPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.visualization_pipeline import RunVisPipeline
from src.pipeline.prediction_pipeline import ModelPredictor
from src.pipeline.test_pipeline import RunVisPipelineTest
import pandas as pd
import logging
from bs4 import BeautifulSoup
import traceback
from playwright.sync_api import sync_playwright
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST as StockClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from datetime import datetime, timedelta
import pandas as pd
import threading
import requests
import threading
import plotly.graph_objs as go
import plotly.express as px
import plotly
from dateutil import parser
from src.pipeline.predict_pipeline_fa import load_models, generate_embeddings, generate_long_answer

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv() 

app = Flask(__name__)

CORS(app)

# Alpaca API setup
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
yt_data_api_key = os.getenv('yt_data_api_key')
max_results = 50

PLACEHOLDER_IMAGE = '/static/images/svg/idea-image-placeholder.svg'  # For paper trading

stock_client = StockClient(API_KEY, API_SECRET, BASE_URL)
crypto_client = CryptoHistoricalDataClient()

# Ticker, period, and interval variables
ticker = 'AVAXUSDT'  # Default ticker
period = '10 day'   # Default period
interval = '1MINUTE'  # Default interval
refresh_task_enabled = False

# Initialize scheduler
scheduler = BackgroundScheduler(daemon=True)
lock = threading.Lock()

def backtest_strategy(data_path, initial_cash=10000, hold_period=3, output_dir='backtest'):
    # Load historical data
    data = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)

    # Initialize variables for backtesting
    cash = initial_cash
    position = 0  # Number of shares held

    # List to track portfolio value over time and trades
    portfolio_values = []
    trades = []

    # Loop through the data
    for i in range(len(data)):
        prediction = data['Prediction'].iloc[i]
        close_price = data['Close'].iloc[i]
        
        if prediction == 0 and position > 0:  # Sell signal
            sell_price = close_price
            cash = position * sell_price
            position = 0
            trades.append(('Sell', sell_price, data.index[i]))
            
            # Buy after hold_period candles if within bounds
            buy_index = i + hold_period
            if buy_index < len(data):
                buy_price = data['Close'].iloc[buy_index]
                position = cash / buy_price
                cash = 0
                trades.append(('Buy', buy_price, data.index[buy_index]))
        
        elif prediction in [1, 2, 3, 4, 5] and cash > 0:  # Buy signals
            buy_price = close_price
            position = cash / buy_price
            cash = 0
            trades.append(('Buy', buy_price, data.index[i]))
            
            # Sell after hold_period + (prediction - 2) candles if within bounds
            sell_index = i + hold_period + (prediction - 2)
            if sell_index < len(data):
                sell_price = data['Close'].iloc[sell_index]
                cash = position * sell_price
                position = 0
                trades.append(('Sell', sell_price, data.index[sell_index]))
        
        # Calculate current portfolio value
        current_value = cash + (position * close_price)
        portfolio_values.append(current_value)

    # Final portfolio value
    final_value = cash + (position * data['Close'].iloc[-1])

    # Convert portfolio values list to a pandas Series
    portfolio_values = pd.Series(portfolio_values, index=data.index[:len(portfolio_values)])

    # Calculate cumulative returns
    cumulative_returns = portfolio_values.pct_change().add(1).cumprod().sub(1)

    # Calculate drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max

    # Calculate volatility
    volatility = portfolio_values.pct_change().std() * (252 ** 0.5)  # Annualized volatility assuming 252 trading days

    # Calculate additional metrics
    duration = data.index[-1] - data.index[0]
    exposure_time = len(data[pd.notnull(portfolio_values)]) / len(data) * 100
    buy_and_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    annual_return = ((final_value / initial_cash) ** (252 / len(data)) - 1) * 100
    max_drawdown = drawdown.min() * 100
    avg_drawdown = drawdown.mean() * 100
    max_drawdown_duration = (drawdown[drawdown == drawdown.min()].index[0] - drawdown.cummin().idxmin()).days
    avg_drawdown_duration = drawdown[drawdown < 0].mean()
    win_rate = len([trade for trade in trades if trade[0] == 'Sell' and trade[1] > trades[trades.index(trade) - 1][1]]) / len([trade for trade in trades if trade[0] == 'Sell']) * 100
    best_trade = max([trade[1] - trades[trades.index(trade) - 1][1] for trade in trades if trade[0] == 'Sell'])
    worst_trade = min([trade[1] - trades[trades.index(trade) - 1][1] for trade in trades if trade[0] == 'Sell'])
    avg_trade = sum([trade[1] - trades[trades.index(trade) - 1][1] for trade in trades if trade[0] == 'Sell']) / len([trade for trade in trades if trade[0] == 'Sell'])
    profit_factor = sum([trade[1] - trades[trades.index(trade) - 1][1] for trade in trades if trade[0] == 'Sell' and trade[1] > trades[trades.index(trade) - 1][1]]) / -sum([trade[1] - trades[trades.index(trade) - 1][1] for trade in trades if trade[0] == 'Sell' and trade[1] < trades[trades.index(trade) - 1][1]])
    expectancy = avg_trade / initial_cash * 100
    sqn = (avg_trade / volatility) * (len(trades) ** 0.5)

    # Calculate the number of days of trades
    number_of_trading_days = (data.index[-1] - data.index[0]).total_seconds()/3600

    # Plot the portfolio value over time
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)

    # Save the plot and the metrics in a PDF
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with PdfPages(os.path.join(output_dir, 'backtest_report.pdf')) as pdf:
        pdf.savefig()  # Save the current figure
        
        # Create a new figure for the metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.axis('off')
        backtest_report = {
            'Initial Cash': initial_cash,
            'Final Portfolio Value': final_value,
            'Net Profit': final_value - initial_cash,
            'Cumulative Returns': cumulative_returns.iloc[-1],
            'Max Drawdown': max_drawdown,
            'Avg. Drawdown': avg_drawdown,
            'Max Drawdown Duration': max_drawdown_duration,
            'Avg. Drawdown Duration': avg_drawdown_duration,
            'Annualized Volatility': volatility,
            'Exposure Time [%]': exposure_time,
            'Return [%]': (final_value - initial_cash) / initial_cash * 100,
            'Buy & Hold Return [%]': buy_and_hold_return,
            'Return (Ann.) [%]': annual_return,
            'Sharpe Ratio': 0,  # Placeholder for now
            'Sortino Ratio': 0,  # Placeholder for now
            'Calmar Ratio': 0,  # Placeholder for now
            'Win Rate [%]': win_rate,
            'Best Trade [%]': best_trade / initial_cash * 100,
            'Worst Trade [%]': worst_trade / initial_cash * 100,
            'Avg. Trade [%]': avg_trade / initial_cash * 100,
            'Max. Trade Duration': 0,  # Placeholder for now
            'Avg. Trade Duration': 0,  # Placeholder for now
            'Profit Factor': profit_factor,
            'Expectancy [%]': expectancy,
            'SQN': sqn,
            'Number of Trading Hourserty': number_of_trading_days
        }
        
        table_data = list(backtest_report.items())
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        pdf.savefig()  # Save the table figure
        plt.close()

    # Add metrics to the dataframe
    data['Portfolio Value'] = portfolio_values
    data['Cumulative Returns'] = cumulative_returns
    data['Drawdown'] = drawdown

    # Save the backtest result to a CSV
    counter = 1
    while True:
        filename = os.path.join(output_dir, f'backtesting_{counter}.csv')
        if not os.path.exists(filename):
            break
        counter += 1

    data.to_csv(filename)

'''def backtest_strategy(data_path, initial_cash=10000, hold_period=3, output_dir='backtest'):
    # Read the CSV file
    data = []
    with open(data_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['datetime'] = datetime.strptime(row['Datetime'], '%Y-%m-%d %H:%M:%S%z')
            row['open'] = float(row['Open'])
            row['high'] = float(row['High'])
            row['low'] = float(row['Low'])
            row['close'] = float(row['Close'])
            row['volume'] = float(row['Volume'])
            row['prediction'] = int(row['Prediction'])
            data.append(row)

    # Initialize variables for backtesting
    cash = initial_cash
    position = 0
    equity_curve = []
    trades = []

    # Helper function to find the index of a future datetime
    def find_future_index(current_index, minutes_ahead):
        future_time = data[current_index]['datetime'] + timedelta(minutes=minutes_ahead)
        for i in range(current_index + 1, len(data)):
            if data[i]['datetime'] >= future_time:
                return i
        return len(data) - 1

    # Backtesting loop
    for i in range(len(data)):
        prediction = data[i]['prediction']
        current_price = data[i]['close']

        if position == 0:  # No open position
            if prediction // 10 == 1:  # Downtrend signal
                num_candles = prediction % 10
                future_index = find_future_index(i, num_candles)
                future_price = data[future_index]['close']
                position = -1
                cash += (current_price - future_price) * position
                trades.append(('Sell', current_price, data[i]['datetime'], future_price, data[future_index]['datetime']))
            elif prediction // 10 == 2:  # Uptrend signal
                num_candles = prediction % 10
                future_index = find_future_index(i, num_candles)
                future_price = data[future_index]['close']
                position = 1
                cash += (future_price - current_price) * position
                trades.append(('Buy', current_price, data[i]['datetime'], future_price, data[future_index]['datetime']))
        else:
            position = 0

        equity_curve.append(cash)

    # Convert equity curve to a pandas Series
    equity_curve = pd.Series(equity_curve, index=[row['datetime'] for row in data[:len(equity_curve)]])

    # Calculate performance metrics
    final_cash = cash
    total_return = (final_cash - initial_cash) / initial_cash * 100
    cumulative_returns = equity_curve.pct_change().add(1).cumprod().sub(1)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    avg_drawdown = drawdown.mean() * 100
    max_drawdown_duration = (drawdown[drawdown == drawdown.min()].index[0] - drawdown.cummin().idxmin()).days
    avg_drawdown_duration = drawdown[drawdown < 0].mean()
    volatility = equity_curve.pct_change().std() * (252 ** 0.5)  # Annualized volatility assuming 252 trading days
    exposure_time = len(equity_curve.dropna()) / len(data) * 100
    buy_and_hold_return = (data[-1]['close'] / data[0]['close'] - 1) * 100
    annual_return = ((final_cash / initial_cash) ** (252 / len(data)) - 1) * 100
    win_rate = len([trade for trade in trades if trade[0] == 'Sell' and trade[3] > trade[1]]) / len([trade for trade in trades if trade[0] == 'Sell']) * 100
    best_trade = max([trade[3] - trade[1] for trade in trades if trade[0] == 'Sell'])
    worst_trade = min([trade[3] - trade[1] for trade in trades if trade[0] == 'Sell'])
    avg_trade = sum([trade[3] - trade[1] for trade in trades if trade[0] == 'Sell']) / len([trade for trade in trades if trade[0] == 'Sell'])
    profit_factor = sum([trade[3] - trade[1] for trade in trades if trade[0] == 'Sell' and trade[3] > trade[1]]) / -sum([trade[3] - trade[1] for trade in trades if trade[0] == 'Sell' and trade[3] < trade[1]])
    expectancy = avg_trade / initial_cash * 100
    sqn = (avg_trade / volatility) * (len(trades) ** 0.5)
    number_of_trading_days = (data[-1]['datetime'] - data[0]['datetime']).days

    # Print results
    print(f"Initial Cash: ${initial_cash:.2f}")
    print(f"Final Cash: ${final_cash:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Annualized Volatility: {volatility:.2f}")
    print(f"Exposure Time: {exposure_time:.2f}%")
    print(f"Buy & Hold Return: {buy_and_hold_return:.2f}%")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Best Trade: {best_trade:.2f}")
    print(f"Worst Trade: {worst_trade:.2f}")
    print(f"Average Trade: {avg_trade:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expectancy: {expectancy:.2f}%")
    print(f"SQN: {sqn:.2f}")

    # Plot the equity curve
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title('Equity Curve Over Time')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)

    # Save the plot and the metrics in a PDF
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with PdfPages(os.path.join(output_dir, 'backtest_report.pdf')) as pdf:
        pdf.savefig()  # Save the current figure

        # Create a new figure for the metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.axis('off')
        backtest_report = {
            'Initial Cash': initial_cash,
            'Final Portfolio Value': final_cash,
            'Net Profit': final_cash - initial_cash,
            'Cumulative Returns': cumulative_returns.iloc[-1],
            'Max Drawdown': max_drawdown,
            'Avg. Drawdown': avg_drawdown,
            'Max Drawdown Duration': max_drawdown_duration,
            'Avg. Drawdown Duration': avg_drawdown_duration,
            'Annualized Volatility': volatility,
            'Exposure Time [%]': exposure_time,
            'Return [%]': total_return,
            'Buy & Hold Return [%]': buy_and_hold_return,
            'Return (Ann.) [%]': annual_return,
            'Sharpe Ratio': 0,  # Placeholder for now
            'Sortino Ratio': 0,  # Placeholder for now
            'Calmar Ratio': 0,  # Placeholder for now
            'Win Rate [%]': win_rate,
            'Best Trade [%]': best_trade / initial_cash * 100,
            'Worst Trade [%]': worst_trade / initial_cash * 100,
            'Avg. Trade [%]': avg_trade / initial_cash * 100,
            'Max. Trade Duration': 0,  # Placeholder for now
            'Avg. Trade Duration': 0,  # Placeholder for now,
            'Profit Factor': profit_factor,
            'Expectancy [%]': expectancy,
            'SQN': sqn,
            'Number of Trading Days': number_of_trading_days
        }
        
        table_data = list(backtest_report.items())
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        pdf.savefig()  # Save the table figure
        plt.close()

    # Save the backtest result to a CSV
    counter = 1
    while True:
        filename = os.path.join(output_dir, f'backtesting_{counter}.csv')
        if not os.path.exists(filename):
            break
        counter += 1

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Type', 'Entry Price', 'Entry Time', 'Exit Price', 'Exit Time'])
        writer.writerows(trades)

    # Plot the equity curve (optional, requires matplotlib)
    plt.plot(equity_curve)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Equity Curve')'''


def scheduled_job():
    if refresh_task_enabled:
        if lock.acquire(blocking=False):  # Attempt to acquire the lock without blocking
            try:
                prediction()
                print("Scheduled job executed")
            finally:
                lock.release()  # Ensure the lock is released after the job is done
        else:
            print("Another job is already running, skipping this execution")

def prediction():
        global ticker, period, interval
        pipeline = StockDataPipeline(ticker=ticker, period=period, interval=interval, num_rows=300)
        pipeline.run_pipeline()

        predictor = ModelPredictor()
        predictor.predict_master_signal()

        '''train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        vis_pipeline = RunVisPipelineTest()
        vis_pipeline.run_pipeline(MF=1.0, x=None, y=None)'''

     
def process_stock_data():
    global ticker, period, interval
    try:
        # Download stock data
        downloader = StockDataDownloader(ticker=ticker, period=period, interval=interval,num_rows=50000)
        downloader.download_data()

        # Process the downloaded data
        pipeline = StockDataPipeline(ticker, interval, period,num_rows=20000)
        pipeline.run_pipeline()

        # Train the model
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        # Visualize the data
        vis_pipeline = RunVisPipeline()
        vis_pipeline.run_pipeline(MF=1.0, x=None, y=None)

        graph_filename = 'plot.html'  # Adjust this based on your actual filename
        return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return render_template('index.html')  # Render the form

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')  # Render the form

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Render the form

@app.route('/master_strategy')
def master_strategy():
    return render_template('index_new.html')

@app.route('/process', methods=['POST'])
def process():
    global ticker, period, interval
    try:
        data = request.json  # Access JSON data sent from frontend
        ticker = data.get('ticker', 'AVAXUSDT')
        period = data.get('period', '10 day')
        interval = data.get('interval', '1MINUTE')

        return process_stock_data()
        
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
    
def deep_backtest():
    df1 = pd.read_csv('deepbacktest.csv',index_col='Datetime')
    df2 = pd.read_csv('check1.csv',index_col='Datetime')
    combined_df = pd.concat([df1, df2.tail(2)])
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df.to_csv('deepbacktest.csv')

@app.route('/toggle_refresh', methods=['POST'])
def toggle_refresh():
    global refresh_task_enabled
    refresh_task_enabled = not refresh_task_enabled
    if refresh_task_enabled:
        if not scheduler.get_job('scheduled_job'):
            scheduler.add_job(scheduled_job, 'interval', seconds=13, id='scheduled_job')
            #scheduler.add_job(process_stock_data,'interval', seconds=600, id='process')
            scheduler.add_job(deep_backtest,'interval', seconds=13, id='deepbacktest')
        scheduler.start()
    else:
        if scheduler.get_job('scheduled_job'):
            scheduler.remove_job('scheduled_job')
            scheduler.remove_job('process')
            scheduler.remove_job('deepbacktest')
    return jsonify({'status': 'success', 'refresh_task_enabled': refresh_task_enabled})

@app.route('/update_plot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        MF = float(data.get('MF'))
        x = pd.to_datetime(data.get('x'), format='%Y-%m-%dT%H:%M', utc=True)
        y = pd.to_datetime(data.get('y'), format='%Y-%m-%dT%H:%M', utc=True)

        vis_pipeline = RunVisPipeline()
        vis_pipeline.run_pipeline(MF=MF, x=x, y=y)

        graph_filename = 'plot.html'  # Adjust this based on your actual filename
        return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/renderplot/<path:plot_filename>')
def render_plot(plot_filename):
    try:
        # Serve the generated plot file from the templates directory
        return send_from_directory('templates', plot_filename)

    except Exception as e:
        print("Error:", e)
        return render_template('error.html', error="An error occurred while rendering the plot.")
    
def load_data():
    # Load your data here, for example from a CSV file or database
    data = pd.read_csv('check1.csv')
    return data

@app.route('/get_data')
def get_data():
    data = load_data()
    # Convert DataFrame to dictionary
    dict_data = data.to_dict(orient='records')
    return jsonify(dict_data)

@app.route('/login')
def login():
    return render_template('login.html')  # Render the login page

@app.route('/login/google')
def login_google():
    return redirect(url_for('index'))  # Redirect to the index page

@app.route('/global_market')
def global_market():
    return render_template('global_market.html')

@app.route('/portfolio')
def portfolio_index():
    return render_template('portfolio.html')

@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    try:
        account = stock_client.get_account()
        cash = float(account.cash)
        equity = float(account.equity)

        holdings = stock_client.list_positions()
        holdings_list = []

        if not holdings:
            return jsonify({'error': 'No holdings found'}), 404

        for holding in holdings:
            asset_symbol = holding.symbol
            asset_quantity = float(holding.qty)
            asset_market_value = float(holding.market_value)
            current_price = float(holding.current_price)
            profitloss = float(holding.unrealized_pl)

            holdings_list.append({
                'symbol': asset_symbol,
                'quantity': asset_quantity,
                'market_value': asset_market_value,
                'current_price': current_price,
                'profit_and_loss': profitloss
            })

        transactions = stock_client.get_activities(activity_types='FILL')
        transactions_list = []

        if not transactions:
            return jsonify({'error': 'No transactions found'}), 404

        for transaction in transactions:
            asset_symbol = transaction.symbol
            trade_type = transaction.side
            trade_quantity = float(transaction.qty)
            average_cost = float(transaction.price)
            amount = float(trade_quantity * average_cost)
            status = transaction.order_status
            date = transaction.transaction_time

            transactions_list.append({
                'symbol': asset_symbol,
                'type': trade_type,
                'quantity': trade_quantity,
                'average_cost': average_cost,
                'amount': amount,
                'status': status,
                'date': date
            })

        portfolio_data = {
            'cash': cash,
            'equity': equity,
            'holdings': holdings_list,
            'transactions': transactions_list
        }
        return jsonify(portfolio_data)

    except Exception as e:
        logging.error(f"Error fetching portfolio data: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error fetching portfolio data'}), 500

@app.route('/api/price/<path:symbol>', methods=['GET'])
def get_price(symbol):
    try:
        # Validate and split the symbol correctly for cryptocurrency symbols
        if '/' in symbol:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            if bars.df.empty:
                raise ValueError(f"No bars found for symbol '{symbol}'")
            
            price = bars.df['close'].iloc[-1]
        else:
            # It's a stock symbol
            latest_trade = stock_client.get_latest_trade(symbol)
            if latest_trade is None:
                raise ValueError(f"Unable to fetch latest trade for symbol '{symbol}'")

            price = latest_trade.price

        if price is None:
            raise ValueError(f"Unable to fetch price for the symbol '{symbol}'")

        # Return the price as JSON
        return jsonify({'symbol': symbol, 'price': price})

    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Error fetching price for {symbol}"}), 500

@app.route('/api/buy', methods=['POST'])
def buy():
    try:
        data = request.get_json()
        symbol = data['symbol']
        amount = float(data['amount'])
        order_type = data['order_type']

        if "/" in symbol:  # Check if the symbol is for a cryptocurrency
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            last_price = bars.df['close'].iloc[-1]
        else:  # It's a stock symbol
            last_price = stock_client.get_latest_bar(symbol).c

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price

        logging.info(f"Received buy request: Symbol={symbol}, Amount={amount}, Order Type={order_type}")
        logging.info(f"Fetched price for {symbol}: {price}")

        qty = float(amount / price)
        qty = round(qty, 2)
        logging.info(f"Calculated quantity: {qty}")

        if '/' in symbol:
            time_in_force = 'ioc'  # For symbols containing '/', use IOC
        else:
            time_in_force = 'day' if qty != int(qty) else 'gtc'

        if order_type == 'market':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force=time_in_force
            )
        elif order_type == 'limit':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force=time_in_force,
                limit_price=price
            )

        time.sleep(2)

        return jsonify({'message': 'Buy order placed successfully!'})

    except KeyError as e:
        logging.error(f"Invalid symbol '{symbol}': {str(e)}")
        return jsonify({'error': f"Invalid symbol '{symbol}'"}), 400

    except ValueError as e:
        logging.error(f"Invalid amount '{data['amount']}': {str(e)}")
        return jsonify({'error': f"Invalid amount '{data['amount']}'"}), 400

    except tradeapi.rest.APIError as e:
        error_msg = str(e)
        logging.error(f"Alpaca API error: {error_msg}")
        if 'insufficient' in error_msg.lower():
            return jsonify({'error': 'Insufficient funds to place order'}), 400
        elif 'market is closed' in error_msg.lower():
            return jsonify({'error': 'Market is closed, cannot place order'}), 400
        else:
            return jsonify({'error': 'Error placing order'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in buy endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing buy request'}), 500

@app.route('/api/sell', methods=['POST'])
def sell():
    try:
        data = request.get_json()
        symbol = data['symbol']
        amount = float(data['amount'])
        order_type = data['order_type']

        if "/" in symbol:  # Check if the symbol is for a cryptocurrency
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            last_price = bars.df['close'].iloc[-1]
        else:  # It's a stock symbol
            last_price = stock_client.get_latest_bar(symbol).c

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price

        logging.info(f"Received buy request: Symbol={symbol}, Amount={amount}, Order Type={order_type}")
        logging.info(f"Fetched price for {symbol}: {price}")

        qty = float(amount / price)
        qty = round(qty, 2)
        logging.info(f"Calculated quantity: {qty}")

        if '/' in symbol:
            time_in_force = 'ioc'  # For symbols containing '/', use IOC
        else:
            time_in_force = 'day' if qty != int(qty) else 'gtc'

        if order_type == 'market':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force=time_in_force
            )
        elif order_type == 'limit':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='limit',
                time_in_force=time_in_force,
                limit_price=price
            )

        time.sleep(2)

        return jsonify({'message': 'sell order placed successfully!'})

    except KeyError as e:
        logging.error(f"Invalid symbol '{symbol}': {str(e)}")
        return jsonify({'error': f"Invalid symbol '{symbol}'"}), 400

    except ValueError as e:
        logging.error(f"Invalid amount '{data['amount']}': {str(e)}")
        return jsonify({'error': f"Invalid amount '{data['amount']}'"}), 400

    except tradeapi.rest.APIError as e:
        error_msg = str(e)
        logging.error(f"Alpaca API error: {error_msg}")
        if 'insufficient' in error_msg.lower():
            return jsonify({'error': 'Insufficient funds to place order'}), 400
        elif 'market is closed' in error_msg.lower():
            return jsonify({'error': 'Market is closed, cannot place order'}), 400
        else:
            return jsonify({'error': 'Error placing order'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in buy endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing sell request'}), 500


def read_last_row(file_path):
    return pd.read_csv(file_path).iloc[-1]

def buy_stock(symbol, amount):
    buy_data = {
        'symbol': symbol,
        'amount': amount,
        'order_type': 'market'
    }
    response = requests.post('http://127.0.0.1:5000/api/buy', json=buy_data)
    return response

def sell_stock(symbol, amount):
    sell_data = {
        'symbol': symbol,
        'amount': amount,
        'order_type': 'market'
    }
    response = requests.post('http://127.0.0.1:5000/api/sell', json=sell_data)
    return response

def start_trading(amount, symbol):
    try:
        cash = amount
        position_value = 0
        buy_price = None
        sell_due_time = None

        while True:
            try:
                # Read the last row of the CSV
                last_row = read_last_row('artifacts/predictions_new.csv')
                prediction = int(last_row['Prediction'])
                close_price = float(last_row['Close'])
                signal_time = parser.parse(last_row['Datetime'])

                current_time = datetime.now()

                if position_value > 0 and sell_due_time and current_time >= sell_due_time:
                    # Try to sell the current position after the hold period
                    response = sell_stock(symbol, position_value)
                    if response.status_code == 200:
                        cash += position_value
                        position_value = 0
                        sell_due_time = None
                        print(f"Selling at {close_price} on {current_time}")
                    else:
                        logging.error(f"Failed to sell: {response.json().get('error')}")

                if cash > 0 and position_value == 0 and prediction in [1, 2, 3, 4, 5]:
                    # Buy shares if the prediction is valid
                    buy_price = close_price
                    response = buy_stock(symbol, cash)
                    if response.status_code == 200:
                        position_value = cash
                        cash = 0
                        hold_period = prediction+ 1
                        sell_due_time = current_time + timedelta(minutes=hold_period)
                        print(f"Buying at {buy_price} on {current_time}, hold for {hold_period} minutes")

                time.sleep(8)

            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                traceback.print_exc()

    except Exception as e:
        logging.error(f"Error in start_trading function: {e}")
        traceback.print_exc()

def start_automation(amount, symbol):
    try:
        thread = threading.Thread(target=start_trading, args=(amount, symbol))
        thread.start()

        return jsonify({'status': 'success', 'message': 'Trading automation started'}), 200

    except Exception as e:
        logging.error(f"Unexpected error in start_automation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing start automation request'}), 500

@app.route('/start-automation', methods=['POST'])
def handle_start_automation():
    try:
        data = request.json
        amount = float(data['amount'])
        symbol = data['symbol']

        return start_automation(amount, symbol)

    except KeyError as e:
        logging.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({'error': 'Missing key in request JSON'}), 400

    except ValueError as e:
        logging.error(f"Invalid value in request JSON: {str(e)}")
        return jsonify({'error': f"Invalid value in request JSON: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error in handle_start_automation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing start automation request'}), 500

@app.route('/fa')
def fa():
    # Load models at the start of the application
    return render_template('fa.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    sbert_model, faiss_index, text_generation_tokenizer, text_generation_model, text_chunks = load_models()
    data = request.json
    question = data['question']

    # Generate embedding for the question
    question_embedding = generate_embeddings(sbert_model, [question])

    # Search for the most similar chunks
    k = 10  # Number of most similar chunks to retrieve
    distances, indices = faiss_index.search(question_embedding, k)

    # Retrieve the most relevant chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]

    # Concatenate relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Generate the long answer
    answer = generate_long_answer(text_generation_tokenizer, text_generation_model, question, context)

    return jsonify({"answer": answer})

def fetch_top_cryptos():
    endpoint = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    response = requests.get(endpoint)
    data = response.json()
    
    # Sort by quoteVolume (24h trading volume) and get top 50
    sorted_data = data
    return sorted_data

# Step 2: Fetch Historical Data
def fetch_historical_data(crypto_id):
    endpoint = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '1',
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    # Extract timestamps and corresponding prices
    timestamps = [pd.to_datetime(entry[0], unit='ms') for entry in data.get('prices', [])]
    prices = [entry[1] for entry in data.get('prices', [])]
    
    return timestamps,prices


# Step 3: Process Data
def process_data():
    top_cryptos = fetch_top_cryptos()
    processed_data = []
    for item in top_cryptos:
        name = str(item["name"])
        symbol = item["id"]
        price = float(item["current_price"])
        price_change_24h = float(item["price_change_percentage_24h"])
        market_cap_change_24h = float(item["market_cap_change_24h"])
        market_cap = item["market_cap"]
        logo = item["image"]
        circulating_supply = float(item["circulating_supply"])
        
        # Fetch historical data for the graph
        timestamps,prices = fetch_historical_data(symbol)

        
        processed_data.append({
            "name": name,
            "market_cap_change_24h": market_cap_change_24h,
            "market_cap": market_cap,
            "logo":logo,
            "symbol": symbol,
            "price": price,
            "price_change_24h": price_change_24h,
            "circulating_supply": circulating_supply,
            "timestamps": timestamps,
            "prices": prices
        })
        
    return processed_data

# Step 4: Display Data
@app.route('/watchlist')
def watchlist():
    data = process_data()
    tables = []
    graphs = []
    
    for crypto in data:
        tables.append({
            "name": crypto['name'],
            "market_cap_change_24h": crypto['market_cap_change_24h'],
            "market_cap": crypto['market_cap'],
            "logo":crypto['logo'],
            "symbol": crypto['symbol'],
            "price": crypto['price'],
            "price_change_24h": crypto['price_change_24h'],
            "circulating_supply": crypto['circulating_supply'],
            "timestamps": crypto['timestamps'],
            "prices": crypto['prices']
        })
        
        color = 'green' if crypto["price_change_24h"] > 0 else 'red'
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=crypto["timestamps"],
            y=crypto["prices"],
            mode='lines',
            line=dict(color=color)
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=70,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphs.append({"symbol": crypto["symbol"], "graph": graph_json})
        
    return render_template('watchlist.html', tables=tables, graphs=graphs)

@app.route('/search', methods=['POST'])
def search():
    symbol = request.form['symbol'].upper()
    return redirect(f"https://www.tradingview.com/symbols/{symbol}/")

@app.route('/download_backtest_report')
def download_backtest_report():
    backtest_strategy('check1.csv')
    directory = './backtest'  # Current working directory
    filename = 'backtest_report.pdf'
    return send_from_directory(directory, filename)

@app.route('/fetch_videos', methods=['GET'])
def fetch_videos():
    query = request.args.get('query', 'crypto trading live')
    is_live = request.args.get('is_live', '').lower() in ['true', '1']
    published_after = request.args.get('published_after', '')
    
    url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'part': 'snippet',
        'type': 'video',
        'q': query,
        'maxResults': max_results,
        'key': yt_data_api_key
    }
    if is_live:
        params['eventType'] = 'live'
    if published_after:
        params['publishedAfter'] = published_after
    
    response = requests.get(url, params=params)
    data = response.json()
    
    videos = []
    for item in data.get('items', []):
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_thumbnail = item['snippet']['thumbnails']['high']['url']
        video_source = item['snippet']['channelTitle']
        video_embed_url = f'https://www.youtube.com/watch?v={video_id}'
        videos.append({
            'video_id': video_id,
            'video_title': video_title,
            'video_embed_url': video_embed_url,
            'video_thumbnail': video_thumbnail,
            'video_source':video_source
        })
    
    return jsonify(videos)

@app.route('/news', methods=['GET'])
def get_news():
    if request.headers.get('Accept') == 'application/json':
        symbol = request.args.get('symbol', 'BTCUSD,ETHUSD,USDTUSD,BNBUSD,SOLUSD,XRPUSD,TONUSD,DOGEUSD,ADAUSD,TRXUSD,AVAXUSD,SHIBUSD,DOTUSD')
        limit = int(request.args.get('limit', 500))
        page = int(request.args.get('page', 1))
        
        news = stock_client.get_news(symbol=symbol, limit=limit)
        articles_per_page = 50  # Adjust the number of articles per page if needed
        start_idx = (page - 1) * articles_per_page
        end_idx = page * articles_per_page
        articles = news[start_idx:end_idx]
        
        results = []
        for article in articles:
            small_image_url = None
            for image in article.images:
                if image['size'] == 'small':
                    small_image_url = image['url']
                    break  # Break inside the if condition
            
            results.append({
                'author': article.author,
                'content': article.content,
                'created_at': article.created_at,
                'headline': article.headline,
                'id': article.id,
                'images': small_image_url,
                'source': article.source,
                'summary': article.summary,
                'symbols': article.symbols,
                'updated_at': article.updated_at,
                'url': article.url
            })

        return jsonify(results)
    else:
        return render_template('news.html')
    
def scrape_tradingview_videos_market(url):
    videos = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            html = page.content()
            browser.close()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        for item in soup.find_all('article', class_='card-exterior-Us1ZHpvJ card-AyE8q7_6 stretch-link-title-AyE8q7_6 idea-card-R05xWTMw js-userlink-popup-anchor'):
            video = {}

            thumbnail_tag = item.find('img', class_='image-gDIex6UB')
            if thumbnail_tag:
                thumbnail_src = thumbnail_tag['src']
                if thumbnail_src == PLACEHOLDER_IMAGE:
                    print('Placeholder thumbnail found, skipping item.')
                    continue
                else:
                    video['thumbnail'] = thumbnail_src
            else:
                print('Thumbnail tag not found, skipping item.')
                continue

            title_tag = item.find('a', class_='title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl')
            if title_tag:
                video['title'] = title_tag.text.strip()
            else:
                print('Title not found, skipping item.')
                continue

            url_tag = item.find('a', class_='title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl')
            if url_tag:
                video['url'] = url_tag['href']
            else:
                print('URL not found, skipping item.')
                continue

            duration_tag = item.find('span', class_='content-PlSmolIm')
            if duration_tag:
                video['duration'] = duration_tag.text.strip()
            else:
                print('Duration not found, skipping item.')
                continue

            author_tag = item.find('span', class_='card-author-BhFUdJAZ typography-social-BhFUdJAZ')
            if author_tag:
                video['author'] = author_tag.text.strip()
            else:
                print('Author not found, skipping item.')
                continue

            published_tag = item.find('time', class_='publication-date-CgENjecZ apply-common-tooltip typography-social-CgENjecZ')
            if published_tag and 'title' in published_tag.attrs:
                video['published'] = published_tag['title']
            else:
                print('Published date not found, skipping item.')
                continue

            print(f"Collected video data: {video}")
            videos.append(video)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    return videos

@app.route('/<market_type>_insight')
def market_insight(market_type):
    valid_market_types = [
        'crypto', 'etfs', 'forex', 'futures', 'us_stocks', 'world_stocks', 'bonds'
    ]
    if market_type in valid_market_types:
        return render_template(f'{market_type}_insight.html', videos=[])
    else:
        return "Invalid market type", 404

@app.route('/load_videos')
def load_videos():
    market_type = request.args.get('market_type')
    base_url_dict = {
        'crypto': 'https://www.tradingview.com/markets/cryptocurrencies/ideas/page-{}/?is_video=true',
        'etfs': 'https://www.tradingview.com/markets/etfs/ideas/page-{}/?is_video=true',
        'forex': 'https://www.tradingview.com/markets/currencies/ideas/page-{}/?is_video=true',
        'futures': 'https://www.tradingview.com/markets/futures/ideas/page-{}/?is_video=true',
        'us_stocks': 'https://www.tradingview.com/markets/stocks-usa/ideas/page-{}/?is_video=true',
        'world_stocks': 'https://www.tradingview.com/markets/world-stocks/ideas/page-{}/?is_video=true',
        'bonds': 'https://www.tradingview.com/markets/bonds/ideas/page-{}/?is_video=true'
    }
    
    if market_type not in base_url_dict:
        return jsonify(error="Invalid market type"), 400

    base_url = base_url_dict[market_type]
    videos_per_page = 2
    page = int(request.args.get('page', 1))  # Get the page number from the query string, default to page 1

    start_index = (page - 1) * videos_per_page + 1
    end_index = page * videos_per_page

    all_videos = []
    for page_num in range(start_index, end_index + 1):
        url = base_url.format(page_num)
        videos = scrape_tradingview_videos_market(url)
        all_videos.extend(videos)

    return jsonify(videos=all_videos)

@app.route('/<market_type>')
def market_page(market_type):
    valid_market_types = [
        'us_stocks', 'world_stocks', 'etfs', 'crypto', 'forex', 'futures', 'bonds'
    ]
    if market_type in valid_market_types:
        return render_template(f'{market_type}.html')
    else:
        return "Invalid market type", 404

@app.route('/<market>_live')
def market_live(market):
    templates = {
        'us_stocks': 'us_stocks_live.html',
        'world_stocks': 'world_stocks_live.html',
        'etfs': 'etfs_live.html',
        'crypto': 'crypto_live.html',
        'forex': 'forex_live.html',
        'futures': 'futures_live.html',
        'bonds': 'bonds_live.html'
    }

    template = templates.get(market)
    if template:
        return render_template(template)
    else:
        return "Market not found", 404
    
@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
