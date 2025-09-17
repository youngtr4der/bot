import backtrader as bt
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os
import quantstats

class MlStrategy(bt.Strategy):
    params = (('time_steps', 10), ('feature_df', None),)

    def __init__(self):
        self.model = load_model('final_model.keras')
        self.scaler = joblib.load('final_scaler.joblib')
        self.order = None
        self.features = self.p.feature_df.drop(columns=['label', 'touch_time'], errors='ignore')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} --- {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
            else: self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.position: return
        current_date = self.datas[0].datetime.datetime(0)
        try:
            past_data = self.features.loc[:current_date].tail(self.p.time_steps)
            if len(past_data) < self.p.time_steps: return

            scaled_features = self.scaler.transform(past_data)
            model_input = np.reshape(scaled_features, (1, self.p.time_steps, scaled_features.shape[1]))
            prediction_probs = self.model.predict(model_input, verbose=0)[0]
            predicted_class = np.argmax(prediction_probs) - 1

            self.log(f'Close: {self.datas[0].close[0]:.2f}, Signal: {predicted_class}')

            if predicted_class == 1:
                self.order = self.buy()
            elif predicted_class == -1:
                self.order = self.sell()
        except KeyError:
            return

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    feature_df = pd.read_parquet('data/featured_labeled_data.parquet')
    price_df = pd.read_parquet('data/xbtusd_2h.parquet')
    price_df = price_df.loc[feature_df.index]
    data_feed = bt.feeds.PandasData(dataname=price_df)

    cerebro.adddata(data_feed)
    cerebro.addstrategy(MlStrategy, feature_df=feature_df)

    # Add a sizer for position management (Risk Management)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=20) # Bet 20% of portfolio

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # --- Generate QuantStats Report ---
    print("\n--- Generating QuantStats Report ---")
    strat = results[0]
    portfolio_returns = pd.Series(strat.analyzers.time_return.get_analysis())

    if not portfolio_returns.empty:
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        quantstats.reports.html(portfolio_returns, output='report.html', title='ML Strategy Backtest')
        print("Report saved to report.html")
    else:
        print("Could not generate report: No returns data.")

    print("\n--- Backtest Complete ---")
