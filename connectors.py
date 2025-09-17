import requests
import pandas as pd
from datetime import datetime

class BinanceConnector:
    """
    A connector to fetch public market data from Binance.
    """
    BASE_URL = "https://api.binance.com/api/v3/"

    def fetch_ohlcv(self, symbol: str, interval: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical OHLCV (Klines) data for a specific symbol.
        """
        endpoint = f"{self.BASE_URL}klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()

class KrakenConnector:
    """
    A connector to fetch public market data from Kraken.
    """
    BASE_URL = "https://api.kraken.com/0/public/"

    def fetch_ohlcv(self, pair: str, interval: int = 60, since: int = None) -> (pd.DataFrame, int):
        """
        Fetches historical OHLCV data for a specific pair from Kraken.
        """
        endpoint = f"{self.BASE_URL}OHLC"
        params = {
            'pair': pair.upper(),
            'interval': interval,
        }
        if since:
            params['since'] = since
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('error'):
                return pd.DataFrame(), None
            result = data.get('result', {})
            last_timestamp = result.get('last')
            result_key = next((k for k in result if k != 'last'), None)
            if not result_key:
                return pd.DataFrame(), None
            raw_ohlc = result[result_key]
            df = pd.DataFrame(raw_ohlc, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df.set_index('open_time', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            return df, last_timestamp
        except Exception as e:
            print(f"Error fetching data from Kraken: {e}")
            return pd.DataFrame(), None
