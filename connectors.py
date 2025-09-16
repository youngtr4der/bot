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

        :param symbol: The trading symbol (e.g., 'BTCUSDT').
        :param interval: The candle interval (e.g., '1m', '5m', '1h', '1d').
        :param limit: The number of candles to retrieve (max 1000).
        :return: A pandas DataFrame with OHLCV data, indexed by datetime. Returns an empty DataFrame on error.
        """
        endpoint = f"{self.BASE_URL}klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if not data:
                print(f"Warning: No data returned from Binance for symbol {symbol}.")
                return pd.DataFrame()

            # Process data into a DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Select and convert data types
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()
        except ValueError as e:
            print(f"Error processing data: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    print("Running BinanceConnector example...")
    connector = BinanceConnector()
    btc_data = connector.fetch_ohlcv('BTCUSDT', '1h')

    if not btc_data.empty:
        print("\nSuccessfully fetched BTC/USDT 1h data:")
        print(btc_data.head())
        print("\nData Info:")
        btc_data.info()
    else:
        print("\nFailed to fetch data or no data available.")


class KrakenConnector:
    """
    A connector to fetch public market data from Kraken.
    """
    BASE_URL = "https://api.kraken.com/0/public/"

    def fetch_ohlcv(self, pair: str, interval: int = 60, since: int = None) -> (pd.DataFrame, int):
        """
        Fetches historical OHLCV data for a specific pair from Kraken.

        :param pair: The trading pair (e.g., 'XBTUSD').
        :param interval: The candle interval in minutes.
        :param since: Return data since given timestamp ID.
        :return: A tuple containing (DataFrame with OHLCV data, last_timestamp_id).
                 Returns (empty DataFrame, None) on error.
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
                print(f"Kraken API error: {data['error']}")
                return pd.DataFrame(), None

            result = data.get('result', {})
            last_timestamp = result.get('last')
            result_key = next((k for k in result if k != 'last'), None)

            if not result_key:
                print("Could not find pair data in Kraken response.")
                return pd.DataFrame(), None

            raw_ohlc = result[result_key]
            df = pd.DataFrame(raw_ohlc, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])

            # Select and convert data types
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df.set_index('open_time', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)

            return df, last_timestamp

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Kraken: {e}")
            return pd.DataFrame(), None
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing Kraken API response: {e}. Response was: {data}")
            return pd.DataFrame(), None
