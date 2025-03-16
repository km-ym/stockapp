import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from .models import Stock, StockPriceHistory
import logging
logger = logging.getLogger(__name__)

def fetch_stock_data():
    stocks = Stock.objects.all()
    for stock in stocks:
        df = yf.download(stock.ticker, period="7d")  # 1週間分のデータ取得

        # 最新の終値を保存
        stock.current_price = df['Close'].iloc[-1]
        stock.previous_close = df['Close'].iloc[-2] if len(df) > 1 else stock.current_price  # 前日終値
        stock.save()

        # StockPriceHistory オブジェクトをまとめて作成
        history_list = []
        for index, row in df.iterrows():
            history = StockPriceHistory(
                stock=stock,  # Stockのインスタンスをセット
                date=row.name.date(),  # 日付
                open_price=row['Open'],  # 始値
                high_price=row['High'],  # 高値
                low_price=row['Low'],  # 安値
                close_price=row['Close'],  # 終値
                volume=row['Volume'],  # 出来高
            )
            history_list.append(history)

        # バッチでデータベースに保存
        StockPriceHistory.objects.bulk_create(history_list)

def fetch_realtime_price():
    stocks = Stock.objects.all()
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock.ticker)
            realtime_data = ticker.history(period="1d")['Close']
            if realtime_data.empty:
                logger.warning(f"No realtime data found for {stock.ticker}")
                continue
            stock.current_price = realtime_data.iloc[-1]
            stock.save()
        except Exception as e:
            logger.error(f"Failed to fetch real-time data for {stock.ticker}: {e}")