import matplotlib
matplotlib.use('Agg')

from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import TemplateView, ListView
from django.http import HttpResponse
import mplfinance as mpf
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf 
import logging  
import pandas as pd
from .models import Stock, StockPriceHistory
from .forms import StockSearchForm, StockCreateForm

logger = logging.getLogger(__name__)  

class TopView(TemplateView):
    template_name = "stock/top.html"

# 株一覧ページ
def stock_list(request):
    stocks = Stock.objects.all()
    search_form = StockSearchForm()
    create_form = StockCreateForm()

    # 検索処理
    if request.method == 'GET' and 'search' in request.GET:
        query = request.GET.get('query', '')
        stocks = Stock.objects.filter(name__icontains=query)
    else:
        stocks = Stock.objects.all()

    # 新規登録処理
    if request.method == 'POST' and 'create' in request.POST:
        create_form = StockCreateForm(request.POST)
        if create_form.is_valid():
            create_form.save()
            return redirect('stock_list')

    return render(request, 'stock/stock_list.html', {
        'stocks': stocks,
        'search_form': search_form,
        'create_form': create_form    
        })


# チャート生成関数
def generate_chart(stock):
    logger.debug(f"Fetching stock data for {stock.ticker}")
    df = yf.download(stock.ticker, period="1y")  # Yahoo Finance からデータ取得
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
    
    # NaNを含む行を削除（エラー回避のため、カラムが存在するかチェック）
    required_columns = ['Open', 'Close', 'High', 'Low']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            return None  # 必要なカラムがない場合はエラー回避

    # NaNを含む行を削除
    df = df.dropna(subset=['Open', 'Close', 'High', 'Low'])

    # 移動平均を計算
    df['MA25'] = df['Close'].rolling(window=25).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA75'] = df['Close'].rolling(window=75).mean()

    # 直近6ヶ月（半年分）のデータを使用
    formatted_date = df.index[-min(len(df), 120):]
    ticker_chart = df.loc[formatted_date]

    last_close = ticker_chart['Close'].iloc[-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    add_plots = [
        mpf.make_addplot(ticker_chart['MA25'], color='yellow', linewidths=0.2),
        mpf.make_addplot(ticker_chart['MA50'], color='tomato', linewidths=0.2),
        mpf.make_addplot(ticker_chart['MA75'], color='lime', linewidths=0.2),
        mpf.make_addplot([last_close] * len(ticker_chart), color='gray', linestyle='dashed', linewidths=1)  # 終値の点線
        ]
    mpf.plot(ticker_chart, type="candle", style="mike", addplot=add_plots, volume=True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return image_base64

# 株の詳細ページ
def stock_detail(request, stock_id):
    stock = get_object_or_404(Stock, id=stock_id)
    
    # 最新の日付の株価データを取得
    latest_history = StockPriceHistory.objects.filter(stock=stock).order_by('-date').first()

    # テンプレートへ渡すコンテキスト
    context = {
        'stock': stock,
        'chart_image': generate_chart(stock),
        'open_price': latest_history.open_price if latest_history else None,
        'high_price': latest_history.high_price if latest_history else None,
        'low_price': latest_history.low_price if latest_history else None,
    }

    return render(request, 'stock/stock_detail.html', context)