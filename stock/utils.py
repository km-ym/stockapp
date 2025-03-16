# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler
# from datetime import timedelta
# from django.utils import timezone
# import matplotlib.pyplot as plt
# import io
# import base64
# from .models import StockPriceHistory, Prediction

# # LSTM の入力形式に変換
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i + seq_length])
#         y.append(data[i + seq_length])
#     return np.array(X), np.array(y)

# # LSTM モデルの定義
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1])  
#         return out

# # 予測データ取得
# def predict_stock_prices(stock):
#     print("データ取得開始")
#     five_years_ago = timezone.now() - timedelta(days=10*365)
#     history = StockPriceHistory.objects.filter(
#         stock=stock,
#         date__gte=five_years_ago
#         ).order_by('date')

#     close_prices = np.array([h.close_price for h in history]).reshape(-1, 1)
    
#     print("データ取得完了")

#     # データの正規化
#     min_val = np.min(close_prices)
#     max_val = np.max(close_prices)
#     scaled_data = (close_prices - min_val) / (max_val - min_val)

#     seq_length = 30
#     X, y = create_sequences(scaled_data, seq_length)

#     print("データの正規化完了")

#     # PyTorch のテンソルに変換
#     X_train = torch.FloatTensor(X)
#     y_train = torch.FloatTensor(y)
#     print("テンソルに変換完了")

#     # モデルの設定
#     input_size = 1
#     hidden_size = 50
#     num_layers = 2
#     output_size = 1

#     model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
#     model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#     print("モデルの設定完了")

#     # 損失関数と最適化
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     print("損失関数と最適化完了")

#     # モデルの学習
#     num_epochs = 70
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         output = model(X_train)
#         loss = criterion(output, y_train)
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 5 == 0:  # 5エポックごとに出力
#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#     print("モデルの学習完了")

#     # 予測
#     model.eval()
#     future_predictions = []
#     with torch.no_grad():
#         input_seq = X_train[-1].unsqueeze(0)

#         for _ in range(7):
#             pred = model(input_seq).item()
#             future_predictions.append(pred)
#             new_input = torch.FloatTensor([[[pred]]])
#             input_seq = torch.cat((input_seq[:, 1:, :], new_input), dim=1)
#     print("予測完了")

#     # 予測データをスケールを元に戻す
#     future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

#     # 予測データをDBに保存
#     future_dates = pd.date_range(start=history.last().date, periods=8, freq='B')[1:]

#     for i, date in enumerate(future_dates):
#         Prediction.objects.update_or_create(
#             stock=stock, date=date,
#             defaults={'predited_price': future_predictions[i][0]}
#         )

#     return future_predictions.flatten()
#     print("保存完了")

# # 予測チャートの生成
# def generate_prediction_chart(stock):
#     history = StockPriceHistory.objects.filter(stock=stock).order_by('date')

#     if history.count() < 30:
#         return None  # データが足りない場合は表示しない

#     actual_prices = [h.close_price for h in history][-100:]
#     actual_dates = [h.date for h in history][-100:]

#     predictions = Prediction.objects.filter(stock=stock).order_by('date')
#     if not predictions.exists():
#         return None  # 予測データがなければ何もしない

#     future_prices = [p.predited_price for p in predictions]
#     future_dates = [p.date for p in predictions]

#     combined_dates = actual_dates + future_dates
#     combined_prices = actual_prices + future_prices

#     # プロット
#     plt.figure(figsize=(10, 5))
#     plt.plot(combined_dates, combined_prices, label="Predicted Price", linestyle='dashed', color='red')
#     plt.plot(actual_dates, actual_prices, label="Actual Price", color='blue')

#     plt.title(f"Stock Price Prediction for {stock.name}")
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.legend()
#     plt.grid(True)

#     # 画像を Base64 エンコード
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches='tight')
#     buf.seek(0)
#     plt.close()

#     image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#     buf.close()
    
#     return image_base64