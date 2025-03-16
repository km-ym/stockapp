from django.db import models

class Stock(models.Model):
    name = models.CharField(max_length=100, unique=True)  # 銘柄名
    ticker = models.CharField(max_length=20, unique=True)  # 銘柄
    current_price = models.FloatField(null=True, blank=True) # 現在価格
    previous_close = models.FloatField(null=True, blank=True) # 前日終値
    last_update = models.DateTimeField(auto_now=True) # 更新日

    def price_change(self):
        # 前日比
        if self.previous_close and self.current_price:
            return self.current_price - self.previous_close
        return None
    
    def __str__(self):
        return self.name

class StockPriceHistory(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateField()  # 日付
    open_price = models.FloatField()  # 始値
    high_price = models.FloatField()  # 高値
    low_price = models.FloatField()  # 安値
    close_price = models.FloatField()  # 終値
    volume = models.BigIntegerField()  # 出来高

class Prediction(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateField()  
    predicted_price = models.FloatField()