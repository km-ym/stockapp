from apscheduler.schedulers.background import BackgroundScheduler
from .tasks import fetch_stock_data, fetch_realtime_price
from apscheduler.triggers.cron import CronTrigger
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Stock

def start():
    scheduler = BackgroundScheduler()

    # 毎日AM1時にfetch_stock_dataを実行
    scheduler.add_job(fetch_stock_data, CronTrigger(hour=1, minute=0, second=0))
    
    # リアルタイムデータを毎日9時～16時の毎分実行
    scheduler.add_job(fetch_realtime_price, 'cron', hour='9-16', minute='*', second='0')    

    scheduler.start()

# 新規Stock登録時にfetch_stock_dataを実行
@receiver(post_save, sender=Stock)
def stock_post_save(sender, instance, created, **kwargs):
    if created:  # 新規登録された場合
        fetch_stock_data()