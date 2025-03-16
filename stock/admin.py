from django.contrib import admin
from .models import Stock, StockPriceHistory

class StockAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'ticker')
    search_fields = ('name', 'ticker')

admin.site.register(Stock, StockAdmin)
admin.site.register(StockPriceHistory)