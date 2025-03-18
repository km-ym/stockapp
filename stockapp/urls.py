from django.contrib import admin
from django.urls import path
from stock import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.TopView.as_view(), name="top"),
    path("list/", views.stock_list, name="stock_list"),
    path('stock/<int:stock_id>/', views.stock_detail, name="stock_detail"),
]
