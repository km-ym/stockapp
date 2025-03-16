from django import forms
from .models import Stock

class StockSearchForm(forms.Form):
    query = forms.CharField(
        label='検索',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-2 py-1 border border-gray-300 rounded',
        })
    )
    

class StockCreateForm(forms.ModelForm):
    class Meta:
        model = Stock
        fields = ['name', 'ticker']
        widgets = {
            'name': forms.TextInput(attrs={
                'placeholder': '銘柄名を入力',
                'class': 'w-full px-2 py-1 border border-gray-300 rounded',
                'style': 'width: 100%; max-width: 16rem;',
            }),
            'ticker': forms.TextInput(attrs={
                'placeholder': '銘柄コードを入力（例:7203.T）',
                'class': 'w-full px-2 py-1 border border-gray-300 rounded',
                'style': 'width: 100%; max-width: 16rem;',
            }),
        }