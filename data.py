import requests
import pandas as pd
import time
import datetime


class Data:
    """
        :param periodType: The type of period to show. Valid values are day, month, year, or ytd (year to date). Default is day.
        :param frequencyType: The type of frequency with which a new candle is formed. (day: minute, month: daily, weekly, year: daily, weekly, monthly, ytd: daily, weekly)
        :param frequency: The number of the frequencyType to be included in each candle. (minute: 1, 5, 10, 15, 30, daily: 1, weekly, 1, monthly: 1)
        :param end_date: End date as milliseconds since epoch. If startDate and endDate are provided, period should not be provided. Default is previous trading day.
        :param start_date: Start date as milliseconds since epoch. If startDate and endDate are provided, period should not be provided.
    """

    def __init__(self, periodType, frequencyType, frequency, needExtendedHoursData, ticker, period=None, start_date=None, end_date=None):
        self.endpoint = r'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(
            ticker)
        if start_date and end_date is None:
            self.payload = {'apikey': config.API_KEY,
                            'periodType': periodType,
                            'frequencyType': frequencyType,
                            'frequency': frequency,
                            'period': period,
                            'needExtendedHoursData': needExtendedHoursData
                            }
        else:
            self.payload = {'apikey': 'YH6LPT14YIMMVFOTP0TH7ZTVSWBL3JTD',
                            'periodType': periodType,
                            'frequencyType': frequencyType,
                            'frequency': frequency,
                            'needExtendedHoursData': needExtendedHoursData,
                            'startDate': start_date,
                            'endDate': end_date
                            }

    def get_content(self):
        content = requests.get(url=self.endpoint, params=self.payload)
        data = content.json()
        return data

    def get_prices_formatted(self):
        prices = {'date': [], 'open': [], 'high': [], 'low': [], 'close': []}
        content = requests.get(url=self.endpoint, params=self.payload)
        data = content.json()
        for i in data['candles']:
            prices['date'].append(i['datetime'])
            prices['open'].append(i['open'])
            prices['high'].append(i['high'])
            prices['low'].append(i['low'])
            prices['close'].append(i['close'])
        return prices

    def get_price_non_epoch(self, prices_df):
        assert isinstance(prices_df, pd.DataFrame)
        prices_df['date'] = prices_df['date'].apply(
            lambda x: time.strftime("%d %b %Y %H:%M:%S", time.localtime(x/1000)))
        return prices_df

    def get_train_test_split(self, prices, split_on=0.8):
        train_end = int(len(prices['date']) * 0.8)
        train_set = {}
        test_set = {}
        for key in prices.keys():
            train_set[key] = prices[key][:train_end]
            test_set[key] = prices[key][train_end:]
        return train_set, test_set
