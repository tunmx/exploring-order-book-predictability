from tardis_dev import datasets
from datetime import datetime
from dateutil.relativedelta import relativedelta
import nest_asyncio
import pandas as pd

nest_asyncio.apply()

today = datetime.now()
if today.day <= 2:
  today = today - relativedelta(months=1)

days = [datetime(today.year, today.month, 1) - i*relativedelta(months=1) for i in range(6)][::-1]

for day in days:
  datasets.download(
      exchange="binance",
      data_types=["book_ticker"],
      from_date=day.strftime("%Y-%m-%d"),
      to_date=(day+relativedelta(days=1)).strftime("%Y-%m-%d"),
      symbols=["BTCUSDT"])

file_paths = [f'datasets/binance_book_ticker_{day.strftime("%Y-%m-%d")}_BTCUSDT.csv.gz' for day in days]
for file_path in file_paths:
  print(file_path)

pd.set_option('display.max_columns', 100)

df = pd.concat([pd.read_csv(file_path, engine='pyarrow') for file_path in file_paths], axis=0)
df.drop(['local_timestamp'], axis=1, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
df.set_index('timestamp', inplace=True)

print(df)