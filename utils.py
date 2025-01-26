import pandas as pd
from duckduckgo_search import DDGS
from classes import *
import openbb as obb
import pandas_ta as ta


def search(keyword: str, max_results=100) -> pd.DataFrame:
    search_tool = DDGS()
    results = search_tool.news(
        keywords=keyword, safesearch="off", timelimit="m", max_results=max_results
    )
    df = pd.DataFrame.from_records(results)
    df["date"] = pd.to_datetime(df.date)
    return df.sort_values(by="date", ascending=False)


def get_price_data(
    ticker: Ticker, time_frame: TimeFrame = TimeFrame.DAILY
) -> pd.DataFrame:
    # print(f"ticker is {ticker.name}")
    # print("name of ticker is" + ticker.name)
    df = obb.obb.crypto.price.historical(
        symbol=f"{ticker.name}INR", start_date="2010-01-01"
    )
    df["date"] = pd.to_datetime(df.index)
    df["ticker"] = ticker.name
    df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date")
    if time_frame == TimeFrame.DAILY:
        return df

    interval = "W" if time_frame == TimeFrame.WEEKLY else "M"
    return df.resample(interval).agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def get_news_data(ticker: Ticker, max_articles_per_day: int = 5) -> pd.DataFrame:
    crypto_news_df = search(keyword="Cryptocurrency")
    crypto_news_df["ticker"] = None
    currency_news_df = search(keyword=top_crypto_dict[ticker.name])
    currency_news_df["ticker"] = ticker.name
    df = pd.concat([crypto_news_df, currency_news_df], axis=0)
    df = df.sort_values(by="date").reset_index(drop=True)
    return df


def get_money_supply() -> pd.DataFrame:
    money_df = obb.obb.economy.money_measures(start_date="2010-01-01")
    money_df.month = pd.to_datetime(money_df.month)
    money_df = money_df[["month", "M1", "M2"]]
    money_df.columns = ["date", "m1", "m2"]
    return money_df[["date", "m1", "m2"]].set_index("date")


def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    macd_df = ta.macd(data["close"])
    bbands_df = ta.bbands(data["close"])

    data["ma_50"] = data["close"].rolling(50).mean()
    data["ma_200"] = data["close"].rolling(200).mean()
    data["rsi"] = ta.rsi(data["close"])
    data["macd"], data["macd_signal"], data["macd_histogram"] = (
        macd_df.iloc[:, 0],
        macd_df.iloc[:, 1],
        macd_df.iloc[:, 2],
    )
    data["bb_lower"], data["bb_middle"], data["bb_upper"] = (
        bbands_df.iloc[:, 0],
        bbands_df.iloc[:, 1],
        bbands_df.iloc[:, 2],
    )
    return data


def calculate_50_percent(data: pd.DataFrame, n_weeks: int = 4):
    last_n_weeks = data.tail(n=n_weeks)
    low = last_n_weeks.low.min()
    high = last_n_weeks.high.max()
    return (low + high) / 2, low, high
