from typing import Annotated, List, Literal, TypedDict
from enum import Enum, auto
import pandas as pd
import operator
from consts import *
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AnyMessage


class TimeFrame(Enum):
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()


Ticker = Enum("Ticker", {symbol: symbol for symbol in top_crypto_dict.keys()})
TickerSymbols = Literal[tuple(top_crypto_dict.keys())]


class TickerQuery(BaseModel):
    """Ticker symbol requested by the user"""

    ticker: TickerSymbols = Field(
        description="Ticker symbol for the chosen cryptocurrency. Choose NoCoin if no crypto coin name does not explicitly exist in query",
    )


class FinalReport(BaseModel):
    """Report created by the financial reporter"""

    action: Literal["BUY", "HODL", "SELL"] = Field(
        description="Action to take with the chosen cryptocurrency"
    )
    score: int = Field(
        description="Bullishness market score between 0 (extremely bearish) and 100 (extremely bullish)"
    )
    trend: Literal["UP", "NEUTRAL", "DOWN"] = Field(
        description="Price trend for the chosen cryptocurrency",
    )
    sentiment: Literal["GREED", "NEUTRAL", "FEAR"] = Field(
        description="Sentiment from the news for the chosen cryptocurrency"
    )
    price_predictions: List[float] = Field(
        description="Price predictions for 1, 2, 3 and 4 weeks ahead"
    )
    summary: str = Field(
        description="Summary of the current market conditions (1-3 sentences)"
    )


class AppState(TypedDict):
    user_query: str
    ticker: Ticker
    news: pd.DataFrame
    prices: pd.DataFrame
    price_analyst_report: str
    news_analyst_report: str
    final_report: FinalReport
    final_response: Annotated[list[AnyMessage], operator.add]
    messages: Annotated[list[AnyMessage], operator.add]
