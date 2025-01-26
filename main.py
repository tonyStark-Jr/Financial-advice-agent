import os
from dotenv import load_dotenv
import streamlit as st
import textwrap
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from openbb import obb
from langchain_core.messages import HumanMessage

load_dotenv()

from utils import *
from consts import *
from classes import *

MODEL = "llama-3.3-70b-versatile"

llm = ChatGroq(temperature=0, model_name=MODEL, api_key=os.environ.get("GROQ_API_KEY"))
obb.obb.account.login(pat=os.environ.get("OPENBB_API_KEY"))
# obb.user.credentials.orats_api_key = os.environ.get("OPENBB_API_KEY")
obb.obb.user.preferences.output_type = "dataframe"


def ticker_extractor(state: AppState):
    """Extracts which is the ticker or cryptocurrency that is being mentioned in the user's query.

    Args:
        state: An AppState object containing the user's query in "user_query".

    Returns:
        A dictionary with the extracted ticker symbol.
    """
    ticker_extractor_llm = llm.with_structured_output(TickerQuery)
    extraction = ticker_extractor_llm.invoke([HumanMessage(state["user_query"])])
    return {"ticker": Ticker[extraction.ticker]}


def news_retriever(state: AppState):
    """Retrieves news for the given ticker.

    Args:
        state: An AppState object containing the ticker in "ticker".

    Returns:
        A dictionary with the news data for the ticker.
    """
    ticker = state["ticker"]
    news_df = get_news_data(ticker)
    return {"news": news_df}


def price_retriever(state: AppState):
    """Retrieves and processes price data for the given ticker.

    Args:
        state: An AppState object containing the ticker in "ticker".

    Returns:
        A dictionary with processed price data for the ticker.
    """
    ticker = state["ticker"]
    price_df = get_price_data(ticker, time_frame=TimeFrame.WEEKLY)
    price_df = add_indicators(price_df)
    price_df = price_df.tail(n=24)

    return {"prices": price_df}


def price_analyst(state: AppState):
    """Analyzes price data and generates a prediction report.

    Args:
        state: An AppState object containing price data in "prices" and the user's query in "user_query".

    Returns:
        A dictionary with the price analysis report.
    """
    price_df = state["prices"]
    weeks_4_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=4)
    weeks_12_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=12)
    weeks_26_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=26)

    money_supply_df = get_money_supply()
    money_supply_text = str(money_supply_df["m2"])

    prompt = f"""You have extensive knowledge of the cryptocurrency market and historical data.
Think step-by-step and focus on the technical indicators.
Use the following weekly close price history and technical indicators for the particular currency:

price history:
{str(price_df[price_df.columns[2:]])}
M2 money supply history:
{money_supply_text}

4 weeks 50% level: {weeks_4_50_percent}
12 weeks 50% level: {weeks_12_50_percent}
26 weeks 50% level: {weeks_26_50_percent}

Predict the next 4 weekly prices based on the data. Put the predictions on separate line.
How certain are you of your predictions? Use a number between 0 (not certain at all) to 10 (sure thing).
What is the overall trend outlook? Explain your predictions in 1-3 sentences.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
    response = llm.invoke([HumanMessage(prompt)])
    return {"price_analyst_report": response.content}


def news_analyst(state: AppState):
    """Analyzes news sentiment and generates a sentiment score.

    Args:
        state: An AppState object containing news data in "news" and the user's query in "user_query".

    Returns:
        A dictionary with the news sentiment analysis report.
    """

    news_df = state["news"]
    news_text = ""
    for date, row in list(news_df.iterrows())[:20]:
        news_text += f"{date}\n{row.title}\n{row.body}\n---\n"

    prompt = f"""Choose a combined sentiment that best represents these news articles:

```
{news_text}
```

Each article is separated by `---`.

Pick a number for the sentiment between 0 and 100 where:

- 0 is extremely bearish
- 100 is extremely bullish

Reply only with the sentiment and a short explanation (1-2 sentences) of why.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
    response = llm.invoke([HumanMessage(prompt)])
    return {"news_analyst_report": response.content}


def financial_reporter(state: AppState):
    """Generates a final financial report based on price and news analyses.

    Args:
        state: An AppState object containing reports from the price analyst ("price_analyst_report")
               and news analyst ("news_analyst_report"), along with the user's query in "user_query".

    Returns:
        A dictionary with the final financial report.
    """
    price_report = state["price_analyst_report"]
    news_report = state["news_analyst_report"]
    prompt = f"""You're a senior cryptocurrency expert that makes extremely accurate predictions
about future prices and trend in the crypto market. You're well versed into technological advancements
and tokenomics of various projects.

You're working with two other agents that have created reports of the current state of the crypto market.

Report of the news analyst:

```
{news_report}
```

Report of the price analyst:

```
{price_report}
```

Based on the provided information, create a final report for the user.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
    final_report_llm = llm.with_structured_output(FinalReport)
    response = final_report_llm.invoke([HumanMessage(prompt)])
    return {"final_report": response}


graph = StateGraph(AppState)


# Node
def reasoner(state):
    query = state["query"]
    messages = state["messages"]
    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with using search, the yahoo finance tool and performing arithmetic on a set of inputs."
    )
    message = HumanMessage(content=query)
    messages.append(message)
    result = [llm_with_tools.invoke([sys_msg] + messages)]
    return {"messages": result}


graph.add_node("reasoner", reasoner)
graph.add_node("ticker_extractor", ticker_extractor)
graph.add_node("news_retriever", news_retriever)
graph.add_node("price_retriever", price_retriever)
graph.add_node("price_analyst", price_analyst)
graph.add_node("news_analyst", news_analyst)
graph.add_node("financial_reporter", financial_reporter)

graph.add_edge("ticker_extractor", "price_retriever")
graph.add_edge("ticker_extractor", "news_retriever")
graph.add_edge("news_retriever", "news_analyst")
graph.add_edge("price_retriever", "price_analyst")
graph.add_edge("price_analyst", "financial_reporter")
graph.add_edge("news_analyst", "financial_reporter")


graph.set_entry_point("ticker_extractor")
graph.set_finish_point("financial_reporter")
app = graph.compile()


# App Title
st.title("🤖 Crypto Financial Advisor")

# User Input Section
st.subheader("Enter Your Query")
user_query = st.text_input("Ask your financial question (e.g., 'BTC price analysis')")

# Process Query Button
if st.button("Get Report"):
    if user_query:
        # Invoke the AI agent
        state = app.invoke({"user_query": user_query})

        # Display Price Analyst Report
        st.subheader("📈 Price Analyst Report")
        price_report = state.get("price_analyst_report", "No report available")
        for line in price_report.split("\n"):
            st.write(textwrap.fill(line, 80))

        # Display News Analyst Report
        st.subheader("📰 News Analyst Report")
        news_report = state.get("news_analyst_report", "No report available")
        for line in news_report.split("\n"):
            st.write(textwrap.fill(line, 80))

        # Display Final Report
        report = state.get("final_report", {})

        if report:
            st.subheader("📊 Final Report")
            # st.write(type(report))
            # print(report)
            st.write(f"**Action:** {report.action}")
            st.write(f"**Score:** {report.score}")
            st.write(f"**Trend:** {report.trend}")
            st.write(f"**Sentiment:** {report.sentiment}")

            st.subheader("🔮 Price Predictions (4 Weeks Ahead)")
            st.write(report.get("price_predictions", "No predictions available"))

            st.subheader("📃 Summary")
            st.write(textwrap.fill(report.summary))
        else:
            st.error("No final report available.")
    else:
        st.error("Please enter a query.")

# Footer
st.markdown("---")
st.markdown("Developed by Prakhar Shukla with ❤️")
