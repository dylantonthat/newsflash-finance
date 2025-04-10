# Fetch news & stock info

import yfinance as yf
import requests

def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Open', 'Close']]

def get_news(api_key, company_name, from_date, to_date):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={company_name}&"
        f"from={from_date}&"
        f"to={to_date}&"
        f"language=en&"
        f"sortBy=popularity&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()
    print(data)
    articles = data.get("articles", [])
    return [a["title"] for a in articles]

def get_ticker_from_company(company_name):
    """
    Given a company name like 'Tesla', returns the stock ticker like 'TSLA'.
    Uses Yahoo Finance autocomplete API.
    """
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers)
        result = response.json()

        if "quotes" in result:
            for item in result["quotes"]:
                if item.get("quoteType") in ["EQUITY", "ETF"] and item.get("symbol"):
                    return item["symbol"]
    except Exception as e:
        print(f"Error fetching ticker: {e}")
    
    return None
