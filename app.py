import streamlit as st
from data import get_stock_data, get_news, get_ticker_from_company
from model import vectorize_text, label_stock_movement, train_model, predict_sentiment

# -----------------------------
# CONFIGURATION
# -----------------------------
from dotenv import load_dotenv
import os

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
START_DATE = "2025-03-10"
END_DATE = "2025-04-01"

# -----------------------------
# 📊 STREAMLIT APP
# -----------------------------
st.title("📉 NewsFlash Finance")
st.subheader("Using NLP to Gauge Stock Reactions to Breaking News")

company_name = st.text_input("Enter a company name (e.g., Apple, Tesla, Nvidia)")

if company_name:
    with st.spinner("Fetching data..."):
        try:
            ticker = get_ticker_from_company(company_name)
            if not ticker:
                st.error("Could not find a valid stock ticker for this company.")
            else:
                st.write(f"Matched ticker: `{ticker}`")

                stock_df = get_stock_data(ticker, START_DATE, END_DATE)
                stock_df = label_stock_movement(stock_df)
                
                headlines = get_news(NEWS_API_KEY, company_name, START_DATE, END_DATE)

                if not headlines:
                    st.warning("No headlines found — using fallback data.")
                    headlines = [
                        f"{company_name} stock surges after strong earnings",
                        f"{company_name} beats expectations and stock climbs",
                        "Market falls amid global economic fears",
                        "Recession worries push markets down"
                    ]
                    stock_labels = [1, 1, 0, 0]
                else:
                    stock_labels = stock_df['Movement'].astype(int).tolist()
                    min_len = min(len(headlines), len(stock_labels))
                    headlines = headlines[:min_len]
                    stock_labels = stock_labels[:min_len]

                X, vectorizer = vectorize_text(headlines)
                model = train_model(X, stock_labels)

                st.success("Model trained")

                # Headline input (always above chart)
                user_input = st.text_input("Enter a breaking news headline")

                if user_input:
                    pred = predict_sentiment([user_input], model, vectorizer)
                    st.write("Predicted Market Reaction:", "To the Moon" if pred[0] else "Down to Earth")

                # Display stock price chart regardless of prediction input
                st.subheader(f"Stock Price Chart for {company_name} ({ticker})")
                st.line_chart(stock_df['Close'])

        except Exception as e:
            st.error(f"Error: {str(e)}")
