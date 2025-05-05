import streamlit as st
from data import get_stock_data, get_news, get_ticker_from_company
from model import vectorize_text, label_stock_movement, train_model, predict_sentiment

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Set date range
END_DATE = datetime.today().date().isoformat()
START_DATE = (datetime.today() - timedelta(days=29)).date().isoformat()

# Load API key
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Streamlit UI
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

                # Fetch stock and label movement
                stock_df = get_stock_data(ticker, START_DATE, END_DATE)
                print(stock_df)
                print(stock_df.dtypes)

                stock_df = label_stock_movement(stock_df)

                # Fetch headlines
                headlines = get_news(NEWS_API_KEY, company_name, START_DATE, END_DATE)

                # Align headlines with movement labels
                stock_labels = stock_df['Movement'].astype(int).tolist()
                min_len = min(len(headlines), len(stock_labels))
                headlines = headlines[:min_len]
                stock_labels = stock_labels[:min_len]

                # Train model
                X, vectorizer = vectorize_text(headlines)
                model = train_model(X, stock_labels)

                st.success("Model trained")

                # User input for new headline
                user_input = st.text_input("Enter a breaking news headline")

                if user_input:
                    preds, probs = predict_sentiment([user_input], model, vectorizer)
                    confidence = probs[0][list(model.classes_).index(preds[0])] * 100

                    reaction_map = {1: "To the Moon 🚀", 0: "Meh 😐", -1: "Down to Earth 📉"}
                    reaction = reaction_map.get(preds[0], "Unknown")

                    st.write("Predicted Market Reaction:", reaction)
                    st.write(f"Confidence: **{confidence:.2f}%**")

                # Show closing price chart
                st.subheader(f"Stock Price Chart for {company_name} ({ticker})")
                st.line_chart(stock_df['Close'])

        except Exception as e:
            st.error(f"Error: {str(e)}")
