import streamlit as st
import preprocessor, helper, sentiment
import matplotlib.pyplot as plt
import seaborn as sns
from emoji import EMOJI_DATA
import plotly.express as px
import plotly.graph_objects as go

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="📊 WhatsApp Chat Analyzer",
                   layout="wide", page_icon="💬")

sns.set_theme(style="whitegrid")

# ----------------------
# Logo & Title
# ----------------------
st.image("logo.png", width=120)  # Add your logo here
st.title("WhatsApp Chat Analyzer")
st.markdown("Deep Insights into Your WhatsApp Chats 💡")

# ----------------------
# Sidebar
# ----------------------
st.sidebar.header("Upload & Analyze Chat")
uploaded_file = st.sidebar.file_uploader("Choose WhatsApp chat file", type=["txt"])
selected_user = None

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # User Selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Select User for Analysis", user_list)
    show_analysis = st.sidebar.button("🔍 Show Analysis")

    # Sidebar Summary Cards
    total_messages, total_words, total_media, total_links = helper.fetch_stats("Overall", df)
    st.sidebar.markdown("### 📊 File Summary")
    st.sidebar.metric("Messages", total_messages, "✉️")
    st.sidebar.metric("Words", total_words, "📝")
    st.sidebar.metric("Media", total_media, "📷")
    st.sidebar.metric("Links", total_links, "🔗")
else:
    show_analysis = False

# ----------------------
# Analysis Tabs
# ----------------------
if uploaded_file is not None and show_analysis:

    tabs = st.tabs([
        "📄 File Review", "📊 Stats", "📅 Timelines", 
        "📈 Activity", "💬 Wordcloud & Words", "😊 Emoji", "🧠 Sentiment"
    ])

    # ----------------------
    # Tab 1: File Review
    # ----------------------
    with tabs[0]:
        st.header("📄 File Review")
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**File Size:** {len(bytes_data)/1024:.2f} KB")
        st.write(f"**Total Messages:** {df.shape[0]}")
        st.subheader("Preview (First 5 Messages)")
        st.dataframe(df.head(5), use_container_width=True)

    # ----------------------
    # Tab 2: Stats
    # ----------------------
    with tabs[1]:
        st.header("📊 Key Metrics")
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages, "✉️")
        col2.metric("Words", words, "📝")
        col3.metric("Media Shared", num_media_messages, "📷")
        col4.metric("Links Shared", num_links, "🔗")

    # ----------------------
    # Tab 3: Timelines
    # ----------------------
    with tabs[2]:
        st.header("📅 Timeline Analysis")
        monthly = helper.monthly_timeline(selected_user, df)
        daily = helper.daily_timeline(selected_user, df)

        # Interactive Plotly Monthly Timeline
        fig_month = px.line(monthly, x='time', y='message', markers=True, title="Monthly Timeline")
        st.plotly_chart(fig_month, use_container_width=True)

        # Interactive Plotly Daily Timeline
        fig_day = px.line(daily, x='only_date', y='message', markers=True, title="Daily Timeline")
        st.plotly_chart(fig_day, use_container_width=True)

    # ----------------------
    # Tab 4: Activity
    # ----------------------
    with tabs[3]:
        st.header("📈 Activity Analysis")
        busy_day = helper.week_activity_map(selected_user, df)
        busy_month = helper.month_activity_map(selected_user, df)
        user_heatmap = helper.activity_heatmap(selected_user, df)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=busy_day.index, y=busy_day.values, title="Most Busy Day", color=busy_day.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=busy_month.index, y=busy_month.values, title="Most Busy Month", color=busy_month.values, color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weekly Heatmap")
        if user_heatmap is not None and not user_heatmap.empty:
            fig, ax = plt.subplots(figsize=(12,5))
            sns.heatmap(user_heatmap, cmap="YlGnBu", linewidths=.5, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough data for heatmap.")

    # ----------------------
    # Tab 5: Wordcloud & Common Words
    # ----------------------
    with tabs[4]:
        st.header("💬 Wordcloud & Most Common Words")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc is not None:
            fig, ax = plt.subplots(figsize=(10,6))
            ax.imshow(df_wc)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Not enough words for wordcloud.")

        most_common_df = helper.most_common_words(selected_user, df)
        fig = px.bar(most_common_df, x=1, y=0, orientation='h', title="Most Common Words", color=1, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Tab 6: Emoji
    # ----------------------
    with tabs[5]:
        st.header("😊 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig = px.pie(emoji_df.head(), values=1, names=0, title="Top Emojis")
            st.plotly_chart(fig, use_container_width=True)

        if selected_user != "Overall":
            messages = df[df['user']==selected_user]['message']
            emojis = [c for msg in messages for c in msg if c in EMOJI_DATA]
            st.write(" ".join(emojis) if emojis else "No emojis found.")
        else:
            st.warning("Emoji extraction not available for Overall.")

    # ----------------------
    # Tab 7: Sentiment
    # ----------------------
    with tabs[6]:
        st.header("🧠 Sentiment Analysis")
        sentiment_df = sentiment.comprehensive_sentiment_analysis(selected_user, df)
        if sentiment_df is not None and not sentiment_df.empty:
            summary = sentiment.get_sentiment_summary(sentiment_df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Positive", summary['positive_messages'])
            col2.metric("Negative", summary['negative_messages'])
            col3.metric("Neutral", summary['neutral_messages'])
            col4.metric("Avg Polarity", f"{summary['avg_polarity']:.3f}")

            # Sentiment Timeline
            fig = sentiment.plot_sentiment_timeline(sentiment_df, selected_user)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Dangerous Messages
            dangerous_messages = sentiment.get_dangerous_messages(sentiment_df, limit=10)
            if not dangerous_messages.empty:
                with st.expander("🚨 Potentially Concerning Messages (Click to Expand)"):
                    st.dataframe(dangerous_messages, use_container_width=True)
            else:
                st.success("✅ No concerning messages detected!")

            # Keyword Analysis
            keywords = sentiment.get_keyword_analysis(sentiment_df)
            if keywords:
                with st.expander("🔤 Keyword Analysis by Sentiment"):
                    cols = st.columns(3)
                    sentiments = ['positive','negative','neutral']
                    for i, s in enumerate(sentiments):
                        cols[i].write(f"**{s.capitalize()} Keywords:**")
                        if keywords.get(s):
                            for word, count in keywords[s][:10]:
                                cols[i].write(f"• {word} ({count})")
                        else:
                            cols[i].write("No keywords found")
        else:
            st.warning("Not enough data for sentiment analysis.")
