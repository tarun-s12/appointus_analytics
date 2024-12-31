import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter 

nltk.download('vader_lexicon')

def get_connection():
    return sqlite3.connect('appointus.db')

def load_data(query):
    conn = get_connection()
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def analyze_sentiment_ml(comment):
    """
    Perform sentiment analysis using the VADER sentiment analyzer.
    
    Parameters:
        comment (str): The comment to analyze.
        
    Returns:
        str: 'Positive', 'Negative', or 'Neutral' based on the sentiment score.
    """
    if not isinstance(comment, str) or not comment.strip():
        return "Invalid input. Please provide a valid comment."
    
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(comment)
    
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"
    
def provide_insights(provider_id, cluster_model, scaler, cluster_summary, df, features):
    provider_data = df[df['ProviderID'] == provider_id][features].values[0]
    # Predict the cluster
    provider_scaled = scaler.transform([provider_data])
    cluster_label = cluster_model.predict(provider_scaled)[0]
    # Get cluster insights
    cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster_label]
    high_cluster = cluster_summary[cluster_summary["Cluster"] == 0]

    if cluster_label == 2:
        group = "Poor Performing"
    elif cluster_label == 1:
        group = "Avergate Performing"
    else:
        group = "Good Performing"

    st.write(f"\nThe provider belongs to {group} Group")
    st.write("\nCluster Averages:")
    st.write(cluster_info)

    if cluster_label != 0:
        st.write("\nActionable Insights:")
        st.write(f"- Aim to maintain an AvgRating around {high_cluster['AvgRating'].values[0]:.2f}")
        st.write(f"- Reduce ResponseTime to {high_cluster['ResponseTime'].values[0]:.2f} hours or below")
        st.write(f"- Ensure CompletionRate remains above {high_cluster['CompletionRate'].values[0]:.2f}%")
        st.write(f"- Focus on RetentionRate exceeding {high_cluster['RetentionRate'].values[0]:.2f}%")

        st.write("\nSophisticated Suggestions:")
        st.write(f"- Consider offering promotions or discounts to attract new customers and increase AvgRevenue.")
        st.write(f"- Actively request reviews and feedback from satisfied customers to boost your AvgRating.")
        st.write(f"- Streamline booking and communication processes to reduce ResponseTime.")
        st.write(f"- Provide loyalty rewards or incentives for repeat customers to improve RetentionRate.")

    else:
        st.write("You are already in the top performing group!")


def clustering_analysis(provider_id):
    data = {
        "ProviderID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "AvgRating": [4.8, 4.2, 3.9, 4.5, 4.0, 3.5, 4.9, 4.6, 3.8, 4.1],
        "ResponseTime": [1.2, 2.5, 3.0, 1.8, 2.2, 3.5, 1.0, 1.6, 2.8, 2.3],
        "CompletionRate": [95, 90, 85, 93, 88, 80, 98, 94, 83, 89],
        "RetentionRate": [90, 85, 75, 88, 80, 70, 95, 89, 77, 84],
        "AvgRevenue": [5000, 4200, 3900, 4800, 4000, 3500, 5100, 4600, 3800, 4100]
    }

    df = pd.DataFrame(data)

        # Feature selection and scaling
    features = ["AvgRating", "ResponseTime", "CompletionRate", "RetentionRate", "AvgRevenue"]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    
    # Choose optimal number of clusters (e.g., from Elbow Curve, say k=3)
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    kmeans.fit(df_scaled)

    # Add cluster labels to the original DataFrame
    df['Cluster'] = kmeans.labels_

    # Calculate cluster averages for benchmarking
    cluster_summary = df.groupby('Cluster')[features].mean().reset_index()

    # def provide_insights(provider_data, cluster_model, scaler, cluster_summary):
    provide_insights(provider_id,kmeans,scaler,cluster_summary, df, features)

# Provider Visualization Page
def show_provider_visualization(provider_id, provider_name):
    # st.header(f"Welcome, {provider_name}")
    st.title("Welcome to Appointus Analytics Dashboard!")

    # Load provider-specific appointments
    appointments = load_data(f"""
        SELECT * FROM appointments 
        WHERE provider_id = {provider_id}
    """)

    if appointments.empty:
        st.warning("No appointments found for this provider.")
        return

    # Appointment status visualization
    if "Appointment Status Distribution" in selected_analytics:
        status_count = appointments['status'].value_counts().reset_index()
        status_count.columns = ['Status', 'Count']
        st.subheader("Appointment Status Distribution")
        st.plotly_chart(px.pie(status_count, values='Count', names='Status', title='Appointment Status'))

    # Feedback sentiment analysis
    if "Customer Sentiment Analysis" in selected_analytics:
        feedback = load_data(f"""
            SELECT f.comments 
            FROM feedback f 
            INNER JOIN appointments a ON f.appointment_id = a.id 
            WHERE a.provider_id = {provider_id}
        """)

        if not feedback.empty:
            feedback['Sentiment'] = feedback['comments'].apply(analyze_sentiment_ml)
            st.subheader("Customer Sentiment Analysis")

            sentiment_summary = feedback['Sentiment'].value_counts().reset_index()
            sentiment_summary.columns = ['Sentiment', 'Count']
            st.write("Sentiment Summary:", sentiment_summary)

            st.plotly_chart(px.bar(
                sentiment_summary, x='Sentiment', y='Count', 
                title='Sentiment Analysis Distribution', labels={'Sentiment': 'Sentiment', 'Count': 'Count'}
            ))

    # Income over time visualization with added analysis
    if "Income Over Time" in selected_analytics:
        income_data = load_data(f"""
            SELECT appointment_date, SUM(payment_amount) as total_income 
            FROM appointments 
            WHERE provider_id = {provider_id} 
            GROUP BY appointment_date
        """)

        if not income_data.empty:
            st.subheader("Income Over Time")

            # Convert 'appointment_date' to datetime and extract day of the week
            income_data['appointment_date'] = pd.to_datetime(income_data['appointment_date'])
            income_data['day_of_week'] = income_data['appointment_date'].dt.day_name()

            # Calculate a moving average to smooth the data
            income_data['moving_avg'] = income_data['total_income'].rolling(window=7).mean()

            # Plot the raw income data and the moving average
            fig = go.Figure()

            # Raw income line
            fig.add_trace(go.Scatter(
                x=income_data['appointment_date'],
                y=income_data['total_income'],
                mode='lines',
                name='Daily Income',
                line=dict(color='blue', width=2)
            ))

            # Moving average line
            fig.add_trace(go.Scatter(
                x=income_data['appointment_date'],
                y=income_data['moving_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='orange', width=3, dash='dash')
            ))

            fig.update_layout(
                title="Income Over Time (with Moving Average)",
                xaxis_title="Date",
                yaxis_title="Total Income",
                showlegend=True
            )

            st.plotly_chart(fig)

            # Day of the week analysis
            income_by_day = income_data.groupby('day_of_week')['total_income'].sum().reset_index()
            income_by_day = income_data.groupby('day_of_week')['total_income'].sum().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0
                    ).reset_index()

            st.subheader("Income by Day of the Week")
            st.plotly_chart(px.bar(
                income_by_day, x='day_of_week', y='total_income',
                title="Income by Day of the Week", labels={'day_of_week': 'Day', 'total_income': 'Total Income'}
            ))

            # Suggestion based on the pattern observed
            highest_income_day = income_by_day.loc[income_by_day['total_income'].idxmax()]
            lowest_income_day = income_by_day.loc[income_by_day['total_income'].idxmin()]

            st.write(f"Based on the income distribution by day of the week, {highest_income_day['day_of_week']} tends to have the highest income, while {lowest_income_day['day_of_week']} tends to have the lowest/no income. Providers may consider optimizing their schedules and focusing on marketing or promotional efforts on days with lower income.")

            # Outlier detection: using a simple threshold for spikes
            income_data['outlier'] = income_data['total_income'] > income_data['total_income'].mean() + 2 * income_data['total_income'].std()
            outliers = income_data[income_data['outlier']]

            if not outliers.empty:
                st.subheader("Income Peaks (Unusual Highs)")
                
                # Explain what outliers mean
                st.write(
                    "We’ve identified a few dates where your income was significantly higher than usual. These peaks could be due to special events, promotions, or unexpected high demand. "
                    "It's important to understand the factors that caused these spikes, as they might offer valuable insights into what works well for your business."
                )
                
                st.write("Here are the dates with unusually high income:")
                st.write(outliers[['appointment_date', 'total_income']])

                # Suggestions for leveraging income peaks
                st.write(
                    "Consider analyzing the events or circumstances around these dates (e.g., specific promotions, holidays, or increased bookings). "
                    "These insights could help you replicate successful strategies or prepare for similar spikes in the future. "
                    "For example, if a particular promotion led to higher income, consider running similar promotions or events in the future to drive demand."
                )

                # Visual representation of income spikes
                st.plotly_chart(px.scatter(
                    outliers, x='appointment_date', y='total_income',
                    title="Income Peaks on Certain Dates", labels={'appointment_date': 'Date', 'total_income': 'Total Income'}
                ))

    # Geographic distribution of seekers
    if "Geo-Location Analytics" in selected_analytics:
        seeker_locations = load_data(f"""
            SELECT s.location,s.latitude, s.longitude 
            FROM service_seekers s 
            INNER JOIN appointments a ON s.id = a.seeker_id 
            WHERE a.provider_id = {provider_id}
        """)

        if not seeker_locations.empty:
            st.subheader("Geographic Distribution of Service Seekers")

            print(seeker_locations[['latitude', 'longitude']])

            # Plot locations on a map using st.map
            st.map(seeker_locations[['latitude', 'longitude']])

            # Generate insights based on the geographic distribution
            num_seekers = seeker_locations.shape[0]
            st.write(f"Total number of service seekers: {num_seekers}")

            # Count number of unique locations (e.g., cities or regions)
            unique_locations = seeker_locations['location'].nunique()
            st.write(f"Number of unique locations: {unique_locations}")

            # Identify the region with the highest concentration of service seekers
            location_counts = seeker_locations['location'].value_counts().reset_index()
            location_counts.columns = ['Location', 'Count']
            top_location = location_counts.iloc[0]
            st.write(f"The region with the highest concentration of service seekers is {top_location['Location']} with {top_location['Count']} service seekers.")

            # Suggest focus areas based on the distribution
            st.write(
                "Based on the distribution of your service seekers, it may be beneficial to focus marketing efforts or expand services in the areas with higher concentrations of seekers. "
                "If you notice a high number of seekers in certain locations but limited service capacity, this could indicate untapped demand that you could address."
            )
            
        # Optionally, you could also show a cluster analysis or heatmap (advanced) if your dataset includes enough geographic data

    if "Performance Improvement" in selected_analytics:
        st.subheader("Performance Improvement Analysis")
        clustering_analysis(6)


# Main App

st.sidebar.title("Navigation")
st.sidebar.subheader("Select Analytics")
analytics_options = [
    "Appointment Status Distribution", 
    "Customer Sentiment Analysis", 
    "Income Over Time", 
    "Geo-Location Analytics",
    "Performance Improvement"
]
selected_analytics = st.sidebar.multiselect("Choose analytics to display:", analytics_options)

selected_page = "Home"


if selected_page == "Home":
    provider_name = "Fix Speed Plumbers"

    if provider_name:
        providers = load_data("SELECT id, name FROM service_providers")
        if provider_name in providers['name'].values:
            provider_id = providers[providers['name'] == provider_name]['id'].values[0]
            show_provider_visualization(provider_id, provider_name)
        else:
            st.error("Provider not found. Please check the name and try again.")