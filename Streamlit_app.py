#Introduction: Streamlit dashbord interface for bearing analysis
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Configure the page settings
st.set_page_config(
    page_title="Bearing Vibrations Analysis Dashboard",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Get the data for the dashboard
file_path = "/Users/luciefourcault/PycharmProjects/pythonProject/Corporate_Project/subgroup_ranges.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)


# Define the plotting functions
# Function for the first algorithm: Detection Process with Control Chart
def plot_range_chart(data, column_name, window_size=20, outlier_threshold=10, initial_points=1000):
    subgroup_ranges = data[column_name]

    initial_subgroup_ranges = data['B1x'][:initial_points]
    range_mean = initial_subgroup_ranges.mean()
    range_std = initial_subgroup_ranges.std()

    ucl = range_mean + 3 * range_std

    outliers = (subgroup_ranges > ucl)

    rolling_outliers = outliers.rolling(window=window_size).sum()

    alerts = (rolling_outliers > outlier_threshold) & (outliers)

    alert_indices = alerts[alerts].index
    for idx in alert_indices:
        print(
            f"Alert: anomalies detected within a rolling window of {window_size} and outlier threshold of {outlier_threshold} at measurement ID {idx}, value: {subgroup_ranges[idx]}")

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(subgroup_ranges.index, subgroup_ranges, marker='o', linestyle='-', color='b', markersize=3, lw=0.2,
            label='Subgroup Range')
    ax.axhline(y=range_mean, color='green', linestyle='--', label='Center Line (Mean Range)')
    ax.axhline(y=ucl, color='blue', linestyle='--', label='UCL')

    ax.scatter(subgroup_ranges.index[outliers], subgroup_ranges[outliers], color='red', label='Outliers', zorder=5)
    ax.scatter(alert_indices, subgroup_ranges[alert_indices], color='purple', label='Alerts', zorder=5)

    ax.set_title(f'Range Control Chart for {column_name}')
    ax.set_xlabel('Measurement ID')
    ax.set_ylabel('Range Value')
    ax.legend()

    return fig


# Function for the second algorithm: Identifying Trends and Detection Process
def plot_range_chart_trends(data, column_name, sd=1.9, window_size=20, initial_points=1000):
    subgroup_ranges = data[column_name]

    initial_subgroup_ranges = data['B1x'][:initial_points]
    range_mean = initial_subgroup_ranges.mean()
    range_std = initial_subgroup_ranges.std()

    ucl = range_mean + 3 * range_std

    outliers = (subgroup_ranges > ucl)

    rolling_mean = subgroup_ranges.rolling(window=window_size).mean()
    rolling_std = subgroup_ranges.rolling(window=window_size).std()

    anomaly_intervals = []
    for i in range(2 * window_size, len(subgroup_ranges)):
        prev_interval_mean = rolling_mean[i - 2 * window_size:i - window_size].mean()
        prev_interval_std = rolling_std[i - 2 * window_size:i - window_size].mean()

        curr_interval_mean = rolling_mean[i - window_size:i].mean()

        if curr_interval_mean > prev_interval_mean + sd * prev_interval_std:
            interval_indices = subgroup_ranges.index[i - window_size:i]
            anomaly_intervals.append(interval_indices)
            for idx in interval_indices:
                print(
                    f"Alert found: anomalies detected within a rolling window of {window_size} and standard deviation comparison of {sd} at measurement ID {idx}, value: {subgroup_ranges[idx]}")

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(subgroup_ranges.index, subgroup_ranges, marker='o', linestyle='-', color='b', markersize=3, lw=0.2,
            label='Subgroup Range')
    ax.axhline(y=range_mean, color='green', linestyle='--', label='Center Line (Mean Range)')
    ax.axhline(y=ucl, color='blue', linestyle='--', label='UCL')

    outliers = subgroup_ranges[(subgroup_ranges > ucl)]
    ax.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=5)

    for anomaly_interval in anomaly_intervals:
        ax.scatter(anomaly_interval, subgroup_ranges.loc[anomaly_interval], color='yellow', zorder=5)

    ax.set_title(f'Range Control Chart for {column_name}')
    ax.set_xlabel('Measurement ID')
    ax.set_ylabel('Range Value')
    ax.legend()

    return fig


# Function for the third algorithm: Detecting Anomalies Based on Rolling Mean
def plot_range_chart_rolling_mean(data, column_name, x=13, window_size=30, initial_points=1000):
    subgroup_ranges = data[column_name]

    initial_subgroup_ranges = data['B1x'][:initial_points]
    range_mean = initial_subgroup_ranges.mean()
    range_std = initial_subgroup_ranges.std()

    ucl = range_mean + 3 * range_std

    rolling_mean = subgroup_ranges.rolling(window=window_size).mean()

    consecutive_increases = 0
    anomaly_intervals = []

    for i in range(window_size, len(subgroup_ranges)):
        if rolling_mean[i] > rolling_mean[i - 1]:
            consecutive_increases += 1
        else:
            consecutive_increases = 0

        if consecutive_increases >= x:
            interval_indices = subgroup_ranges.index[i - x + 1:i + 1]
            anomaly_intervals.append(interval_indices)
            for idx in interval_indices:
                print(
                    f"Alert found: rolling mean increased {x} times in a row at measurement ID {idx}, value: {subgroup_ranges[idx]}")

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(subgroup_ranges.index, subgroup_ranges, marker='o', linestyle='-', color='b', markersize=3, lw=0.2,
            label='Subgroup Range')
    ax.axhline(y=range_mean, color='green', linestyle='--', label='Center Line (Mean Range)')
    ax.axhline(y=ucl, color='blue', linestyle='--', label='UCL')

    outliers = subgroup_ranges[(subgroup_ranges > ucl)]
    ax.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=5)

    for anomaly_interval in anomaly_intervals:
        ax.scatter(anomaly_interval, subgroup_ranges.loc[anomaly_interval], color='yellow', zorder=5)

    ax.set_title(f'Range Control Chart for {column_name}')
    ax.set_xlabel('Measurement ID')
    ax.set_ylabel('Range Value')
    ax.legend()

    return fig


# Calculate the number of anomalies and other statistics
def calculate_anomalies(data, columns, initial_points=1000):
    total_anomalies = 0
    bearing_anomalies = {column: 0 for column in columns}

    for column in columns:
        subgroup_ranges = data[column]

        initial_subgroup_ranges = data['B1x'][:initial_points]
        range_mean = initial_subgroup_ranges.mean()
        range_std = initial_subgroup_ranges.std()

        ucl = range_mean + 3 * range_std

        outliers = (subgroup_ranges > ucl)
        bearing_anomalies[column] = outliers.sum()
        total_anomalies += outliers.sum()

    best_bearing = min(bearing_anomalies, key=bearing_anomalies.get)

    return total_anomalies, range_mean, range_std, best_bearing


# Sidebar for user inputs
with st.sidebar:
    st.title('⚙️ Bearings Vibration: User Input Parameters')
    page = st.radio('Navigation', ['Welcome Page', 'Algorithm Analysis'])
    if page == 'Algorithm Analysis':
        Selectbox_Algorithm = st.selectbox(
            'Select Algorithm to analyze',
            [
                'Detection Process with Control Chart',
                'Identifying Trends and Detection Process',
                'Detecting Anomalies Based on Rolling Mean'
            ]
        )

# Main content
st.title("Bearings Vibration Analysis Dashboard")

if page == 'Welcome Page':
    st.markdown("<h1 style='color: #0000FF;'>Welcome to our Bearing Vibration Analysis Application</h1>",
                unsafe_allow_html=True)
    st.write(
        "For the purpose of our Capstone Project, we had to collaborate with the Company Indra-Minsait to develop a robust predictive maintenance system for bearing failure detection using vibration data from industrial machines")
    st.write("")
    st.markdown("### This project has been developed by Group 18 including the following Team Members:")
    st.markdown(
        """
        - Lucie Fourcault
        - Víctor González
        - Vicente Llobell
        - Jacopo Marzotto
        """
    )

elif page == 'Algorithm Analysis':
    columns = ['B1x', 'B1y', 'B2x', 'B2y', 'B3x', 'B3y', 'B4x', 'B4y']

    total_anomalies, range_mean, range_std, best_bearing = calculate_anomalies(df, columns)

    # Display widgets
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Anomalies", total_anomalies)
    col2.metric("Overall Mean", round(range_mean, 2))
    col3.metric("Overall Std Dev", round(range_std, 2))
    col1.metric("Best Performing Bearing", best_bearing)

    if Selectbox_Algorithm == 'Detection Process with Control Chart':
        st.subheader("Detection Process with Control Chart")
        st.markdown(
            "A control chart is a simple graph used to monitor how a process changes over time. It shows data points plotted in time order, with a central line (CL) for the average, an upper line for the upper control limit (UCL), and a lower line for the lower control limit (LCL).")

        outlier_threshold = st.slider("Select Outlier Threshold", min_value=1, max_value=100, value=10, step=1)

        # Plotting the charts in a 3-column layout
        for i in range(0, len(columns), 2):
            st.write(f"### Bearing {i // 2 + 1} analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Bearing {i // 2 + 1}x")
            with col2:
                fig1 = plot_range_chart(df, columns[i], outlier_threshold=outlier_threshold)
                st.pyplot(fig1)
            with col3:
                fig2 = plot_range_chart(df, columns[i + 1], outlier_threshold=outlier_threshold)
                st.pyplot(fig2)

    elif Selectbox_Algorithm == 'Identifying Trends and Detection Process':
        st.subheader("Identifying Trends and Detection Process")
        st.markdown(
            "This method uses rolling means and standard deviations to identify trends and detect anomalies. When a rolling mean of the range of values within subgroups shows a significant upward or downward trend, an anomaly is detected.")

        rolling_window_size = st.slider("Select Rolling Window Size", min_value=1, max_value=100, value=20, step=1)
        standard_deviation = st.slider("Select Standard Deviation", min_value=1.0, max_value=10.0, value=1.9, step=0.1)

        # Plotting the charts in a 3-column layout
        for i in range(0, len(columns), 2):
            st.write(f"### Bearing {i // 2 + 1} analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Bearing {i // 2 + 1}x")
            with col2:
                fig1 = plot_range_chart_trends(df, columns[i], sd=standard_deviation, window_size=rolling_window_size)
                st.pyplot(fig1)
            with col3:
                fig2 = plot_range_chart_trends(df, columns[i + 1], sd=standard_deviation,
                                               window_size=rolling_window_size)
                st.pyplot(fig2)

    elif Selectbox_Algorithm == 'Detecting Anomalies Based on Rolling Mean':
        st.subheader("Detecting Anomalies Based on Rolling Mean")
        st.markdown(
            "This method focuses on the rolling mean of the range of values within subgroups. If the rolling mean increases consecutively for a defined number of intervals, an anomaly is flagged.")

        rolling_window_size = st.slider("Select Rolling Window Size", min_value=1, max_value=100, value=30, step=1)
        x_parameter = st.slider("Select 'x' parameter", min_value=1, max_value=20, value=13, step=1)

        # Plotting the charts in a 3-column layout
        for i in range(0, len(columns), 2):
            st.write(f"### Bearing {i // 2 + 1} analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Bearing {i // 2 + 1}x")
            with col2:
                fig1 = plot_range_chart_rolling_mean(df, columns[i], x=x_parameter, window_size=rolling_window_size)
                st.pyplot(fig1)
            with col3:
                fig2 = plot_range_chart_rolling_mean(df, columns[i + 1], x=x_parameter, window_size=rolling_window_size)
                st.pyplot(fig2)
