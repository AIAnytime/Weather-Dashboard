import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from windrose import WindroseAxes
import matplotlib.cm as cm

# ------------------------------------------------------------------
# PAGE CONFIG & STYLES
# ------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Weather Dashboard", page_icon="☁️")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATA LOADING & ENRICHMENT
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Weather_EDA_V2.0.csv")

    # Dates
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    # Core temporal fields
    df['Year']      = df['Date'].dt.year
    df['Month']     = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['Day']       = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['WeekNum']   = df['Date'].dt.isocalendar().week

    # Season mapping (STANDARD seasonal mapping)
    season_map_std = {12:'Winter',1:'Winter',2:'Winter',
                      3:'Spring',4:'Spring',5:'Spring',
                      6:'Summer',7:'Summer',8:'Summer',
                      9:'Autumn',10:'Autumn',11:'Autumn'}
    df['Season'] = df['Month'].map(season_map_std)

    # (If you want original custom mapping instead, uncomment and override)
    # season_mapping_custom = {1:'Summer',2:'Summer',3:'Autumn',4:'Autumn',5:'Autumn',
    #                          6:'Winter',7:'Winter',8:'Winter',9:'Spring',10:'Spring',
    #                          11:'Spring',12:'Summer'}
    # df['Season'] = df['Month'].map(season_mapping_custom)

    # Flags
    df['IsRainy']   = df['Rainfall'] > 0
    df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

    # Core derived metrics
    df['AvgTemp']        = (df['MinTemp'] + df['MaxTemp']) / 2
    df['AvgHumidity']    = (df['Humidity9am'] + df['Humidity3pm']) / 2
    df['HumidityVariation'] = (df['Humidity3pm'] - df['Humidity9am']).abs()

    # Temperature Stability Index (TSI)  (handle division safety)
    denom = (df['MaxTemp'].abs() + df['MinTemp'].abs())
    df['TSI'] = 1 - (df['MaxTemp'] - df['MinTemp']).abs() / denom.replace(0, np.nan)
    df['TSI'] = df['TSI'].clip(0, 1)

    # Avg Wind Speed if columns exist
    if {'WindSpeed9am', 'WindSpeed3pm'}.issubset(df.columns):
        df['AvgWindSpeed'] = df[['WindSpeed9am', 'WindSpeed3pm']].mean(axis=1)
    elif 'WindSpeed3pm' in df.columns:
        df['AvgWindSpeed'] = df['WindSpeed3pm']
    elif 'WindSpeed9am' in df.columns:
        df['AvgWindSpeed'] = df['WindSpeed9am']
    else:
        df['AvgWindSpeed'] = np.nan

    # Rolling metrics (7d) – for optional plots
    df['AvgTemp_7d']  = df['AvgTemp'].rolling(7, min_periods=3).mean()
    df['Rain_7d_Sum'] = df['Rainfall'].rolling(7, min_periods=3).sum()

    # OPTIONAL anomalies / extremes (commented out - enable if needed)
    # df['Temp_Z']       = (df['AvgTemp'] - df['AvgTemp'].mean()) / df['AvgTemp'].std()
    # df['Temp_Anomaly'] = df['Temp_Z'].abs() > 2
    # df['ExtremeRain']  = df['Rainfall'] > df['Rainfall'].quantile(0.95)
    # df['ExtremeHeat']  = df['MaxTemp'] > df['MaxTemp'].quantile(0.95)

    return df

# ------------------------------------------------------------------
# AGGREGATION FUNCTIONS (UPDATED for KPI completeness)
# ------------------------------------------------------------------
def aggregate_by_week(df):
    week = (df.groupby(['Year','WeekNum'])
              .agg(MinTemp=('MinTemp','min'),
                   MaxTemp=('MaxTemp','max'),
                   AvgTemp=('AvgTemp','mean'),
                   Rainfall=('Rainfall','sum'),
                   RainfallPerDay=('Rainfall','mean'),
                   AvgHumidity=('AvgHumidity','mean'),
                   HumidityVar=('HumidityVariation','mean'),
                   TSI=('TSI','mean'),
                   AvgWindSpeed=('AvgWindSpeed','mean'),
                   RainyDays=('IsRainy','sum'),
                   TotalDays=('IsRainy','count'),
                   Date=('Date','first'))
              .reset_index())
    week['RainyDayPct'] = 100 * week['RainyDays'] / week['TotalDays']
    week['TimeLabel']   = week['Date'].dt.strftime('%Y-W%W')
    return week

def aggregate_by_month(df):
    month = (df.groupby(['Year','Month'])
               .agg(MinTemp=('MinTemp','min'),
                    MaxTemp=('MaxTemp','max'),
                    AvgTemp=('AvgTemp','mean'),
                    Rainfall=('Rainfall','sum'),
                    RainfallPerDay=('Rainfall','mean'),
                    AvgHumidity=('AvgHumidity','mean'),
                    HumidityVar=('HumidityVariation','mean'),
                    TSI=('TSI','mean'),
                    AvgWindSpeed=('AvgWindSpeed','mean'),
                    RainyDays=('IsRainy','sum'),
                    TotalDays=('IsRainy','count'),
                    Date=('Date','first'))
               .reset_index())
    month['RainyDayPct']  = 100 * month['RainyDays'] / month['TotalDays']
    month['MonthYear']    = month['Date'].dt.strftime('%b %Y')
    month['TimeLabel']    = month['MonthYear']
    return month

def aggregate_by_season(df):
    season = (df.groupby(['Year','Season'])
                .agg(AvgTemp=('AvgTemp','mean'),
                     Rainfall=('Rainfall','sum'),
                     RainfallPerDay=('Rainfall','mean'),
                     AvgHumidity=('AvgHumidity','mean'),
                     HumidityVar=('HumidityVariation','mean'),
                     TSI=('TSI','mean'),
                     AvgWindSpeed=('AvgWindSpeed','mean'),
                     RainyDays=('IsRainy','sum'),
                     TotalDays=('IsRainy','count'))
                .reset_index())
    season['RainyDayPct'] = 100 * season['RainyDays'] / season['TotalDays']
    return season

# ------------------------------------------------------------------
# VISUALIZATION HELPERS
# ------------------------------------------------------------------
def plot_wind_polar(wind_data, wind_dir_column, wind_speed_column, title="Wind Direction & Speed"):
    valid = wind_data.dropna(subset=[wind_dir_column, wind_speed_column])
    if len(valid) < 10:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.text(0.5, 0.5, 'Insufficient wind data', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Optional: true wind rose (uncomment to switch)
    # fig = plt.figure(figsize=(6,6))
    # ax = WindroseAxes.from_ax(fig=fig)
    # ax.bar(valid[wind_dir_column], valid[wind_speed_column], normed=True, opening=0.8, edgecolor='white')
    # ax.set_legend()
    # return fig

    counts = valid[wind_dir_column].value_counts()
    speed_by_dir = valid.groupby(wind_dir_column)[wind_speed_column].mean()
    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(counts.index, counts.values, alpha=0.8)
    speeds = [speed_by_dir.get(d, 0) for d in counts.index]
    norm = plt.Normalize(min(speeds), max(speeds) if max(speeds)>0 else 1)
    colors = plt.cm.viridis(norm(speeds))
    for b, c in zip(bars, colors): b.set_color(c)
    ax.set_title(title)
    ax.set_xlabel("Wind Direction")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)
    if speeds:
        ax.text(0.99, 0.02,
                f"Speed Range: {min(speeds):.1f}-{max(speeds):.1f}",
                ha='right', va='bottom', transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    return fig

def plot_temp_trends(data, time_col):
    fig = go.Figure()
    if 'AvgTemp' in data:
        fig.add_trace(go.Scatter(x=data[time_col], y=data['AvgTemp'],
                                 mode='lines', name='Avg Temp', line=dict(width=2)))
    if {'MinTemp','MaxTemp'}.issubset(data.columns):
        fig.add_trace(go.Scatter(x=data[time_col], y=data['MinTemp'], mode='lines',
                                 name='Min Temp', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=data[time_col], y=data['MaxTemp'], mode='lines',
                                 name='Max Temp', line=dict(width=1)))
    if 'AvgTemp_7d' in data.columns and data['AvgTemp_7d'].notna().sum() > 5:
        fig.add_trace(go.Scatter(x=data[time_col], y=data['AvgTemp_7d'],
                                 mode='lines', name='7d Avg Temp', line=dict(dash='dot', width=2)))
    fig.update_layout(title='Temperature Trends',
                      xaxis_title='Time Period', yaxis_title='Temperature (°C)',
                      height=420, margin=dict(l=10,r=10,t=50,b=10),
                      legend=dict(orientation='h', y=1.03, x=1, xanchor='right'))
    return fig

def plot_rainfall(data, time_col):
    fig = px.bar(data, x=time_col, y='Rainfall',
                 title='Rainfall (Sum)', labels={'Rainfall':'Rainfall (mm)'},
                 color='Rainfall', color_continuous_scale='Blues')
    if 'Rain_7d_Sum' in data.columns and data['Rain_7d_Sum'].notna().sum() > 5:
        fig2 = px.line(data, x=time_col, y='Rain_7d_Sum')
        for t in fig2.data:
            t.name = '7d Rain Sum'
            t.line.width = 2
            fig.add_trace(t)
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10), coloraxis_showscale=False)
    return fig

def plot_humidity_boxplot(df, group_col):
    hdf = df[[group_col, 'Humidity9am', 'Humidity3pm']].melt(
        id_vars=group_col, value_vars=['Humidity9am','Humidity3pm'],
        var_name='Time', value_name='Humidity')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x=group_col, y='Humidity', hue='Time', data=hdf,
                palette='pastel', ax=ax)
    ax.set_title('Humidity Variation Distribution')
    ax.set_ylabel('Humidity (%)')
    for label in ax.get_xticklabels(): label.set_rotation(45)
    ax.legend(title='Time of Day')
    fig.tight_layout()
    return fig

def detect_outliers(df, cols):
    out = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            Q1, Q3 = df[c].quantile([0.25,0.75])
            IQR = Q3-Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            out[c] = ((df[c]<lb)|(df[c]>ub)).sum()
    return out

# ------------------------------------------------------------------
# SPELL METRICS (optional)
# ------------------------------------------------------------------
def longest_consecutive(mask_series):
    run = max_run = 0
    for v in mask_series:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def spell_metrics(df):
    dry = (df['Rainfall'] == 0)
    wet = (df['Rainfall'] > 0)
    return {
        "Longest Dry Spell": longest_consecutive(dry),
        "Longest Wet Spell": longest_consecutive(wet)
    }

# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
def main():
    st.title("Weather Dashboard & EDA Tool")

    with st.spinner("Loading weather data..."):
        df = load_data()

    tab_dashboard, tab_eda = st.tabs(["Dashboard", "EDA & Analysis"])

    # ===================== DASHBOARD TAB ============================
    with tab_dashboard:
        st.sidebar.header("Filters")

        time_agg = st.sidebar.radio("Time Aggregation",
                                    ["Daily","Weekly","Monthly"], index=0)

        # Year filter
        years = sorted(df['Year'].dropna().unique())
        selected_years = st.sidebar.multiselect("Select Years",
                                                years,
                                                default=years[-5:] if len(years)>5 else years)
        filtered = df[df['Year'].isin(selected_years)] if selected_years else df.copy()

        # Season filter
        seasons = sorted(filtered['Season'].dropna().unique())
        selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=[])
        if selected_seasons:
            filtered = filtered[filtered['Season'].isin(selected_seasons)]

        # Month filter (Daily / Weekly)
        if time_agg in ("Daily","Weekly"):
            months = sorted(filtered['Month'].unique())
            month_names = [datetime(2000,m,1).strftime('%B') for m in months]
            selected_month_names = st.sidebar.multiselect("Select Months", month_names, default=[])
            if selected_month_names:
                sel_month_nums = [datetime.strptime(m,'%B').month for m in selected_month_names]
                filtered = filtered[filtered['Month'].isin(sel_month_nums)]

        # Rain filter
        rain_filter = st.sidebar.radio("Precipitation Filter",
                                       ["All Days","Rainy Days Only","Dry Days Only"], index=0)
        if rain_filter == "Rainy Days Only":
            filtered = filtered[filtered['IsRainy']]
        elif rain_filter == "Dry Days Only":
            filtered = filtered[~filtered['IsRainy']]

        # Wind direction filter
        if 'WindDir3pm' in filtered.columns:
            wind_dirs = sorted(filtered['WindDir3pm'].dropna().unique())
            selected_wdirs = st.sidebar.multiselect("Wind Direction (3pm)", wind_dirs, default=[])
            if selected_wdirs:
                filtered = filtered[filtered['WindDir3pm'].isin(selected_wdirs)]

        # (Optional) Weekend filter
        weekend_opt = st.sidebar.selectbox("Weekend Filter",
                                           ["All","Weekdays Only","Weekends Only"], index=0)
        if weekend_opt == "Weekdays Only":
            filtered = filtered[~filtered['IsWeekend']]
        elif weekend_opt == "Weekends Only":
            filtered = filtered[filtered['IsWeekend']]

        # Apply aggregation
        if time_agg == "Weekly":
            display_df = aggregate_by_week(filtered)
            time_col = 'TimeLabel'
        elif time_agg == "Monthly":
            display_df = aggregate_by_month(filtered)
            time_col = 'TimeLabel'
        else:
            display_df = filtered.copy()
            display_df['TimeLabel'] = display_df['Date']
            time_col = 'TimeLabel'

        if len(display_df)==0:
            st.warning("No data for selected filters.")
            return

        # ---------------- KPI ROWS ----------------
        st.subheader("Key Weather Metrics")

        k1, k2, k3, k4 = st.columns(4)
        avg_temp = display_df['AvgTemp'].mean()
        k1.metric("Average Temperature", f"{avg_temp:.1f}°C")

        rainy_days = display_df['IsRainy'].sum() if 'IsRainy' in display_df.columns else (
            display_df['RainyDays'].sum() if 'RainyDays' in display_df.columns else 0)
        if time_agg == "Daily":
            rain_pct = (rainy_days / len(display_df))*100 if len(display_df) else 0
            k2.metric("Rainy Days (%)", f"{int(rainy_days)} ({rain_pct:.1f}%)")
        else:
            total_days = display_df['TotalDays'].sum() if 'TotalDays' in display_df.columns else len(display_df)
            rain_pct = (rainy_days / total_days)*100 if total_days>0 else 0
            k2.metric("Rainy Days (%)", f"{int(rainy_days)} ({rain_pct:.1f}%)")

        avg_humidity = display_df['AvgHumidity'].mean() if 'AvgHumidity' in display_df else filtered['AvgHumidity'].mean()
        k3.metric("Average Humidity", f"{avg_humidity:.1f}%")

        total_rain = display_df['Rainfall'].sum()
        k4.metric("Total Rainfall", f"{total_rain:.1f} mm")

        # SECOND KPI ROW (TSI, Humidity Variation, Wind, Rainfall Intensity)
        k5, k6, k7, k8 = st.columns(4)
        if 'TSI' in display_df:
            k5.metric("Temp Stability Index (TSI)", f"{display_df['TSI'].mean():.3f}")
        if 'HumidityVar' in display_df:
            k6.metric("Humidity Variation (avg Δ%)",
                      f"{display_df['HumidityVar'].mean():.1f}")
        else:
            k6.metric("Humidity Variation (avg Δ%)",
                      f"{filtered['HumidityVariation'].mean():.1f}")
        if 'AvgWindSpeed' in display_df:
            k7.metric("Avg Wind Speed", f"{display_df['AvgWindSpeed'].mean():.1f} km/h")
        rainfall_intensity = (display_df['Rainfall'].sum() /
                              display_df['TotalDays'].sum()) if 'TotalDays' in display_df else display_df['Rainfall'].mean()
        k8.metric("Rainfall Intensity (mm/day)", f"{rainfall_intensity:.2f}")

        # ---------------- CHARTS ----------------
        st.subheader("Temperature Trends")
        st.plotly_chart(plot_temp_trends(display_df, time_col), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Rainfall")
            st.plotly_chart(plot_rainfall(display_df, time_col), use_container_width=True)
        with c2:
            st.subheader("Humidity Variation Distribution")
            if time_agg == "Monthly":
                group_col = 'MonthYear'
            elif time_agg == "Weekly":
                group_col = 'TimeLabel'
            else:
                group_col = 'MonthName'
            if time_agg == "Daily" and filtered[group_col].nunique() > 12:
                st.info("Too many groups; refine filters for boxplot.")
            else:
                st.pyplot(plot_humidity_boxplot(filtered, group_col))

        st.subheader("Wind Patterns")
        w1, w2 = st.columns(2)
        with w1:
            if time_agg == "Daily":
                st.write("Wind: 9am")
                st.pyplot(plot_wind_polar(filtered, 'WindDir9am', 'WindSpeed9am',
                                          "Wind Direction & Speed (9am)"))
            else:
                st.info("Daily aggregation required for wind direction detail.")
        with w2:
            if time_agg == "Daily":
                st.write("Wind: 3pm")
                st.pyplot(plot_wind_polar(filtered, 'WindDir3pm', 'WindSpeed3pm',
                                          "Wind Direction & Speed (3pm)"))
            else:
                st.info("Daily aggregation required for wind direction detail.")

        # ---------------- SEASONAL & DOW ANALYSIS ----------------
        with st.expander("Seasonal & Day-of-Week KPIs"):
            seasonal_df = aggregate_by_season(filtered)
            st.markdown("**Seasonal Summary**")
            st.dataframe(seasonal_df)

            dow = (filtered.groupby('DayOfWeek')
                           .agg(AvgTemp=('AvgTemp','mean'),
                                HumidityVar=('HumidityVariation','mean'),
                                AvgWind=('AvgWindSpeed','mean'),
                                Rainfall=('Rainfall','mean'))
                           .reset_index())
            # sort by weekday order
            weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow['DayOfWeek'] = pd.Categorical(dow['DayOfWeek'], categories=weekday_order, ordered=True)
            st.markdown("**Average Metrics by Day of Week**")
            st.dataframe(dow.sort_values('DayOfWeek'))

            spells = spell_metrics(filtered)
            st.markdown("**Spell Metrics (Current Filter)**")
            st.write(spells)

        # ---------------- DOWNLOADS ----------------
        with st.expander("Downloads"):
            st.download_button("Download Filtered Daily Data (CSV)",
                               data=filtered.to_csv(index=False),
                               file_name="filtered_daily_weather.csv",
                               mime="text/csv")
            if time_agg == "Weekly":
                st.download_button("Download Weekly Aggregation (CSV)",
                                   data=display_df.to_csv(index=False),
                                   file_name="weekly_weather.csv",
                                   mime="text/csv")
            if time_agg == "Monthly":
                st.download_button("Download Monthly Aggregation (CSV)",
                                   data=display_df.to_csv(index=False),
                                   file_name="monthly_weather.csv",
                                   mime="text/csv")
            st.download_button("Download Seasonal Aggregation (CSV)",
                               data=aggregate_by_season(filtered).to_csv(index=False),
                               file_name="seasonal_weather.csv",
                               mime="text/csv")

    # ===================== EDA TAB ================================
    with tab_eda:
        st.header("Exploratory Data Analysis")

        eda_df = df.copy()

        st.subheader("Data Overview")
        o1, o2 = st.columns(2)
        with o1:
            st.write(f"**Dataset Size:** {eda_df.shape[0]} rows × {eda_df.shape[1]} columns")
            st.write(f"**Date Range:** {eda_df['Date'].min().strftime('%d %b %Y')} → "
                     f"{eda_df['Date'].max().strftime('%d %b %Y')}")
        with o2:
            missing = eda_df.isnull().sum()
            miss_df = (pd.DataFrame({'Missing Values': missing,
                                     'Percent': (missing/len(eda_df))*100})
                       .query('`Missing Values`>0')
                       .sort_values('Missing Values', ascending=False)
                       .head(5))
            st.write("**Top 5 Columns with Missing Values:**")
            st.dataframe(miss_df)

        st.subheader("Correlation Heatmap")
        numeric_df = eda_df.select_dtypes(include=['number'])
        default_corr_cols = [c for c in ['MinTemp','MaxTemp','Rainfall','Humidity9am',
                                         'Humidity3pm','Pressure9am','Pressure3pm','AvgWindSpeed']
                             if c in numeric_df.columns]
        corr_features = st.multiselect("Select features:", numeric_df.columns.tolist(),
                                       default=default_corr_cols)
        if corr_features:
            corr = numeric_df[corr_features].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            corr_flat = (corr.unstack()
                             .reset_index()
                             .rename(columns={'level_0':'Feature 1','level_1':'Feature 2',0:'Correlation'}))
            corr_flat = corr_flat[corr_flat['Feature 1'] != corr_flat['Feature 2']]
            corr_sorted = corr_flat.sort_values('Correlation', ascending=False)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.write("**Top 5 Positive**")
                st.dataframe(corr_sorted.head(5))
            with cc2:
                st.write("**Top 5 Negative**")
                st.dataframe(corr_sorted.tail(5))

        st.subheader("Distribution Analysis")
        dist_feature = st.selectbox("Feature:", numeric_df.columns.tolist(),
                                    index=(numeric_df.columns.tolist().index('AvgTemp')
                                           if 'AvgTemp' in numeric_df.columns else 0))
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(numeric_df[dist_feature].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {dist_feature}")
        st.pyplot(fig)
        st.write("**Descriptive Statistics**")
        st.dataframe(numeric_df[dist_feature].describe())

        st.subheader("Outlier Detection (IQR)")
        outlier_cols = st.multiselect("Columns:", numeric_df.columns.tolist(),
                                      default=[c for c in ['MinTemp','MaxTemp','Rainfall','WindGustSpeed']
                                               if c in numeric_df.columns])
        if outlier_cols:
            detected = detect_outliers(eda_df, outlier_cols)
            out_df = (pd.DataFrame({'Column': list(detected.keys()),
                                    'Outlier Count': list(detected.values())})
                      .sort_values('Outlier Count', ascending=False))
            fig2 = px.bar(out_df, x='Column', y='Outlier Count',
                          title='Outliers per Column',
                          color='Outlier Count', text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

            chosen = st.selectbox("Boxplot column:", outlier_cols)
            fig3, ax3 = plt.subplots(figsize=(8,5))
            sns.boxplot(y=eda_df[chosen], ax=ax3)
            ax3.set_title(f"Boxplot: {chosen}")
            st.pyplot(fig3)

        st.subheader("Monthly Time Series (Aggregated)")
        ts_df = (df
                 .groupby(pd.Grouper(key='Date', freq='M'))
                 .agg(MinTemp=('MinTemp','mean'),
                      MaxTemp=('MaxTemp','mean'),
                      Rainfall=('Rainfall','sum'),
                      Humidity9am=('Humidity9am','mean'),
                      Humidity3pm=('Humidity3pm','mean'),
                      AvgWindSpeed=('AvgWindSpeed','mean'))
                 .reset_index())
        ts_metric = st.selectbox("Timeseries Metric:",
                                 [c for c in ['MinTemp','MaxTemp','Rainfall','Humidity9am',
                                              'Humidity3pm','AvgWindSpeed'] if c in ts_df.columns])
        ts_fig = px.line(ts_df, x='Date', y=ts_metric,
                         title=f"Monthly {ts_metric} Over Time")
        st.plotly_chart(ts_fig, use_container_width=True)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
