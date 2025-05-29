import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
# Add Prophet for forecasting
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# reading the data from excel file
df = pd.read_excel("Adidas.xlsx")
st.set_page_config(layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# PAGE SELECTOR AT THE TOP
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Predictive Analysis", "Advance Predictive Models"])

if page == "Dashboard":
    # --- DASHBOARD CODE START ---
    # Initialize session state for filters
    if 'filters_reset' not in st.session_state:
        st.session_state.filters_reset = False

    # Function to reset filters
    def reset_filters():
        st.session_state.filters_reset = True
        st.rerun()

    st.sidebar.header("Filters")
    # Year Filter
    years = sorted(df['InvoiceDate'].dt.year.unique())
    selected_years = st.sidebar.multiselect(
        "Select Year",
        years,
        default=years
    )

    # Month Filter
    months = sorted(df['InvoiceDate'].dt.month.unique())
    month_names = [datetime.date(2020, month, 1).strftime('%B') for month in months]
    selected_months = st.sidebar.multiselect(
        "Select Month",
        month_names,
        default=month_names
    )

    # Product Filter
    products = sorted(df['Product'].unique())
    selected_products = st.sidebar.multiselect(
        "Select Product",
        products,
        default=products
    )

    # Retailer Filter
    retailers = sorted(df['Retailer'].unique())
    selected_retailers = st.sidebar.multiselect(
        "Select Retailer",
        retailers,
        default=retailers
    )

    # Region Filter
    regions = sorted(df['Region'].unique())
    selected_regions = st.sidebar.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    # Reset button
    if st.sidebar.button("Reset Filters", on_click=reset_filters):
        st.session_state.filters_reset = True
        st.rerun()

    # Apply filters to the dataframe
    filtered_df = df[
        (df['InvoiceDate'].dt.year.isin(selected_years)) &
        (df['InvoiceDate'].dt.strftime('%B').isin(selected_months)) &
        (df['Product'].isin(selected_products)) &
        (df['Retailer'].isin(selected_retailers)) &
        (df['Region'].isin(selected_regions))
    ]

    # Add summary statistics
    st.sidebar.header("Summary Statistics")
    st.sidebar.write(f"Total Sales: ${filtered_df['TotalSales'].sum():,.2f}")
    st.sidebar.write(f"Total Units Sold: {filtered_df['UnitsSold'].sum():,}")
    st.sidebar.write(f"Average Operating Margin: {filtered_df['OperatingMargin'].mean():.2f}%")

    # --- DASHBOARD HEADER ---
    image = Image.open('adidas-logo.jpg')

    col1, col2 = st.columns([0.1,0.9])
    with col1:
        st.image(image,width=100)

    html_title = """
        <style>
        .title-test {
        font-weight:bold;
        padding:5px;
        border-radius:6px;
        }
        </style>
        <center><h1 class="title-test">Adidas Interactive Sales Dashboard</h1></center>"""
    with col2:
        st.markdown(html_title, unsafe_allow_html=True)

    # --- KPI METRICS ---
    st.markdown("---")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        st.markdown("### Total Sales")
        total_sales = filtered_df['TotalSales'].sum()
        st.markdown(f"<h2 style='text-align: center; color: #00B2A9;'>${total_sales:,.2f}</h2>", unsafe_allow_html=True)

    with col_kpi2:
        st.markdown("### Total Quantity Sold")
        total_quantity = filtered_df['UnitsSold'].sum()
        st.markdown(f"<h2 style='text-align: center; color: #00B2A9;'>{total_quantity:,}</h2>", unsafe_allow_html=True)

    with col_kpi3:
        st.markdown("### Total Profit")
        total_profit = filtered_df['OperatingProfit'].sum()
        st.markdown(f"<h2 style='text-align: center; color: #00B2A9;'>${total_profit:,.2f}</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # --- DATE INFO ---
    col3, col4, col5 = st.columns([0.1,0.45,0.45])
    with col3:
        box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
        st.write(f"Last updated by:  \n {box_date}")

    # --- RETAILER SALES CHART ---
    with col4:
        fig = px.bar(filtered_df, x="Retailer", y="TotalSales", 
                     labels={"TotalSales" : "Total Sales {$}"},
                     title="Total Sales by Retailer", 
                     hover_data=["TotalSales"],
                     template="gridon", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # --- DOWNLOAD BUTTONS AND EXPANDERS ---
    _, view1, dwn1, view2, dwn2 = st.columns([0.15,0.20,0.20,0.20,0.20])
    with view1:
        expander = st.expander("Retailer wise Sales")
        data = filtered_df[["Retailer","TotalSales"]].groupby(by="Retailer")["TotalSales"].sum()
        expander.write(data)
    with dwn1:
        st.download_button("Get Data", data=data.to_csv().encode("utf-8"),
                           file_name="RetailerSales.csv", mime="text/csv")

    # --- TIME SERIES CHART ---
    filtered_df["Month_Year"] = filtered_df["InvoiceDate"].dt.strftime("%b'%y")
    result = filtered_df.groupby(by=filtered_df["Month_Year"])["TotalSales"].sum().reset_index()

    with col5:
        fig1 = px.line(result, x="Month_Year", y="TotalSales", 
                       title="Total Sales Over Time",
                       template="gridon")
        st.plotly_chart(fig1, use_container_width=True)

    with view2:
        expander = st.expander("Monthly Sales")
        data = result
        expander.write(data)
    with dwn2:
        st.download_button("Get Data", data=result.to_csv().encode("utf-8"),
                           file_name="MonthlySales.csv", mime="text/csv")

    # --- ADDITIONAL VISUALIZATIONS ---
    st.subheader("Additional Sales Analytics")

    # 1. Product Performance Analysis
    col8, col9 = st.columns(2)
    with col8:
        product_performance = filtered_df.groupby('Product')[['TotalSales', 'UnitsSold']].sum().reset_index()
        fig5 = px.scatter(product_performance, 
                         x='UnitsSold', 
                         y='TotalSales',
                         color='Product',
                         size='TotalSales',
                         title='Product Performance: Sales vs Units Sold',
                         labels={'UnitsSold': 'Units Sold', 'TotalSales': 'Total Sales ($)'},
                         template='gridon')
        st.plotly_chart(fig5, use_container_width=True)

    # 2. Sales Method Distribution
    with col9:
        sales_method = filtered_df.groupby('SalesMethod')['TotalSales'].sum().reset_index()
        fig6 = px.pie(sales_method, 
                     values='TotalSales', 
                     names='SalesMethod',
                     title='Sales Distribution by Method',
                     hole=0.4)
        st.plotly_chart(fig6, use_container_width=True)

    # 3. Profit Margin Analysis
    col10, col11 = st.columns(2)
    with col10:
        profit_margin = filtered_df.groupby('Product')[['TotalSales', 'OperatingMargin']].sum().reset_index()
        fig7 = px.bar(profit_margin, 
                     x='Product', 
                     y='OperatingMargin',
                     title='Operating Margin by Product (%)',
                     template='gridon')
        st.plotly_chart(fig7, use_container_width=True)

    # 4. Top 10 Products by Revenue
    with col11:
        top_products = filtered_df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(10)
        fig8 = px.bar(top_products.reset_index(), 
                     x='TotalSales', 
                     y='Product',
                     title='Top 10 Products by Revenue',
                     orientation='h',
                     template='gridon')
        st.plotly_chart(fig8, use_container_width=True)

    st.divider()

    # --- STATE-WISE SALES AND UNITS SOLD ---
    result1 = filtered_df.groupby(by="State")[["TotalSales","UnitsSold"]].sum().reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=result1["State"], y=result1["TotalSales"], name="Total Sales"))
    fig3.add_trace(go.Scatter(x=result1["State"], y=result1["UnitsSold"], mode="lines",
                              name="Units Sold", yaxis="y2"))
    fig3.update_layout(
        title="Total Sales and Units Sold by State",
        xaxis=dict(title="State"),
        yaxis=dict(title="Total Sales", showgrid=False),
        yaxis2=dict(title="Units Sold", overlaying="y", side="right"),
        template="gridon",
        legend=dict(x=1, y=1.1)
    )

    _, col6 = st.columns([0.1,1])
    with col6:
        st.plotly_chart(fig3, use_container_width=True)

    _, view3, dwn3 = st.columns([0.5,0.45,0.45])
    with view3:
        expander = st.expander("View Data for Sales by Units Sold")
        expander.write(result1)
    with dwn3:
        st.download_button("Get Data", data=result1.to_csv().encode("utf-8"), 
                           file_name="Sales_by_UnitsSold.csv", mime="text/csv")

    st.divider()

    # --- TREEMAP ---
    _, col7 = st.columns([0.1,1])
    treemap = filtered_df[["Region","City","TotalSales"]].groupby(by=["Region","City"])["TotalSales"].sum().reset_index()

    def format_sales(value):
        if value >= 0:
            return '{:.2f} Lakh'.format(value / 1_000_00)

    treemap["TotalSales (Formatted)"] = treemap["TotalSales"].apply(format_sales)

    fig4 = px.treemap(treemap, path=["Region","City"], values="TotalSales",
                      hover_name="TotalSales (Formatted)",
                      hover_data=["TotalSales (Formatted)"],
                      color="City", height=700, width=600)
    fig4.update_traces(textinfo="label+value")

    with col7:
        st.subheader("Total Sales by Region and City in Treemap")
        st.plotly_chart(fig4, use_container_width=True)

    _, view4, dwn4 = st.columns([0.5,0.45,0.45])
    with view4:
        result2 = filtered_df[["Region","City","TotalSales"]].groupby(by=["Region","City"])["TotalSales"].sum()
        expander = st.expander("View data for Total Sales by Region and City")
        expander.write(result2)
    with dwn4:
        st.download_button("Get Data", data=result2.to_csv().encode("utf-8"),
                           file_name="Sales_by_Region.csv", mime="text/csv")

    # --- OPERATING MARGIN RANGE ANALYSIS ---
    st.subheader("Operating Margin Range Analysis")
    try:
        filtered_df['MarginRange'] = pd.cut(filtered_df['OperatingMargin'], 
                                          bins=5, 
                                          labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
        
        margin_analysis = filtered_df.groupby('MarginRange').agg({
            'TotalSales': 'sum',
            'OperatingMargin': 'mean',
            'UnitsSold': 'sum'
        }).reset_index()
        
        # Calculate percentage of total sales
        total_sales = margin_analysis['TotalSales'].sum()
        margin_analysis['SalesPercentage'] = (margin_analysis['TotalSales'] / total_sales * 100).round(2)
        
        fig9 = px.bar(margin_analysis, 
                     x='MarginRange', 
                     y='TotalSales',
                     title='Sales Distribution by Operating Margin Range',
                     template='gridon',
                     labels={
                         'MarginRange': 'Operating Margin Range',
                         'TotalSales': 'Total Sales ($)',
                         'SalesPercentage': 'Percentage of Total Sales'
                     },
                     hover_data=['SalesPercentage', 'OperatingMargin', 'UnitsSold'])
        
        # Update layout
        fig9.update_layout(
            xaxis_title="Operating Margin Range",
            yaxis_title="Total Sales ($)",
            showlegend=False
        )
        
        # Add percentage labels on top of bars
        fig9.update_traces(
            texttemplate='%{customdata[0]:.1f}%',
            textposition='outside'
        )
        
        st.plotly_chart(fig9, use_container_width=True)
        
        # Display detailed analysis
        st.write("### Detailed Margin Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Sales by Margin Range")
            st.dataframe(margin_analysis[[
                'MarginRange', 
                'TotalSales', 
                'SalesPercentage', 
                'OperatingMargin'
            ]].style.format({
                'TotalSales': '${:,.2f}',
                'SalesPercentage': '{:.2f}%',
                'OperatingMargin': '{:.2f}%'
            }))
        
        with col2:
            st.write("#### Key Insights")
            st.write(f"• Highest sales are in the {margin_analysis.loc[margin_analysis['TotalSales'].idxmax(), 'MarginRange']} range")
            st.write(f"• {margin_analysis['SalesPercentage'].max():.1f}% of total sales come from the {margin_analysis.loc[margin_analysis['SalesPercentage'].idxmax(), 'MarginRange']} range")
            st.write(f"• Average operating margin across all ranges: {margin_analysis['OperatingMargin'].mean():.1f}%")

    except Exception as e:
        st.error(f"Error in margin analysis: {str(e)}")

    # --- RAW DATA VIEW ---
    _, view5, dwn5 = st.columns([0.5,0.45,0.45])
    with view5:
        expander = st.expander("View Sales Raw Data")
        expander.write(filtered_df)
    with dwn5:
        st.download_button("Get Raw Data", data=filtered_df.to_csv().encode("utf-8"),
                           file_name="SalesRawData.csv", mime="text/csv")

    st.divider()
    # --- DASHBOARD CODE END ---

elif page == "Predictive Analysis":
    st.title("Predictive Analysis: Future Sales Forecast")
    st.markdown("---")
    st.info("This section uses Facebook Prophet to forecast future sales based on historical data.")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Forecast Settings")
    
    # Metric selection
    forecast_metric = st.sidebar.selectbox(
        "Select Metric to Forecast",
        ["Total Sales", "Units Sold", "Operating Profit", "Operating Margin"],
        index=0,
        key="metric_select"  # Unique key added
    )
    
    # Aggregation level
    agg_level = st.sidebar.selectbox(
        "Aggregation Level",
        ["Overall", "By Product", "By Region"],
        index=0,
        key="agg_level_select"  # Unique key added
    )
    
    # Forecast horizon
    periods = st.sidebar.selectbox(
        "Forecast Horizon (months)", 
        [3, 6, 12], 
        index=1,
        key="horizon_select"  # Unique key added
    )
    
    # Confidence interval
    confidence_level = st.sidebar.slider(
        "Confidence Interval", 
        70, 95, 80,
        key="confidence_slider"  # Unique key added
    )
    
    # --- DATA PREPARATION ---
    def prepare_forecast_data(df, metric, groupby=None):
        """Prepare data for Prophet forecasting"""
        df = df.copy()
        
        # Convert InvoiceDate to datetime if not already
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Select the appropriate metric column
        metric_col = {
            "Total Sales": "TotalSales",
            "Units Sold": "UnitsSold",
            "Operating Profit": "OperatingProfit",
            "Operating Margin": "OperatingMargin"
        }[metric]
        
        if groupby:
            # Group by date and the specified column
            forecast_df = df.groupby(['InvoiceDate', groupby])[metric_col].sum().reset_index()
            forecast_df = forecast_df.rename(columns={"InvoiceDate": "ds", metric_col: "y", groupby: "group"})
        else:
            # Aggregate by date only
            forecast_df = df.groupby('InvoiceDate')[metric_col].sum().reset_index()
            forecast_df = forecast_df.rename(columns={"InvoiceDate": "ds", metric_col: "y"})
        
        return forecast_df
    
    # Get the appropriate data based on selections
    if agg_level == "Overall":
        forecast_data = prepare_forecast_data(df, forecast_metric)
    elif agg_level == "By Product":
        forecast_data = prepare_forecast_data(df, forecast_metric, "Product")
    elif agg_level == "By Region":
        forecast_data = prepare_forecast_data(df, forecast_metric, "Region")
    
    # --- FORECASTING ---
    def run_prophet_forecast(data, periods, confidence_level, groupby=None):
        """Run Prophet forecast on the given data"""
        if groupby:
            # For grouped forecasts
            groups = data['group'].unique()
            forecasts = {}
            
            for group in groups:
                group_data = data[data['group'] == group][['ds', 'y']]
                
                # Handle zero values for multiplicative seasonality
                if group_data['y'].min() <= 0:
                    # Additive seasonality for metrics that can be zero or negative
                    m = Prophet(seasonality_mode='additive', interval_width=confidence_level/100)
                else:
                    # Multiplicative seasonality for strictly positive metrics
                    m = Prophet(seasonality_mode='multiplicative', interval_width=confidence_level/100)
                
                m.fit(group_data)
                future = m.make_future_dataframe(periods=periods, freq='M')
                forecast = m.predict(future)
                forecasts[group] = forecast
            
            return forecasts
        else:
            # For overall forecast
            if data['y'].min() <= 0:
                m = Prophet(seasonality_mode='additive', interval_width=confidence_level/100)
            else:
                m = Prophet(seasonality_mode='multiplicative', interval_width=confidence_level/100)
            
            m.fit(data)
            future = m.make_future_dataframe(periods=periods, freq='M')
            forecast = m.predict(future)
            
            return forecast
    
    # Run the forecast
    if agg_level == "Overall":
        forecast_result = run_prophet_forecast(forecast_data, periods, confidence_level)
    else:
        forecast_result = run_prophet_forecast(
            forecast_data, 
            periods, 
            confidence_level, 
            groupby="Product" if agg_level == "By Product" else "Region"
        )
    
    # --- VISUALIZATION ---
    st.subheader(f"Forecast: {forecast_metric} {agg_level.lower()}")
    
    if agg_level == "Overall":
        # Plot overall forecast
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['y'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_result['ds'],
            y=forecast_result['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='green')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_result['ds'].tolist() + forecast_result['ds'].tolist()[::-1],
            y=forecast_result['yhat_upper'].tolist() + forecast_result['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Formatting
        yaxis_title = {
            "Total Sales": "Total Sales ($)",
            "Units Sold": "Units Sold",
            "Operating Profit": "Operating Profit ($)",
            "Operating Margin": "Operating Margin (%)"
        }[forecast_metric]
        
        fig.update_layout(
            title=f"{forecast_metric} Forecast",
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            template="gridon",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast table
        st.subheader("Forecast Data")
        forecast_display = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_display = forecast_display.rename(columns={
            'ds': 'Date',
            'yhat': f'Predicted {forecast_metric}',
            'yhat_lower': 'Lower Estimate',
            'yhat_upper': 'Upper Estimate'
        })
        
        # Format values based on metric
        if forecast_metric in ["Total Sales", "Operating Profit"]:
            forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"${x:,.2f}")
            forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"${x:,.2f}")
            forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"${x:,.2f}")
        elif forecast_metric == "Units Sold":
            forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"{x:,.0f}")
            forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"{x:,.0f}")
            forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"{x:,.0f}")
        else:  # Operating Margin
            forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"{x:.2f}%")
            forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"{x:.2f}%")
            forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"{x:.2f}%")
        
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%b %Y')
        st.dataframe(forecast_display.reset_index(drop=True))
        
    else:
        # Plot grouped forecasts
        tabs = st.tabs(list(forecast_result.keys()))
        
        for i, (group, forecast) in enumerate(forecast_result.items()):
            with tabs[i]:
                # Get historical data for this group
                hist_data = forecast_data[forecast_data['group'] == group]
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist_data['ds'],
                    y=hist_data['y'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                # Formatting
                yaxis_title = {
                    "Total Sales": "Total Sales ($)",
                    "Units Sold": "Units Sold",
                    "Operating Profit": "Operating Profit ($)",
                    "Operating Margin": "Operating Margin (%)"
                }[forecast_metric]
                
                fig.update_layout(
                    title=f"{forecast_metric} Forecast for {group}",
                    xaxis_title="Date",
                    yaxis_title=yaxis_title,
                    template="gridon",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.subheader(f"Forecast Data for {group}")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                forecast_display = forecast_display.rename(columns={
                    'ds': 'Date',
                    'yhat': f'Predicted {forecast_metric}',
                    'yhat_lower': 'Lower Estimate',
                    'yhat_upper': 'Upper Estimate'
                })
                
                # Format values based on metric
                if forecast_metric in ["Total Sales", "Operating Profit"]:
                    forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"${x:,.2f}")
                    forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"${x:,.2f}")
                    forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"${x:,.2f}")
                elif forecast_metric == "Units Sold":
                    forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"{x:,.0f}")
                    forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"{x:,.0f}")
                    forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"{x:,.0f}")
                else:  # Operating Margin
                    forecast_display[f'Predicted {forecast_metric}'] = forecast_display[f'Predicted {forecast_metric}'].apply(lambda x: f"{x:.2f}%")
                    forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"{x:.2f}%")
                    forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"{x:.2f}%")
                
                forecast_display['Date'] = forecast_display['Date'].dt.strftime('%b %Y')
                st.dataframe(forecast_display.reset_index(drop=True))
    
    # --- MODEL EVALUATION ---
    st.markdown("---")
    st.subheader("Model Evaluation")
    
    # Create train/test split
    test_size = 3  # months
    eval_results = []
    
    if agg_level == "Overall":
        # For overall forecast evaluation
        monthly_data = forecast_data.copy()
        monthly_data['YearMonth'] = monthly_data['ds'].dt.to_period('M')
        monthly_data = monthly_data.groupby('YearMonth')['y'].sum().reset_index()
        monthly_data['ds'] = monthly_data['YearMonth'].dt.to_timestamp()
        monthly_data = monthly_data[['ds', 'y']]
        
        train_df = monthly_data.iloc[:-test_size]
        test_df = monthly_data.iloc[-test_size:]
        
        # Train model
        if monthly_data['y'].min() <= 0:
            m = Prophet(seasonality_mode='additive')
        else:
            m = Prophet(seasonality_mode='multiplicative')
        
        m.fit(train_df)
        
        # Make future dataframe for test period
        future = m.make_future_dataframe(periods=test_size, freq='M')
        forecast = m.predict(future)
        
        # Get test predictions
        pred_test = forecast[['ds', 'yhat']].tail(test_size).reset_index(drop=True)
        actual_test = test_df[['ds', 'y']].reset_index(drop=True)
        
        # Calculate metrics
        mae = mean_absolute_error(actual_test['y'], pred_test['yhat'])
        rmse = np.sqrt(mean_squared_error(actual_test['y'], pred_test['yhat']))
        mape = np.mean(np.abs((actual_test['y'] - pred_test['yhat']) / actual_test['y'])) * 100
        
        # Store results
        eval_results.append({
            "Group": "Overall",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })
        
        # Show comparison
        st.write(f"**Evaluation for Overall {forecast_metric}:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${mae:,.2f}" if forecast_metric in ["Total Sales", "Operating Profit"] else f"{mae:,.2f}{'%' if forecast_metric == 'Operating Margin' else ''}")
        col2.metric("RMSE", f"${rmse:,.2f}" if forecast_metric in ["Total Sales", "Operating Profit"] else f"{rmse:,.2f}{'%' if forecast_metric == 'Operating Margin' else ''}")
        col3.metric("MAPE", f"{mape:.2f}%")
        
        # Plot actual vs predicted
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=actual_test['ds'],
            y=actual_test['y'],
            mode='lines+markers',
            name='Actual'
        ))
        fig_eval.add_trace(go.Scatter(
            x=pred_test['ds'],
            y=pred_test['yhat'],
            mode='lines+markers',
            name='Predicted'
        ))
        fig_eval.update_layout(
            title=f"Actual vs Predicted {forecast_metric} (Test Set)",
            xaxis_title="Date",
            yaxis_title=forecast_metric,
            template="gridon"
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
    else:
        # For grouped forecast evaluation
        groups = forecast_data['group'].unique()
        
        for group in groups:
            group_data = forecast_data[forecast_data['group'] == group][['ds', 'y']]
            
            # Aggregate by month
            monthly_data = group_data.copy()
            monthly_data['YearMonth'] = monthly_data['ds'].dt.to_period('M')
            monthly_data = monthly_data.groupby('YearMonth')['y'].sum().reset_index()
            monthly_data['ds'] = monthly_data['YearMonth'].dt.to_timestamp()
            monthly_data = monthly_data[['ds', 'y']]
            
            # Skip if not enough data
            if len(monthly_data) <= test_size:
                continue
                
            train_df = monthly_data.iloc[:-test_size]
            test_df = monthly_data.iloc[-test_size:]
            
            # Train model
            if monthly_data['y'].min() <= 0:
                m = Prophet(seasonality_mode='additive')
            else:
                m = Prophet(seasonality_mode='multiplicative')
            
            m.fit(train_df)
            
            # Make future dataframe for test period
            future = m.make_future_dataframe(periods=test_size, freq='M')
            forecast = m.predict(future)
            
            # Get test predictions
            pred_test = forecast[['ds', 'yhat']].tail(test_size).reset_index(drop=True)
            actual_test = test_df[['ds', 'y']].reset_index(drop=True)
            
            # Calculate metrics
            mae = mean_absolute_error(actual_test['y'], pred_test['yhat'])
            rmse = np.sqrt(mean_squared_error(actual_test['y'], pred_test['yhat']))
            mape = np.mean(np.abs((actual_test['y'] - pred_test['yhat']) / actual_test['y'])) * 100
            
            # Store results
            eval_results.append({
                "Group": group,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            })
        
        # Show evaluation results in a table
        st.write(f"**Evaluation Metrics by {agg_level.split(' ')[1]}:**")
        eval_df = pd.DataFrame(eval_results)
        
        # Format metrics based on forecast metric
        if forecast_metric in ["Total Sales", "Operating Profit"]:
            eval_df['MAE'] = eval_df['MAE'].apply(lambda x: f"${x:,.2f}")
            eval_df['RMSE'] = eval_df['RMSE'].apply(lambda x: f"${x:,.2f}")
        elif forecast_metric == "Units Sold":
            eval_df['MAE'] = eval_df['MAE'].apply(lambda x: f"{x:,.0f}")
            eval_df['RMSE'] = eval_df['RMSE'].apply(lambda x: f"{x:,.0f}")
        else:  # Operating Margin
            eval_df['MAE'] = eval_df['MAE'].apply(lambda x: f"{x:.2f}%")
            eval_df['RMSE'] = eval_df['RMSE'].apply(lambda x: f"{x:.2f}%")
        
        eval_df['MAPE'] = eval_df['MAPE'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(eval_df)
    
    # --- FORECAST COMPARISON TOOL ---
    st.markdown("---")
    st.subheader("Forecast Comparison Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_metric = st.selectbox(
            "Select Metric to Compare",
            ["Total Sales", "Units Sold", "Operating Profit", "Operating Margin"],
            index=0,
            key="compare_metric_select"  # Unique key added
        )
    
    with col2:
        compare_groups = st.multiselect(
            f"Select {agg_level.split(' ')[1]} to Compare" if agg_level != "Overall" else "Select to Compare",
            list(forecast_result.keys()) if agg_level != "Overall" else ["Overall"],
            default=list(forecast_result.keys())[:2] if agg_level != "Overall" else ["Overall"],
            key="compare_groups_select"  # Unique key added
        )
    
    # Create comparison plot
    if compare_groups:
        fig_compare = go.Figure()
        
        for group in compare_groups:
            if agg_level == "Overall":
                forecast = forecast_result
            else:
                forecast = forecast_result[group]
            
            # Filter to only future period
            future_dates = forecast['ds'] > forecast_data['ds'].max()
            future_forecast = forecast[future_dates]
            
            fig_compare.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines+markers',
                name=group,
                hovertemplate=f"{group}<br>Date: %{{x|%b %Y}}<br>Predicted {compare_metric}: %{{y:,}}"
            ))
        
        # Format y-axis based on metric
        yaxis_title = {
            "Total Sales": "Total Sales ($)",
            "Units Sold": "Units Sold",
            "Operating Profit": "Operating Profit ($)",
            "Operating Margin": "Operating Margin (%)"
        }[compare_metric]
        
        fig_compare.update_layout(
            title=f"{compare_metric} Forecast Comparison",
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            template="gridon",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # --- DOWNLOAD FORECASTS ---
    st.markdown("---")
    st.subheader("Download Forecast Data")
    
    if st.button("Prepare Forecast Data for Download", key="download_btn"):
        if agg_level == "Overall":
            forecast_download = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_download = forecast_download.rename(columns={
                'ds': 'Date',
                'yhat': f'Predicted_{forecast_metric}',
                'yhat_lower': 'Lower_Estimate',
                'yhat_upper': 'Upper_Estimate'
            })
        else:
            forecast_download = pd.DataFrame()
            
            for group, forecast in forecast_result.items():
                group_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                group_forecast['Group'] = group
                forecast_download = pd.concat([forecast_download, group_forecast])
            
            forecast_download = forecast_download.rename(columns={
                'ds': 'Date',
                'yhat': f'Predicted_{forecast_metric}',
                'yhat_lower': 'Lower_Estimate',
                'yhat_upper': 'Upper_Estimate'
            })
        
        # Convert to CSV
        csv = forecast_download.to_csv(index=False).encode('utf-8')
        
        # Download button
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name=f"{forecast_metric.replace(' ', '_')}_Forecast_{agg_level.replace(' ', '_')}.csv",
            mime="text/csv",
            key="final_download_btn"  # Unique key added
        )

elif page == "Advance Predictive Models":
    st.title("Advanced Predictive Analysis: Random Forest Forecasting")
    st.markdown("---")
    st.info("This section uses Random Forest Regressor to forecast future sales based on historical data, with consistent data processing as the Prophet model.")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Forecast Settings")
    
    # Metric selection
    forecast_metric = st.sidebar.selectbox(
        "Select Metric to Forecast",
        ["Total Sales", "Units Sold", "Operating Profit", "Operating Margin"],
        index=0
    )
    
    # Aggregation level
    agg_level = st.sidebar.selectbox(
        "Aggregation Level",
        ["Overall", "By Product", "By Region"],
        index=0
    )
    
    # Forecast horizon
    periods = st.sidebar.selectbox(
        "Forecast Horizon (months)", 
        [3, 6, 12], 
        index=1
    )
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
    
    # --- DATA PREPARATION (MATCHING PROPHET) ---
    def prepare_forecast_data(df, metric, groupby=None):
        """Prepare data consistently with Prophet model"""
        df = df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Select the appropriate metric column
        metric_col = {
            "Total Sales": "TotalSales",
            "Units Sold": "UnitsSold",
            "Operating Profit": "OperatingProfit",
            "Operating Margin": "OperatingMargin"
        }[metric]
        
        if groupby:
            # Group by date and the specified column (monthly aggregation)
            forecast_df = df.groupby([pd.Grouper(key='InvoiceDate', freq='M'), groupby])[metric_col].sum().reset_index()
            forecast_df = forecast_df.rename(columns={"InvoiceDate": "ds", metric_col: "y", groupby: "group"})
        else:
            # Aggregate by date only (monthly)
            forecast_df = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))[metric_col].sum().reset_index()
            forecast_df = forecast_df.rename(columns={"InvoiceDate": "ds", metric_col: "y"})
        
        return forecast_df
    
    # Get the data (same as Prophet)
    if agg_level == "Overall":
        forecast_data = prepare_forecast_data(df, forecast_metric)
    elif agg_level == "By Product":
        forecast_data = prepare_forecast_data(df, forecast_metric, "Product")
    elif agg_level == "By Region":
        forecast_data = prepare_forecast_data(df, forecast_metric, "Region")
    
    # --- RANDOM FOREST IMPLEMENTATION ---
    def run_rf_forecast(data, periods, n_estimators=100, max_depth=10, groupby=None):
        """Random Forest forecasting with consistent evaluation"""
        if groupby:
            # For grouped forecasts
            groups = data['group'].unique()
            forecasts = {}
            models = {}
            test_data_grouped = {}
            
            for group in groups:
                group_data = data[data['group'] == group][['ds', 'y']].copy()
                
                # Convert dates to numerical features (days since start)
                start_date = group_data['ds'].min()
                group_data['days_since_start'] = (group_data['ds'] - start_date).dt.days
                
                # Add more time-based features
                group_data['year'] = group_data['ds'].dt.year
                group_data['month'] = group_data['ds'].dt.month
                group_data['day'] = group_data['ds'].dt.day
                group_data['dayofweek'] = group_data['ds'].dt.dayofweek
                # Use isocalendar().week and convert to int
                group_data['weekofyear'] = group_data['ds'].dt.isocalendar().week.astype(int)

                # Define features (X) and target (y)
                X = group_data[['days_since_start', 'year', 'month', 'day', 'dayofweek', 'weekofyear']]
                y = group_data['y']
                
                # Train-test split (last 20% for testing)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Store model
                models[group] = model
                
                # Store test data for this group if it exists
                if len(X_test) > 0:
                    test_data_grouped[group] = (X_test, y_test)

                # Create future dates
                last_date = group_data['ds'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=periods,
                    freq='M'
                )
                
                # Prepare future features
                # future_days = (future_dates - start_date).days.values.reshape(-1, 1)

                # Create future features DataFrame
                future_features = pd.DataFrame({
                    'ds': future_dates
                })
                future_features['days_since_start'] = (future_features['ds'] - start_date).dt.days
                future_features['year'] = future_features['ds'].dt.year
                future_features['month'] = future_features['ds'].dt.month
                future_features['day'] = future_features['ds'].dt.day
                future_features['dayofweek'] = future_features['ds'].dt.dayofweek
                # Use isocalendar().week and convert to int
                future_features['weekofyear'] = future_features['ds'].dt.isocalendar().week.astype(int)

                # Predict
                y_future = model.predict(future_features[['days_since_start', 'year', 'month', 'day', 'dayofweek', 'weekofyear']]).flatten()
                
                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': y_future,
                    'yhat_lower': y_future - 1.96 * np.std(y_future),
                    'yhat_upper': y_future + 1.96 * np.std(y_future)
                })
                
                forecasts[group] = forecast
            
            return forecasts, models, test_data_grouped
        else:
            # For overall forecast
            data = data[['ds', 'y']].copy()
            
            # Convert dates to numerical features
            start_date = data['ds'].min()
            data['days_since_start'] = (data['ds'] - start_date).dt.days

            # Add more time-based features
            data['year'] = data['ds'].dt.year
            data['month'] = data['ds'].dt.month
            data['day'] = data['ds'].dt.day
            data['dayofweek'] = data['ds'].dt.dayofweek
            # Use isocalendar().week and convert to int
            data['weekofyear'] = data['ds'].dt.isocalendar().week.astype(int)

            # Define features (X) and target (y)
            X = data[['days_since_start', 'year', 'month', 'day', 'dayofweek', 'weekofyear']]
            y = data['y']
            
            # Train-test split (last 20% for testing)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Create future dates
            last_date = data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq='M'
            )
            
            # Prepare future features
            # future_days = (future_dates - start_date).days.values.reshape(-1, 1)

            # Create future features DataFrame
            future_features = pd.DataFrame({
                'ds': future_dates
            })
            future_features['days_since_start'] = (future_features['ds'] - start_date).dt.days
            future_features['year'] = future_features['ds'].dt.year
            future_features['month'] = future_features['ds'].dt.month
            future_features['day'] = future_features['ds'].dt.day
            future_features['dayofweek'] = future_features['ds'].dt.dayofweek
            # Use isocalendar().week and convert to int
            future_features['weekofyear'] = future_features['ds'].dt.isocalendar().week.astype(int)

            # Predict
            y_future = model.predict(future_features[['days_since_start', 'year', 'month', 'day', 'dayofweek', 'weekofyear']])

            # Calculate confidence intervals
            y_train_pred = model.predict(X_train)
            residuals = y_train - y_train_pred
            std = np.std(residuals)

            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': y_future,
                'yhat_lower': y_future - 1.96 * std,
                'yhat_upper': y_future + 1.96 * std
            })

            # For overall forecast, return single test split
            return forecast, model, (X_test, y_test)
    
    # Run the forecast
    if agg_level == "Overall":
        forecast_result, model, (X_test, y_test) = run_rf_forecast(
            forecast_data,
            periods,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    else:
        forecast_result, models, test_data = run_rf_forecast(
            forecast_data,
            periods,
            n_estimators=n_estimators,
            max_depth=max_depth,
            groupby="Product" if agg_level == "By Product" else "Region"
        )
    
    # --- VISUALIZATION (SAME STYLE AS PROPHET) ---
    st.subheader(f"Random Forest Forecast: {forecast_metric} {agg_level.lower()}")
    
    if agg_level == "Overall":
        # Plot overall forecast
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['y'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_result['ds'],
            y=forecast_result['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_result['ds'].tolist() + forecast_result['ds'].tolist()[::-1],
            y=forecast_result['yhat_upper'].tolist() + forecast_result['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Formatting
        yaxis_title = {
            "Total Sales": "Total Sales ($)",
            "Units Sold": "Units Sold",
            "Operating Profit": "Operating Profit ($)",
            "Operating Margin": "Operating Margin (%)"
        }[forecast_metric]
        
        fig.update_layout(
            title=f"{forecast_metric} Forecast (Random Forest)",
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            template="gridon",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Plot grouped forecasts
        tabs = st.tabs(list(forecast_result.keys()))
        
        for i, (group, forecast) in enumerate(forecast_result.items()):
            with tabs[i]:
                # Get historical data for this group
                hist_data = forecast_data[forecast_data['group'] == group]
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist_data['ds'],
                    y=hist_data['y'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='green')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                # Formatting
                yaxis_title = {
                    "Total Sales": "Total Sales ($)",
                    "Units Sold": "Units Sold",
                    "Operating Profit": "Operating Profit ($)",
                    "Operating Margin": "Operating Margin (%)"
                }[forecast_metric]
                
                fig.update_layout(
                    title=f"{forecast_metric} Forecast for {group} (Random Forest)",
                    xaxis_title="Date",
                    yaxis_title=yaxis_title,
                    template="gridon",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # --- MODEL EVALUATION (SAME APPROACH AS PROPHET) ---
    st.markdown("---")
    st.subheader("Model Evaluation")
    
    def calculate_metrics(y_true, y_pred, metric_name):
        """Consistent metric calculation with Prophet"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }
    
    if agg_level == "Overall":
        # For overall forecast evaluation
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, forecast_metric)
        
        st.write(f"**Evaluation for Overall {forecast_metric}:**")
        col1, col2, col3 = st.columns(3)
        
        # Format based on metric type
        if forecast_metric in ["Total Sales", "Operating Profit"]:
            col1.metric("MAE", f"${metrics['MAE']:,.2f}")
            col2.metric("RMSE", f"${metrics['RMSE']:,.2f}")
        else:
            col1.metric("MAE", f"{metrics['MAE']:,.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
        
        col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        
        # Plot actual vs predicted
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=forecast_data['ds'].iloc[-len(y_test):],
            y=y_test,
            mode='lines+markers',
            name='Actual'
        ))
        fig_eval.add_trace(go.Scatter(
            x=forecast_data['ds'].iloc[-len(y_test):],
            y=y_pred,
            mode='lines+markers',
            name='Predicted'
        ))
        fig_eval.update_layout(
            title=f"Actual vs Predicted {forecast_metric} (Test Set)",
            xaxis_title="Date",
            yaxis_title=forecast_metric,
            template="gridon"
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
    else:
        # For grouped forecast evaluation
        eval_results = []

        # Iterate through the grouped test data dictionary
        for group, (X_test_group, y_test_group) in test_data.items():

            # Skip if not enough data for evaluation
            if len(X_test_group) == 0 or len(y_test_group) == 0:
                st.warning(f"Not enough data to evaluate model for {group}")
                continue

            # Get the trained model for this group
            model = models[group]

            # Predict on the test data for this group
            y_pred = model.predict(X_test_group)

            # Calculate metrics for this group
            metrics = calculate_metrics(y_test_group, y_pred, forecast_metric)

            # Store results
            eval_results.append({
                "Group": group,
                "MAE": metrics['MAE'],
                "RMSE": metrics['RMSE'],
                "MAPE": metrics['MAPE']
            })
        
        # Show evaluation results in a table
        st.write(f"**Evaluation Metrics by {agg_level.split(' ')[1]}:**")
        eval_df = pd.DataFrame(eval_results)
        
        # Format metrics based on forecast metric
        if forecast_metric in ["Total Sales", "Operating Profit"]:
            eval_df['MAE'] = eval_df['MAE'].apply(lambda x: f"${x:,.2f}")
            eval_df['RMSE'] = eval_df['RMSE'].apply(lambda x: f"${x:,.2f}")
        else:
            eval_df['MAE'] = eval_df['MAE'].apply(lambda x: f"{x:,.2f}")
            eval_df['RMSE'] = eval_df['RMSE'].apply(lambda x: f"{x:,.2f}")
        
        eval_df['MAPE'] = eval_df['MAPE'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(eval_df)
    
    # --- FEATURE IMPORTANCE ---
    st.markdown("---")
    st.subheader("Feature Importance")
    
    if agg_level == "Overall":
        feature_importance = pd.DataFrame({
            'Feature': ['Days Since Start', 'Year', 'Month', 'Day', 'Day of Week', 'Week of Year'],
            'Importance': model.feature_importances_
        })
        
        fig_importance = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title='Feature Importance for Random Forest Model'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        tabs = st.tabs(list(forecast_result.keys()))
        
        for i, group in enumerate(forecast_result.keys()):
            with tabs[i]:
                feature_importance = pd.DataFrame({
                    'Feature': ['Days Since Start', 'Year', 'Month', 'Day', 'Day of Week', 'Week of Year'],
                    'Importance': models[group].feature_importances_
                })
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title=f'Feature Importance for {group}'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # --- FORECAST COMPARISON TOOL ---
    st.markdown("---")
    st.subheader("Forecast Comparison Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_metric = st.selectbox(
            "Select Metric to Compare",
            ["Total Sales", "Units Sold", "Operating Profit", "Operating Margin"],
            index=0,
            key="compare_metric_rf"
        )
    
    with col2:
        compare_groups = st.multiselect(
            f"Select {agg_level.split(' ')[1]} to Compare" if agg_level != "Overall" else "Select to Compare",
            list(forecast_result.keys()) if agg_level != "Overall" else ["Overall"],
            default=list(forecast_result.keys())[:2] if agg_level != "Overall" else ["Overall"],
            key="compare_groups_rf"
        )
    
    # Create comparison plot
    if compare_groups:
        fig_compare = go.Figure()
        
        for group in compare_groups:
            if agg_level == "Overall":
                forecast = forecast_result
            else:
                forecast = forecast_result[group]
            
            fig_compare.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines+markers',
                name=group,
                hovertemplate=f"{group}<br>Date: %{{x|%b %Y}}<br>Predicted {compare_metric}: %{{y:,}}"
            ))
        
        # Format y-axis based on metric
        yaxis_title = {
            "Total Sales": "Total Sales ($)",
            "Units Sold": "Units Sold",
            "Operating Profit": "Operating Profit ($)",
            "Operating Margin": "Operating Margin (%)"
        }[compare_metric]
        
        fig_compare.update_layout(
            title=f"{compare_metric} Forecast Comparison (Random Forest)",
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            template="gridon",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # --- DOWNLOAD FORECASTS ---
    st.markdown("---")
    st.subheader("Download Forecast Data")
    
    if st.button("Prepare Forecast Data for Download", key="download_btn_rf"):
        if agg_level == "Overall":
            forecast_download = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_download = forecast_download.rename(columns={
                'ds': 'Date',
                'yhat': f'Predicted_{forecast_metric}',
                'yhat_lower': 'Lower_Estimate',
                'yhat_upper': 'Upper_Estimate'
            })
        else:
            forecast_download = pd.DataFrame()
            
            for group, forecast in forecast_result.items():
                group_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                group_forecast['Group'] = group
                forecast_download = pd.concat([forecast_download, group_forecast])
            
            forecast_download = forecast_download.rename(columns={
                'ds': 'Date',
                'yhat': f'Predicted_{forecast_metric}',
                'yhat_lower': 'Lower_Estimate',
                'yhat_upper': 'Upper_Estimate'
            })
        
        # Convert to CSV
        csv = forecast_download.to_csv(index=False).encode('utf-8')
        
        # Download button
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name=f"{forecast_metric.replace(' ', '_')}_Forecast_RF_{agg_level.replace(' ', '_')}.csv",
            mime="text/csv"
        )