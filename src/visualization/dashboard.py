import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime, timedelta
import warnings
import json

# Set page configuration at the very top of the file
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up directory paths
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from models.demand_forecasting import ModelSelector
from models.inventory_optimization import InventoryOptimizer
from data.feature_engineering import FeatureEngineering

# Ignore warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate a timestamp for CSV export filenames
def generate_timestamp():
    """Generate a timestamp string for filenames"""
    # Use time.time() which is safer
    t = time.time()
    time_struct = time.localtime(t)
    return time.strftime('%Y%m%d_%H%M%S', time_struct)

# Custom theme styling
def set_custom_theme():
    st.markdown("""
    <style>
    /* Main layout and fonts */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Dashboard header - smaller and more compact */
    .dashboard-header {
        background: linear-gradient(90deg, #1a5fb4 0%, #3584e4 100%);
        color: white;
        padding: 0.3rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        margin-right: 7rem; /* Add space for the deploy button */
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .dashboard-header img {
        margin-right: 0.4rem;
        width: 18px;
        height: 18px;
    }
    
    /* Section containers */
    .section-container {
        background-color: white;
        padding: 0.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Metrics container - truly horizontal with fixed width */
    .metrics-container {
        display: flex;
        justify-content: space-between;
        gap: 0.2rem;
        margin-bottom: 0.5rem;
        flex-wrap: nowrap;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        padding: 0.3rem 0.4rem;
        flex: 1;
        min-width: 0; /* Allow cards to shrink below min-width */
        border-left: 2px solid #1a5fb4;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        color: #6c757d;
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 0.1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-value {
        color: #212529;
        font-size: 1rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Section headings */
    h2, h3 {
        color: #1a5fb4;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    h2 {
        font-size: 1.2rem;
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 0.3rem;
    }
    
    h3 {
        font-size: 1rem;
    }
    
    /* Reduce spacing in markdown */
    .element-container {
        margin-bottom: 0.4rem !important;
    }
    
    /* Compact paragraphs */
    p {
        margin-bottom: 0.4rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 0.2rem 0.4rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1a5fb4 !important;
        border-bottom: 2px solid #1a5fb4 !important;
    }
    
    /* Chart container */
    .chart-container {
        background-color: white;
        padding: 0.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1a5fb4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.3rem 0.6rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #164a8b;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .success-indicator {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #2ec27e;
    }
    
    .warning-indicator {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #f5c211;
    }
    
    .error-indicator {
        background-color: #f8d7da;
        color: #842029;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #e01b24;
    }
    
    /* Data tables - more compact */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    
    .dataframe th {
        background-color: #f1f3f5;
        padding: 0.4rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 1px solid #dee2e6;
    }
    
    .dataframe td {
        padding: 0.3rem 0.4rem;
        border-bottom: 1px solid #dee2e6;
    }
    
    .dataframe tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Streamlit default components */
    .css-16huue1 {
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
    }
    
    .css-1fcdlhc {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Form inputs */
    .stSelectbox, .stDateInput, .stNumberInput {
        margin-bottom: 0.4rem !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f3f5;
    }
    
    /* Custom sidebar */
    .sidebar-content {
        padding: 0.5rem;
    }
    
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a5fb4;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #dee2e6;
    }
    
    /* Remove excess whitespace in containers */
    div.block-container {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        max-width: 95%;
    }
    
    /* Make select boxes more compact */
    .stSelectbox label {
        margin-bottom: 0.2rem;
    }
    
    /* Main content area padding reduction */
    .main .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 95%;
        margin: 0 auto;
    }
    
    /* Header adjustments */
    header {
        background-color: transparent !important;
        height: auto !important;
        padding-top: 0.1rem !important;
    }
    
    /* Footer adjustments */
    footer {
        margin-top: 0.3rem;
        padding-top: 0.3rem;
        border-top: 1px solid #dee2e6;
    }
    
    /* Reduce margin between components */
    div.stMarkdown {
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
    }
    
    /* Compact plotly charts */
    .js-plotly-plot {
        margin-bottom: 0.3rem;
    }
    
    /* Reduce whitespace in tabs content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.2rem !important;
        padding-left: 0.1rem !important;
        padding-right: 0.1rem !important;
    }
    
    /* Make radio buttons more compact */
    .stRadio [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }
    
    /* Spinner more compact */
    .stSpinner {
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
    }
    
    /* Make selectbox menu smaller */
    .stSelectbox div div div {
        font-size: 0.9rem !important;
    }
    
    /* Fix for filter boxes */
    .sidebar-filter input, .sidebar-filter select {
        display: block;
        width: 100%;
    }
    
    /* Handle Streamlit's deploy/menu button */
    .stDeployButton {
        right: 0.5rem !important;
        top: 0.5rem !important;
        z-index: 99 !important;
    }
    
    button[kind="menuButton"] {
        right: 0.5rem !important;
        top: 0.5rem !important;
        z-index: 99 !important;
    }
    
    /* Adjust container width to prevent overlap with the menu button */
    .stApp > header + div {
        max-width: calc(100% - 4rem) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Apply the custom theme
set_custom_theme()

# Color palettes for consistent visualization
COLOR_PALETTE = {
    'primary': '#1a5fb4',
    'secondary': '#3584e4',
    'success': '#2ec27e',
    'danger': '#e01b24',
    'warning': '#f5c211',
    'info': '#1c71d8',
    'background': '#f6f5f4',
    'text': '#333333',
    'categorical': px.colors.qualitative.Bold  # For category-based visualizations
}

# Load data and initialize components
@st.cache_data
def load_data():
    """Load and cache the supply chain data"""
    data_loader = DataLoader()
    data_loader.load_all_datasets()
    return data_loader


@st.cache_resource
def get_feature_engineering(_data_loader):
    """Initialize and cache the feature engineering component"""
    return FeatureEngineering(_data_loader)


@st.cache_resource
def get_model_selector(_data_loader, _feature_engineering):
    """Initialize and cache the model selector component"""
    return ModelSelector(_data_loader, _feature_engineering)


@st.cache_resource
def get_inventory_optimizer(_data_loader):
    """Initialize and cache the inventory optimizer component"""
    return InventoryOptimizer(_data_loader)


# Function to train and save forecasting models
def train_demand_forecasting_models(model_selector, product_id):
    """
    Train and evaluate multiple demand forecasting models
    
    Parameters:
    -----------
    model_selector : ModelSelector
        Instance of ModelSelector class
    product_id : str
        Product ID to train model for
        
    Returns:
    --------
    tuple: Best model name and the best model
    """
    # Prepare data for the product
    X_train, X_test, y_train, y_test = model_selector.prepare_data(product_id)
    
    if X_train is None:
        st.error(f"No data available for product {product_id}")
        return None, None
    
    # Evaluate models
    with st.spinner(f"Training and evaluating models for {product_id}..."):
        results = model_selector.evaluate_models(X_train, X_test, y_train, y_test)
    
    # Get best model
    best_model_name, best_model = model_selector.get_best_model(results)
    
    # Show evaluation results
    st.write(f"#### Model Evaluation Results for {product_id}")
    
    # Convert results to DataFrame for display
    metrics_df = pd.DataFrame({
        model_name: {
            metric: round(value, 4) 
            for metric, value in results[model_name]['test_metrics'].items()
        } if 'test_metrics' in results[model_name] else {'error': results[model_name].get('error', 'Unknown error')}
        for model_name in results.keys()
    }).T
    
    st.write(metrics_df)
    
    # Highlight best model
    st.write(f"**Best model:** {best_model_name}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('models', f'{product_id}_{best_model_name}.joblib')
    
    try:
        import joblib
        joblib.dump(best_model, model_path)
        st.success(f"Model saved to {model_path}")
    except Exception as e:
        st.warning(f"Could not save model: {str(e)}")
    
    return best_model_name, best_model


# Function to generate and visualize demand forecast
def generate_demand_forecast(model_selector, model, product_id, periods=30, freq='D', last_date=None):
    """
    Generate demand forecast for a specific product
    
    Parameters:
    -----------
    model_selector : ModelSelector
        ModelSelector instance
    model : BaseForecastModel or Prophet
        Trained forecast model
    product_id : str
        Product ID
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of forecast ('D' for daily, 'W' for weekly, etc.)
    last_date : datetime, optional
        Starting point for the forecast
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with forecast
    """
    try:
        # Check if this is a Prophet model
        is_prophet = False
        if hasattr(model, 'predict') and hasattr(model, 'make_future_dataframe'):
            is_prophet = True
        
        # Generate forecast using appropriate method
        forecast = model_selector.forecast_demand(model, product_id, periods=periods, freq=freq, last_date=last_date)
        
        if forecast is not None:
            # Standardize column names - ensure we have 'forecast' not 'demand_forecast'
            if 'demand_forecast' in forecast.columns and 'forecast' not in forecast.columns:
                forecast = forecast.rename(columns={'demand_forecast': 'forecast'})
                
            # If using Prophet, confidence intervals are already included
            if not is_prophet and 'lower_ci' not in forecast.columns:
                # Add confidence intervals (default = ±15%)
                confidence_level = 0.15
                
                # If we have historical data, calculate a better confidence interval
                historical_data = model_selector.data_loader.get_demand_by_product(product_id)
                
                if not historical_data.empty:
                    # Calculate historical volatility using coefficient of variation
                    historical_mean = historical_data['demand'].mean()
                    historical_std = historical_data['demand'].std()
                    
                    if historical_mean > 0:
                        coefficient_of_variation = historical_std / historical_mean
                        
                        # Use coefficient of variation to set confidence bounds
                        # Higher volatility = wider bounds
                        confidence_level = min(0.5, max(0.1, coefficient_of_variation))
                        
                        # Add metadata about forecast quality
                        # Calculate how many days out we're forecasting
                        days_ahead = (forecast['date'].max() - historical_data['date'].max()).days
                        
                        # Determine forecast quality based on historical data and forecast horizon
                        if len(historical_data) < 60:
                            forecast_quality = "Low"
                            quality_color = "red"
                            quality_explanation = "Limited historical data available"
                        elif days_ahead > 90:
                            forecast_quality = "Medium-Low" 
                            quality_color = "orange"
                            quality_explanation = "Long-term forecasts have higher uncertainty"
                        elif coefficient_of_variation > 0.5:
                            forecast_quality = "Medium"
                            quality_color = "yellow"
                            quality_explanation = "Historical demand shows high volatility"
                        else:
                            forecast_quality = "High"
                            quality_color = "green" 
                            quality_explanation = "Good historical data with stable patterns"
                        
                        forecast.attrs['quality'] = forecast_quality
                        forecast.attrs['quality_color'] = quality_color
                        forecast.attrs['quality_explanation'] = quality_explanation
                        forecast.attrs['coefficient_of_variation'] = coefficient_of_variation
                        forecast.attrs['historical_mean'] = historical_mean
                        forecast.attrs['historical_std'] = historical_std
                
                # Add confidence intervals if not already present
                if 'lower_ci' not in forecast.columns:
                    forecast['lower_ci'] = (forecast['forecast'] * (1 - confidence_level)).clip(0)
                    forecast['upper_ci'] = forecast['forecast'] * (1 + confidence_level)
                    forecast.attrs['confidence_level'] = confidence_level
        
        return forecast
    except Exception as e:
        st.error(f"Error in generate_demand_forecast: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


# Function to optimize and visualize inventory
def optimize_inventory(inventory_optimizer, product_id, demand_forecast=None):
    """
    Optimize and visualize inventory for a product
    
    Parameters:
    -----------
    inventory_optimizer : InventoryOptimizer
        Instance of InventoryOptimizer class
    product_id : str
        Product ID to optimize inventory for
    demand_forecast : pandas.DataFrame, optional
        Demand forecast dataframe
        
    Returns:
    --------
    dict: Optimization result
    """
    # Optimize inventory
    with st.spinner(f"Optimizing inventory for {product_id}..."):
        result = inventory_optimizer.optimize_inventory(product_id, demand_forecast)
    
    if result is None:
        st.error(f"Could not optimize inventory for {product_id}")
        return None
    
    # Display results
    st.write(f"#### Inventory Optimization for {product_id}")
    
    # Create two columns for key metrics and visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display key metrics
        st.write("##### Key Metrics")
        
        # Create a metrics table
        metrics = [
            {"Metric": "Current Inventory", "Value": f"{result['current_inventory']:.0f} units"},
            {"Metric": "Daily Demand", "Value": f"{result['daily_demand']:.2f} units"},
            {"Metric": "Economic Order Quantity (EOQ)", "Value": f"{result['eoq']:.0f} units"},
            {"Metric": "Safety Stock", "Value": f"{result['safety_stock']:.0f} units"},
            {"Metric": "Reorder Point", "Value": f"{result['reorder_point']:.0f} units"},
            {"Metric": "Days to Stockout", "Value": f"{result['days_to_stockout']:.1f} days"}
        ]
        
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)
        
        # Order recommendation
        st.write("##### Order Recommendation")
        
        if result['should_order']:
            st.warning(f"**REORDER NEEDED**: Order {result['quantity_to_order']:.0f} units")
            st.write(f"Order Cost: ${result['order_cost']:.2f}")
        else:
            st.success("No order needed at this time")
    
    with col2:
        # Create a visualization of current inventory, safety stock, and reorder point
        fig = go.Figure()
        
        # Add inventory level
        fig.add_trace(go.Bar(
            x=['Current Inventory'],
            y=[result['current_inventory']],
            name='Current Inventory',
            marker_color='blue'
        ))
        
        # Add safety stock
        fig.add_trace(go.Bar(
            x=['Safety Stock'],
            y=[result['safety_stock']],
            name='Safety Stock',
            marker_color='green'
        ))
        
        # Add reorder point
        fig.add_trace(go.Bar(
            x=['Reorder Point'],
            y=[result['reorder_point']],
            name='Reorder Point',
            marker_color='red'
        ))
        
        # Update layout
        fig.update_layout(
            title='Inventory Levels',
            yaxis_title='Units',
            barmode='group',
            height=300
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a gauge chart for days to stockout
        max_days = 60  # Set a reasonable maximum (e.g., 60 days)
        days_to_stockout = min(result['days_to_stockout'], max_days)
        
        # Determine color (green > 30 days, yellow 15-30 days, red < 15 days)
        if days_to_stockout > 30:
            color = "green"
        elif days_to_stockout > 15:
            color = "gold"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=days_to_stockout,
            title={'text': "Days to Stockout"},
            gauge={
                'axis': {'range': [None, max_days]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 15], 'color': "lightcoral"},
                    {'range': [15, 30], 'color': "lightyellow"},
                    {'range': [30, max_days], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    
    # Pending orders
    if result['pending_orders']:
        st.write("##### Pending Orders")
        
        pending_df = pd.DataFrame(result['pending_orders'])
        st.write(pending_df)
    
    return result


# Function to display the overview dashboard
def display_overview_dashboard(data_loader, inventory_optimizer, category_filter="All Categories"):
    """
    Display overall supply chain KPIs and overview
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of DataLoader class
    inventory_optimizer : InventoryOptimizer
        Instance of InventoryOptimizer class
    category_filter : str
        Category filter to apply (default: "All Categories")
    """
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("## Supply Chain Overview")
    
    # Get data
    products_data = data_loader.datasets['products']
    inventory_data = data_loader.datasets['inventory']
    demand_data = data_loader.datasets['demand']
    
    # Apply category filter if needed
    if category_filter != "All Categories":
        filtered_products = products_data[products_data['category'] == category_filter]
    else:
        filtered_products = products_data
    
    # Calculate key metrics
    total_products = len(filtered_products)
    product_categories = filtered_products['category'].unique()
    total_categories = len(product_categories)
    
    # Get latest inventory data
    latest_date = inventory_data['date'].max()
    latest_inventory = inventory_data[inventory_data['date'] == latest_date]
    
    # Calculate total inventory value
    total_inventory_value = 0
    all_inventory_value = []
    turnover_data = []
    stockout_risk_data = []
    last_90_days = latest_date - pd.Timedelta(days=90)
    
    # Calculate days of inventory and turnover metrics
    with st.spinner("Calculating inventory metrics..."):
        for _, product in filtered_products.iterrows():
            product_id = product['product_id']
            product_inventory = latest_inventory[latest_inventory['product_id'] == product_id]
            
            if not product_inventory.empty:
                inventory_level = product_inventory.iloc[0]['ending_inventory']
                product_cost = product['base_cost']
                value = inventory_level * product_cost
                total_inventory_value += value
                
                # Get historical demand for this product
                product_demand = demand_data[demand_data['product_id'] == product_id]
                recent_demand = product_demand[product_demand['date'] >= last_90_days]
                
                # Calculate average daily demand
                if not recent_demand.empty:
                    avg_daily_demand = recent_demand['demand'].mean()
                    
                    # Only proceed if there's actual demand
                    if avg_daily_demand > 0:
                        # Calculate days of supply
                        days_of_supply = inventory_level / avg_daily_demand if avg_daily_demand > 0 else float('inf')
                        
                        # Calculate inventory turnover
                        # For quarterly turnover, we use avg_daily_demand * 90 as quarterly demand
                        quarterly_demand = avg_daily_demand * 90
                        quarterly_turnover = quarterly_demand / max(1, inventory_level)
                        
                        # Get stockout risk based on days of supply
                        if days_of_supply < 7:
                            stockout_risk = "High"
                            risk_color = "danger"
                        elif days_of_supply < 14:
                            stockout_risk = "Medium"
                            risk_color = "warning"
                        elif days_of_supply < 30:
                            stockout_risk = "Low"
                            risk_color = "success"
                        else:
                            stockout_risk = "Very Low"
                            risk_color = "info"
                        
                        # Calculate demand volatility (coefficient of variation)
                        cv = recent_demand['demand'].std() / recent_demand['demand'].mean() if recent_demand['demand'].mean() > 0 else 0
                        demand_volatility = cv * 100  # Express as percentage
                        
                        turnover_data.append({
                            'product_id': product_id,
                            'product_name': product['name'],
                            'category': product['category'],
                            'quarterly_turnover': quarterly_turnover,
                            'days_of_supply': min(days_of_supply, 365),  # Cap at 365 days for visualization
                            'demand_volatility': demand_volatility
                        })
                        
                        stockout_risk_data.append({
                            'product_id': product_id,
                            'product_name': product['name'],
                            'category': product['category'],
                            'current_inventory': inventory_level,
                            'avg_daily_demand': avg_daily_demand,
                            'days_of_supply': min(days_of_supply, 365),
                            'stockout_risk': stockout_risk,
                            'risk_color': risk_color
                        })
                
                all_inventory_value.append({
                    'product_id': product_id,
                    'product_name': product['name'],
                    'category': product['category'],
                    'inventory_units': inventory_level,
                    'inventory_value': value
                })
    
    # Calculate category-level metrics
    category_metrics = {}
    if turnover_data:
        turnover_df = pd.DataFrame(turnover_data)
        for category in product_categories:
            category_data = turnover_df[turnover_df['category'] == category]
            if not category_data.empty:
                avg_turnover = category_data['quarterly_turnover'].mean()
                avg_days_supply = category_data['days_of_supply'].mean()
                category_metrics[category] = {
                    'avg_turnover': avg_turnover,
                    'avg_days_supply': avg_days_supply
                }
    
    # Get pending orders (apply filter if needed)
    pending_orders = data_loader.datasets['pending_orders']
    if category_filter != "All Categories":
        # Filter pending orders by category 
        # First get product IDs in the category
        category_product_ids = filtered_products['product_id'].tolist()
        # Then filter pending orders
        pending_orders = pending_orders[pending_orders['product_id'].isin(category_product_ids)]
    
    total_pending_orders = len(pending_orders)
    
    # Calculate total pending order value
    pending_order_value = 0
    if not pending_orders.empty:
        for _, order in pending_orders.iterrows():
            product_id = order['product_id']
            order_quantity = order['quantity']
            product_info = products_data[products_data['product_id'] == product_id]
            if not product_info.empty:
                product_cost = product_info.iloc[0]['base_cost']
                pending_order_value += order_quantity * product_cost
    
    # Create metrics row - true horizontal layout
    st.markdown("""
    <div style="display: flex; flex-direction: row; justify-content: space-between; gap: 0.2rem; margin-bottom: 0.5rem;">
        <div style="flex: 1; background-color: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 0.3rem 0.4rem; border-left: 2px solid #1a5fb4;">
            <div style="color: #6c757d; font-size: 0.7rem; font-weight: 600; margin-bottom: 0.1rem;">Total Products</div>
            <div style="color: #212529; font-size: 1rem; font-weight: 700;">{}</div>
        </div>
        <div style="flex: 1; background-color: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 0.3rem 0.4rem; border-left: 2px solid #1a5fb4;">
            <div style="color: #6c757d; font-size: 0.7rem; font-weight: 600; margin-bottom: 0.1rem;">Product Categories</div>
            <div style="color: #212529; font-size: 1rem; font-weight: 700;">{}</div>
        </div>
        <div style="flex: 1; background-color: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 0.3rem 0.4rem; border-left: 2px solid #1a5fb4;">
            <div style="color: #6c757d; font-size: 0.7rem; font-weight: 600; margin-bottom: 0.1rem;">Total Inventory Value</div>
            <div style="color: #212529; font-size: 1rem; font-weight: 700;">${:,.2f}</div>
        </div>
        <div style="flex: 1; background-color: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 0.3rem 0.4rem; border-left: 2px solid #1a5fb4;">
            <div style="color: #6c757d; font-size: 0.7rem; font-weight: 600; margin-bottom: 0.1rem;">Pending Orders</div>
            <div style="color: #212529; font-size: 1rem; font-weight: 700;">{} (${:,.2f})</div>
        </div>
    </div>
    """.format(
        total_products, 
        total_categories, 
        total_inventory_value, 
        total_pending_orders, 
        pending_order_value
    ), unsafe_allow_html=True)
    
    # Display filter applied indicator
    if category_filter != "All Categories":
        st.markdown(f"""
        <div style="background-color: #e7f3fe; padding: 0.3rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.8rem;">
            <b>Filter:</b> Showing data for category "<span style="color: #1a5fb4; font-weight: bold;">{category_filter}</span>" 
            <span style="color: #666; margin-left: 0.5rem;">({total_products} products)</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different inventory views
    inventory_tabs = st.tabs(["Inventory Analysis", "Turnover Metrics", "Category Performance", "Stockout Risk", "Supplier Analytics", "Store Analytics"])
    
    with inventory_tabs[0]:
        if all_inventory_value:
            # Create inventory value dataframe
            inventory_df = pd.DataFrame(all_inventory_value)
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Top products by inventory value
                st.markdown("### Top Products by Inventory Value")
                
                # Sort by inventory value
                top_inventory = inventory_df.sort_values('inventory_value', ascending=False).head(10)
                
                # Create horizontal bar chart
                fig = px.bar(
                    top_inventory,
                    y='product_name',
                    x='inventory_value',
                    color='category',
                    orientation='h',
                    height=320,
                    labels={
                        'product_name': 'Product',
                        'inventory_value': 'Inventory Value ($)',
                        'category': 'Category'
                    }
                )
                
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Inventory value by category
                st.markdown("### Inventory by Category")
                
                # Group by category
                category_inventory = inventory_df.groupby('category').agg({
                    'inventory_value': 'sum',
                    'product_id': 'count'
                }).reset_index()
                
                category_inventory.rename(columns={'product_id': 'product_count'}, inplace=True)
                
                # Create pie chart
                fig = px.pie(
                    category_inventory,
                    values='inventory_value',
                    names='category',
                    hover_data=['product_count'],
                    height=320,
                    hole=0.4
                )
                
                fig.update_layout(margin=dict(l=3, r=3, t=15, b=3))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No inventory data available for analysis.")
    
    with inventory_tabs[1]:  # Turnover Metrics tab
        if turnover_data:
            # Create turnover dataframe
            turnover_df = pd.DataFrame(turnover_data)
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Top products by quarterly turnover
                st.markdown("### Inventory Turnover by Product")
                
                # Sort by quarterly turnover
                top_turnover = turnover_df.sort_values('quarterly_turnover', ascending=False).head(15)
                
                # Create horizontal bar chart
                fig = px.bar(
                    top_turnover,
                    y='product_name',
                    x='quarterly_turnover',
                    color='category',
                    orientation='h',
                    height=400,
                    labels={
                        'product_name': 'Product',
                        'quarterly_turnover': 'Quarterly Turnover',
                        'category': 'Category'
                    }
                )
                
                fig.update_layout(
                    title="Products with Highest Turnover Rates",
                    margin=dict(l=5, r=5, t=30, b=5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Days of supply visualization
                st.markdown("### Days of Supply Distribution")
                
                # Create histogram of days of supply
                fig = px.histogram(
                    turnover_df, 
                    x='days_of_supply',
                    color_discrete_sequence=[COLOR_PALETTE['primary']],
                    nbins=20,
                    height=400,
                    labels={'days_of_supply': 'Days of Supply'}
                )
                
                fig.update_layout(
                    title="Distribution of Days of Supply",
                    margin=dict(l=5, r=5, t=30, b=5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional turnover analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot of days of supply vs turnover
                st.markdown("### Days of Supply vs. Turnover")
                
                # Create scatter plot
                fig = px.scatter(
                    turnover_df,
                    x='days_of_supply',
                    y='quarterly_turnover',
                    color='category',
                    size='demand_volatility',
                    hover_name='product_name',
                    height=350,
                    labels={
                        'days_of_supply': 'Days of Supply',
                        'quarterly_turnover': 'Quarterly Turnover',
                        'category': 'Category',
                        'demand_volatility': 'Demand Volatility (%)'
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot of turnover by category
                st.markdown("### Turnover Range by Category")
                
                # Create box plot 
                fig = px.box(
                    turnover_df,
                    x='category',
                    y='quarterly_turnover',
                    color='category',
                    height=350,
                    points="all",
                    labels={
                        'category': 'Category',
                        'quarterly_turnover': 'Quarterly Turnover'
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No turnover data available for analysis.")
    
    with inventory_tabs[2]:  # Category Performance tab
        if turnover_data and category_metrics:
            # Create category metrics dataframe
            category_df = pd.DataFrame([
                {'category': cat, **metrics}
                for cat, metrics in category_metrics.items()
            ])
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Average turnover by category
                st.markdown("### Average Turnover by Category")
                
                # Sort by avg turnover
                category_sorted = category_df.sort_values('avg_turnover', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    category_sorted,
                    x='category',
                    y='avg_turnover',
                    color='category',
                    height=350,
                    labels={
                        'category': 'Category',
                        'avg_turnover': 'Average Quarterly Turnover'
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average days of supply by category
                st.markdown("### Average Days of Supply by Category")
                
                # Sort by avg days supply
                category_sorted = category_df.sort_values('avg_days_supply')
                
                # Create bar chart
                fig = px.bar(
                    category_sorted,
                    x='category',
                    y='avg_days_supply',
                    color='category',
                    height=350,
                    labels={
                        'category': 'Category',
                        'avg_days_supply': 'Average Days of Supply'
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance quadrant analysis
            st.markdown("### Category Performance Quadrant Analysis")
            
            # Create quadrant chart (bubble chart)
            # Higher turnover and lower days of supply is generally better
            fig = px.scatter(
                category_df,
                x='avg_days_supply',
                y='avg_turnover',
                color='category',
                size=[len(turnover_df[turnover_df['category'] == cat]) for cat in category_df['category']],
                hover_name='category',
                height=400,
                labels={
                    'avg_days_supply': 'Average Days of Supply',
                    'avg_turnover': 'Average Quarterly Turnover',
                    'category': 'Category'
                }
            )
            
            # Add quadrant lines
            # Calculate medians for quadrant lines
            median_turnover = category_df['avg_turnover'].median()
            median_days = category_df['avg_days_supply'].median()
            
            # Add vertical line for median days
            fig.add_shape(
                type="line",
                x0=median_days, y0=0,
                x1=median_days, y1=category_df['avg_turnover'].max() * 1.1,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Add horizontal line for median turnover
            fig.add_shape(
                type="line",
                x0=0, y0=median_turnover,
                x1=category_df['avg_days_supply'].max() * 1.1, y1=median_turnover,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Add quadrant annotations
            fig.add_annotation(
                x=median_days / 2,
                y=median_turnover * 1.5,
                text="High Turnover<br>Low Inventory<br>(Optimal)",
                showarrow=False,
                font=dict(size=10, color=COLOR_PALETTE['success'])
            )
            
            fig.add_annotation(
                x=median_days * 1.5,
                y=median_turnover * 1.5,
                text="High Turnover<br>High Inventory<br>(Over-purchasing)",
                showarrow=False,
                font=dict(size=10, color=COLOR_PALETTE['warning'])
            )
            
            fig.add_annotation(
                x=median_days / 2,
                y=median_turnover / 2,
                text="Low Turnover<br>Low Inventory<br>(Underperforming)",
                showarrow=False,
                font=dict(size=10, color=COLOR_PALETTE['warning'])
            )
            
            fig.add_annotation(
                x=median_days * 1.5,
                y=median_turnover / 2,
                text="Low Turnover<br>High Inventory<br>(Poor Performance)",
                showarrow=False,
                font=dict(size=10, color=COLOR_PALETTE['danger'])
            )
            
            # Improve layout
            fig.update_layout(
                title="Category Performance Quadrant Analysis",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No category performance data available for analysis.")
    
    with inventory_tabs[3]:  # Stockout Risk tab
        if stockout_risk_data:
            # Create stockout risk dataframe
            stockout_df = pd.DataFrame(stockout_risk_data)
            
            # Add count of products at risk
            high_risk_count = len(stockout_df[stockout_df['stockout_risk'] == 'High'])
            medium_risk_count = len(stockout_df[stockout_df['stockout_risk'] == 'Medium'])
            low_risk_count = len(stockout_df[stockout_df['stockout_risk'] == 'Low'])
            very_low_risk_count = len(stockout_df[stockout_df['stockout_risk'] == 'Very Low'])
            
            # Display summary metrics
            st.markdown("### Stockout Risk Overview")
            
            # Create metrics row
            cols = st.columns(4)
            
            with cols[0]:
                st.markdown(f'''
                <div style="background-color: rgba(224, 27, 36, 0.15); border-left: 4px solid {COLOR_PALETTE['danger']}; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: {COLOR_PALETTE['danger']};">{high_risk_count}</div>
                    <div style="font-size: 0.85rem; color: #555;">High Risk Products</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f'''
                <div style="background-color: rgba(245, 194, 17, 0.15); border-left: 4px solid {COLOR_PALETTE['warning']}; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: {COLOR_PALETTE['warning']};">{medium_risk_count}</div>
                    <div style="font-size: 0.85rem; color: #555;">Medium Risk Products</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f'''
                <div style="background-color: rgba(46, 194, 126, 0.15); border-left: 4px solid {COLOR_PALETTE['success']}; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: {COLOR_PALETTE['success']};">{low_risk_count}</div>
                    <div style="font-size: 0.85rem; color: #555;">Low Risk Products</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f'''
                <div style="background-color: rgba(28, 113, 216, 0.15); border-left: 4px solid {COLOR_PALETTE['info']}; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: {COLOR_PALETTE['info']};">{very_low_risk_count}</div>
                    <div style="font-size: 0.85rem; color: #555;">Very Low Risk Products</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Create donut chart of stockout risk distribution
                st.markdown("### Stockout Risk Distribution")
                
                # Count products by risk level
                risk_counts = stockout_df['stockout_risk'].value_counts().reset_index()
                risk_counts.columns = ['stockout_risk', 'count']
                
                # Order risk levels
                risk_order = ["High", "Medium", "Low", "Very Low"]
                risk_counts['stockout_risk'] = pd.Categorical(
                    risk_counts['stockout_risk'], 
                    categories=risk_order, 
                    ordered=True
                )
                risk_counts = risk_counts.sort_values('stockout_risk')
                
                # Create custom color map
                color_map = {
                    'High': COLOR_PALETTE['danger'],
                    'Medium': COLOR_PALETTE['warning'],
                    'Low': COLOR_PALETTE['success'],
                    'Very Low': COLOR_PALETTE['info']
                }
                
                # Create donut chart
                fig = px.pie(
                    risk_counts,
                    values='count',
                    names='stockout_risk',
                    hole=0.5,
                    color='stockout_risk',
                    color_discrete_map=color_map,
                    height=350
                )
                
                fig.update_layout(
                    title="Distribution of Stockout Risk",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create histogram of days of supply
                st.markdown("### Days of Supply Distribution")
                
                # Create histogram with color bins by risk level
                fig = go.Figure()
                
                # Add histogram trace
                fig.add_trace(go.Histogram(
                    x=stockout_df['days_of_supply'],
                    nbinsx=30,
                    marker_color=COLOR_PALETTE['primary'],
                    hovertemplate='Days of Supply: %{x}<br>Count: %{y}'
                ))
                
                # Add vertical lines for risk thresholds
                fig.add_shape(
                    type="line",
                    x0=7, y0=0,
                    x1=7, y1=stockout_df['days_of_supply'].value_counts().max() * 1.1,
                    line=dict(color=COLOR_PALETTE['danger'], width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=14, y0=0,
                    x1=14, y1=stockout_df['days_of_supply'].value_counts().max() * 1.1,
                    line=dict(color=COLOR_PALETTE['warning'], width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=30, y0=0,
                    x1=30, y1=stockout_df['days_of_supply'].value_counts().max() * 1.1,
                    line=dict(color=COLOR_PALETTE['success'], width=2, dash="dash")
                )
                
                # Add annotations for risk levels
                fig.add_annotation(
                    x=3.5, y=stockout_df['days_of_supply'].value_counts().max() * 0.9,
                    text="High Risk",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE['danger'], size=10)
                )
                
                fig.add_annotation(
                    x=10.5, y=stockout_df['days_of_supply'].value_counts().max() * 0.9,
                    text="Medium Risk",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE['warning'], size=10)
                )
                
                fig.add_annotation(
                    x=22, y=stockout_df['days_of_supply'].value_counts().max() * 0.9,
                    text="Low Risk",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE['success'], size=10)
                )
                
                fig.update_layout(
                    title="Days of Supply Distribution with Risk Thresholds",
                    xaxis_title="Days of Supply",
                    yaxis_title="Number of Products",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display high risk products
            st.markdown("### Products at High Risk of Stockout")
            
            # Filter for high risk products and sort by days of supply
            high_risk_products = stockout_df[stockout_df['stockout_risk'] == 'High'].sort_values('days_of_supply')
            
            if not high_risk_products.empty:
                # Create a table for high risk products
                high_risk_table = high_risk_products[['product_id', 'product_name', 'category', 'current_inventory', 'avg_daily_demand', 'days_of_supply']]
                
                # Add recommended order column
                high_risk_table['recommended_order'] = (high_risk_table['avg_daily_demand'] * 30 - high_risk_table['current_inventory']).apply(lambda x: max(0, int(x)))
                
                # Style the table
                st.dataframe(
                    high_risk_table.style.format({
                        'current_inventory': '{:.0f}',
                        'avg_daily_demand': '{:.1f}',
                        'days_of_supply': '{:.1f}'
                    }),
                    use_container_width=True
                )
                
                # Action button for high risk products
                if st.button("Generate Purchase Orders for High Risk Products"):
                    st.markdown(f"""
                    <div style="background-color: #d1e7dd; padding: 15px; border-radius: 5px; border-left: 4px solid {COLOR_PALETTE['success']}; margin-top: 15px;">
                        <p style="font-weight: bold; margin-bottom: 5px;">Purchase Orders Generated</p>
                        <p>Purchase orders have been generated for {len(high_risk_products)} high-risk products.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No products at high risk of stockout.")
                
            # Create category risk summary
            st.markdown("### Stockout Risk by Category")
            
            # Count products by category and risk level
            category_risk = stockout_df.groupby(['category', 'stockout_risk']).size().reset_index(name='count')
            
            # Create stacked bar chart
            fig = px.bar(
                category_risk,
                x='category',
                y='count',
                color='stockout_risk',
                color_discrete_map=color_map,
                height=350,
                labels={
                    'category': 'Category',
                    'count': 'Number of Products',
                    'stockout_risk': 'Stockout Risk'
                },
                category_orders={'stockout_risk': risk_order}
            )
            
            fig.update_layout(
                title="Stockout Risk by Category",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stockout risk data available for analysis.")
            
    with inventory_tabs[4]:  # Supplier Analytics tab
        # Get supplier data
        suppliers_data = data_loader.datasets['suppliers']
        product_supplier_mapping = data_loader.datasets['product_supplier_mapping']
        
        if not suppliers_data.empty and not product_supplier_mapping.empty:
            # Create two columns for top-level metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_suppliers = len(suppliers_data)
                st.metric("Total Suppliers", total_suppliers)
            
            with col2:
                avg_lead_time = suppliers_data['average_lead_time_days'].mean()
                st.metric("Average Lead Time", f"{avg_lead_time:.1f} days")
            
            with col3:
                avg_reliability = suppliers_data['reliability_score'].mean()
                st.metric("Average Reliability Score", f"{avg_reliability:.1f}/10")
            
            # Apply category filter to supplier analysis if needed
            if category_filter != "All Categories":
                # Get product IDs in the selected category
                category_product_ids = filtered_products['product_id'].tolist()
                # Filter product-supplier mapping to only include those products
                filtered_mapping = product_supplier_mapping[product_supplier_mapping['product_id'].isin(category_product_ids)]
                # Get the supplier IDs that supply products in this category
                category_supplier_ids = filtered_mapping['supplier_id'].unique()
                # Filter suppliers to only include those that supply this category
                filtered_suppliers = suppliers_data[suppliers_data['supplier_id'].isin(category_supplier_ids)]
            else:
                filtered_suppliers = suppliers_data
                filtered_mapping = product_supplier_mapping
            
            # Create main supplier analysis visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Supplier reliability vs lead time analysis
                st.markdown("### Supplier Reliability vs Lead Time")
                
                # Calculate how many products each supplier provides
                supplier_product_counts = filtered_mapping.groupby('supplier_id').size().reset_index(name='product_count')
                
                # Merge with supplier data
                supplier_analysis = pd.merge(
                    filtered_suppliers,
                    supplier_product_counts,
                    on='supplier_id',
                    how='left'
                )
                
                # Fill NAs for suppliers with no products in the filter
                supplier_analysis['product_count'] = supplier_analysis['product_count'].fillna(0)
                
                # Create a scatter plot of reliability vs lead time
                fig = px.scatter(
                    supplier_analysis,
                    x='average_lead_time_days',
                    y='reliability_score',
                    size='product_count',
                    color='country',
                    hover_name='name',
                    size_max=30,
                    height=350,
                    labels={
                        'average_lead_time_days': 'Average Lead Time (days)',
                        'reliability_score': 'Reliability Score (0-10)',
                        'product_count': 'Number of Products Supplied',
                        'country': 'Country'
                    }
                )
                
                # Add quadrant lines for analysis
                # Vertical line at average lead time
                avg_lead = supplier_analysis['average_lead_time_days'].mean()
                fig.add_vline(x=avg_lead, line_dash="dash", line_color="gray")
                
                # Horizontal line at average reliability score
                avg_reliability = supplier_analysis['reliability_score'].mean()
                fig.add_hline(y=avg_reliability, line_dash="dash", line_color="gray")
                
                # Add quadrant labels
                fig.add_annotation(
                    x=avg_lead/2,
                    y=avg_reliability*1.05,
                    text="Fast & Reliable<br>(Preferred)",
                    showarrow=False,
                    font=dict(size=10, color=COLOR_PALETTE['success'])
                )
                
                fig.add_annotation(
                    x=avg_lead*1.5,
                    y=avg_reliability*1.05,
                    text="Slow but Reliable",
                    showarrow=False,
                    font=dict(size=10, color=COLOR_PALETTE['warning'])
                )
                
                fig.add_annotation(
                    x=avg_lead/2,
                    y=avg_reliability*0.95,
                    text="Fast but Less Reliable",
                    showarrow=False,
                    font=dict(size=10, color=COLOR_PALETTE['warning'])
                )
                
                fig.add_annotation(
                    x=avg_lead*1.5,
                    y=avg_reliability*0.95,
                    text="Slow & Less Reliable<br>(Problematic)",
                    showarrow=False,
                    font=dict(size=10, color=COLOR_PALETTE['danger'])
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Supplier distribution by country
                st.markdown("### Suppliers by Country")
                
                # Count suppliers by country
                country_counts = filtered_suppliers['country'].value_counts().reset_index()
                country_counts.columns = ['country', 'count']
                
                # Create a pie chart
                fig = px.pie(
                    country_counts,
                    values='count',
                    names='country',
                    height=350,
                    hole=0.4
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Product supply analysis
            st.markdown("### Supply Chain Concentration Risk")
            
            # Calculate how many suppliers each product has
            product_supplier_counts = filtered_mapping.groupby('product_id').size().reset_index(name='supplier_count')
            
            # Merge with product data
            product_supplier_analysis = pd.merge(
                filtered_products,
                product_supplier_counts,
                on='product_id',
                how='left'
            )
            
            # Fill NAs for products with no suppliers in the filter
            product_supplier_analysis['supplier_count'] = product_supplier_analysis['supplier_count'].fillna(0)
            
            # Add a risk category
            def get_risk_category(count):
                if count == 0:
                    return "Critical (No Suppliers)"
                elif count == 1:
                    return "High (Single Source)"
                elif count == 2:
                    return "Medium (Dual Source)"
                else:
                    return "Low (Multiple Sources)"
            
            product_supplier_analysis['risk_category'] = product_supplier_analysis['supplier_count'].apply(get_risk_category)
            
            # Order the risk categories
            risk_order = ["Critical (No Suppliers)", "High (Single Source)", "Medium (Dual Source)", "Low (Multiple Sources)"]
            product_supplier_analysis['risk_category'] = pd.Categorical(
                product_supplier_analysis['risk_category'],
                categories=risk_order,
                ordered=True
            )
            
            # Count products by risk category
            risk_counts = product_supplier_analysis['risk_category'].value_counts().reindex(risk_order).reset_index()
            risk_counts.columns = ['risk_category', 'count']
            
            # Create color map for risks
            risk_colors = {
                "Critical (No Suppliers)": COLOR_PALETTE['danger'],
                "High (Single Source)": COLOR_PALETTE['warning'],
                "Medium (Dual Source)": COLOR_PALETTE['info'],
                "Low (Multiple Sources)": COLOR_PALETTE['success']
            }
            
            # Create a horizontal bar chart
            fig = px.bar(
                risk_counts,
                y='risk_category',
                x='count',
                color='risk_category',
                color_discrete_map=risk_colors,
                orientation='h',
                height=250,
                labels={
                    'risk_category': 'Supply Risk Category',
                    'count': 'Number of Products'
                }
            )
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                yaxis=dict(categoryorder='array', categoryarray=risk_order)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display high-risk products table
            high_risk_products = product_supplier_analysis[
                product_supplier_analysis['risk_category'].isin(["Critical (No Suppliers)", "High (Single Source)"])
            ].sort_values('risk_category')
            
            if not high_risk_products.empty:
                st.markdown("### High-Risk Products (Single or No Source)")
                
                # Create a more readable table
                display_df = high_risk_products[['product_id', 'name', 'category', 'supplier_count', 'risk_category']]
                display_df.columns = ['Product ID', 'Product Name', 'Category', 'Supplier Count', 'Risk Level']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No supplier data available for analysis.")
    
    with inventory_tabs[5]:  # Store Analytics tab
        # Get store data
        stores_data = data_loader.datasets['stores']
        
        if not stores_data.empty:
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_stores = len(stores_data)
                st.metric("Total Stores", total_stores)
            
            with col2:
                # Calculate average store age
                current_date = pd.Timestamp.now()
                stores_data['age_days'] = (current_date - stores_data['opening_date']).dt.days
                avg_age_years = stores_data['age_days'].mean() / 365.25
                st.metric("Average Store Age", f"{avg_age_years:.1f} years")
            
            with col3:
                # Store size distribution
                size_counts = stores_data['size_category'].value_counts()
                most_common_size = size_counts.index[0]
                st.metric("Most Common Store Size", most_common_size.title())
            
            # Create visualizations for store analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Store distribution by region
                st.markdown("### Stores by Region")
                
                # Count stores by region
                region_counts = stores_data['region'].value_counts().reset_index()
                region_counts.columns = ['region', 'count']
                
                # Create a bar chart
                fig = px.bar(
                    region_counts,
                    x='region',
                    y='count',
                    color='region',
                    height=350,
                    labels={
                        'region': 'Region',
                        'count': 'Number of Stores'
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Store size distribution
                st.markdown("### Store Size Distribution")
                
                # Count stores by size
                size_counts = stores_data['size_category'].value_counts().reset_index()
                size_counts.columns = ['size_category', 'count']
                
                # Order sizes
                size_order = ["small", "medium", "large"]
                size_counts['size_category'] = pd.Categorical(
                    size_counts['size_category'],
                    categories=size_order,
                    ordered=True
                )
                size_counts = size_counts.sort_values('size_category')
                
                # Create a pie chart
                fig = px.pie(
                    size_counts,
                    values='count',
                    names='size_category',
                    height=350,
                    hole=0.4,
                    color='size_category',
                    color_discrete_map={
                        'small': COLOR_PALETTE['info'],
                        'medium': COLOR_PALETTE['warning'],
                        'large': COLOR_PALETTE['primary']
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Store opening timeline
            st.markdown("### Store Expansion Timeline")
            
            # Sort stores by opening date
            timeline_data = stores_data.sort_values('opening_date')
            
            # Create a running count of stores over time
            timeline_data['store_number'] = range(1, len(timeline_data) + 1)
            
            # Create a line chart of store openings over time
            fig = px.line(
                timeline_data,
                x='opening_date',
                y='store_number',
                height=350,
                labels={
                    'opening_date': 'Date',
                    'store_number': 'Cumulative Number of Stores'
                }
            )
            
            # Add markers for each store opening
            fig.add_trace(
                go.Scatter(
                    x=timeline_data['opening_date'],
                    y=timeline_data['store_number'],
                    mode='markers',
                    name='Store Opening',
                    hovertemplate='%{x|%Y-%m-%d}<br>Store #%{y}: %{text}',
                    text=timeline_data['name'],
                    marker=dict(color=COLOR_PALETTE['success'], size=8)
                )
            )
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store geographic distribution
            st.markdown("### Store Geographic Distribution")
            
            # Create a table with store counts by state
            state_counts = stores_data.groupby(['state', 'region']).size().reset_index(name='count')
            state_counts = state_counts.sort_values(['region', 'count'], ascending=[True, False])
            
            # Create a more visually appealing table
            col1, col2 = st.columns(2)
            
            with col1:
                # Map visualization placeholder (show top states by store count)
                top_states = state_counts.sort_values('count', ascending=False).head(5)
                
                fig = px.bar(
                    top_states,
                    x='state',
                    y='count',
                    color='region',
                    height=350,
                    labels={
                        'state': 'State',
                        'count': 'Number of Stores',
                        'region': 'Region'
                    },
                    title="Top 5 States by Store Count"
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a heat table of store density by region
                region_size_counts = stores_data.groupby(['region', 'size_category']).size().unstack().fillna(0)
                
                # Ensure all size categories are present
                for size in size_order:
                    if size not in region_size_counts.columns:
                        region_size_counts[size] = 0
                
                # Reorder columns
                region_size_counts = region_size_counts[size_order]
                
                # Add a total column
                region_size_counts['total'] = region_size_counts.sum(axis=1)
                
                # Convert to percentage for a stacked bar chart
                region_size_pct = region_size_counts.copy()
                for size in size_order:
                    region_size_pct[size] = region_size_counts[size] / region_size_counts['total'] * 100
                
                # Create a stacked bar chart
                region_size_pct = region_size_pct.reset_index()
                region_size_pct = pd.melt(
                    region_size_pct, 
                    id_vars=['region'],
                    value_vars=size_order,
                    var_name='size_category',
                    value_name='percentage'
                )
                
                fig = px.bar(
                    region_size_pct,
                    x='region',
                    y='percentage',
                    color='size_category',
                    height=350,
                    labels={
                        'region': 'Region',
                        'percentage': 'Percentage of Stores',
                        'size_category': 'Store Size'
                    },
                    title="Store Size Distribution by Region",
                    color_discrete_map={
                        'small': COLOR_PALETTE['info'],
                        'medium': COLOR_PALETTE['warning'],
                        'large': COLOR_PALETTE['primary']
                    }
                )
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(ticksuffix='%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Show full store list in an expandable section
            with st.expander("View Complete Store List"):
                display_df = stores_data[['store_id', 'name', 'city', 'state', 'region', 'size_category', 'opening_date']]
                display_df.columns = ['Store ID', 'Store Name', 'City', 'State', 'Region', 'Size', 'Opening Date']
                
                # Format the date
                display_df['Opening Date'] = display_df['Opening Date'].dt.strftime('%Y-%m-%d')
                
                # Capitalize size category
                display_df['Size'] = display_df['Size'].str.capitalize()
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No store data available for analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)


# Add this function to display forecast quality information
def display_forecast_quality(forecast, historical_data=None):
    """
    Display forecast quality information
    
    Parameters:
    -----------
    forecast : pandas.DataFrame
        Forecast DataFrame
    historical_data : pandas.DataFrame, optional
        Historical demand data
    """
    if forecast is None:
        return
    
    # Create a container for forecast quality information
    quality_container = st.container()
    
    with quality_container:
        # Check if forecast has quality attributes
        if hasattr(forecast, 'attrs') and 'quality' in forecast.attrs:
            quality = forecast.attrs['quality']
            quality_color = forecast.attrs['quality_color']
            quality_explanation = forecast.attrs['quality_explanation']
            confidence_level = forecast.attrs.get('confidence_level', 0.15) * 100
            
            # Create columns for quality metrics
            col1, col2, col3 = st.columns(3)
            
            # Display quality indicator
            with col1:
                st.markdown(f"""
                <div style="padding:10px; background-color:#f8f9fa; border-radius:5px; border:1px solid #ddd;">
                    <span style="font-weight:bold;">Forecast Quality:</span> 
                    <span style="color:{quality_color}; font-weight:bold;">{quality}</span>
                    <div style="font-size:0.9em; color:#666; margin-top:5px;">{quality_explanation}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display confidence interval
            with col2:
                st.markdown(f"""
                <div style="padding:10px; background-color:#f8f9fa; border-radius:5px; border:1px solid #ddd;">
                    <span style="font-weight:bold;">Confidence Interval:</span> 
                    <span style="font-weight:bold;">±{confidence_level:.1f}%</span>
                    <div style="font-size:0.9em; color:#666; margin-top:5px;">Range within which actual values are expected to fall</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display coefficient of variation if available
            with col3:
                if 'coefficient_of_variation' in forecast.attrs:
                    cv = forecast.attrs['coefficient_of_variation'] * 100
                    cv_level = "Low" if cv < 20 else "Medium" if cv < 50 else "High"
                    cv_color = "green" if cv < 20 else "orange" if cv < 50 else "red"
                    
                    st.markdown(f"""
                    <div style="padding:10px; background-color:#f8f9fa; border-radius:5px; border:1px solid #ddd;">
                        <span style="font-weight:bold;">Historical Variability:</span> 
                        <span style="color:{cv_color}; font-weight:bold;">{cv_level} ({cv:.1f}%)</span>
                        <div style="font-size:0.9em; color:#666; margin-top:5px;">Coefficient of Variation in historical demand</div>
                    </div>
                    """, unsafe_allow_html=True)
        elif historical_data is not None:
            # Calculate basic statistics for historical data
            historical_mean = historical_data['demand'].mean()
            historical_std = historical_data['demand'].std()
            
            if historical_mean > 0:
                cv = (historical_std / historical_mean) * 100
                forecast_mean = forecast['forecast'].mean() if 'forecast' in forecast.columns else 0
                
                # Compare forecast mean with historical mean
                percent_diff = ((forecast_mean - historical_mean) / historical_mean) * 100
                diff_direction = "higher" if percent_diff > 0 else "lower"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"Historical demand variability: CV = {cv:.1f}%")
                
                with col2:
                    st.info(f"Forecast is {abs(percent_diff):.1f}% {diff_direction} than historical average")
        else:
            st.warning("Limited quality information available for this forecast")


# Main dashboard application
def main():
    """Main function to render the dashboard"""
    # Apply custom styling
    set_custom_theme()
    
    # Initialize session state for storing forecasts
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    
    # Create sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/supply-chain.png", width=80)
        st.title("Supply Chain Analytics")
        
        # Navigation
        selected_page = st.radio(
            "Navigation",
            ["Supply Chain Overview", "Demand Forecasting", "Inventory Optimization"]
        )
    
    # Initialize data loader with error handling
    try:
        data_loader = DataLoader()
        data_loader.load_all_datasets()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Create inventory optimizer
    try:
        inventory_optimizer = InventoryOptimizer(data_loader)
    except Exception as e:
        st.error(f"Error initializing inventory optimizer: {str(e)}")
        return
    
    # Create model selector
    try:
        feature_engineering = FeatureEngineering(data_loader)
        model_selector = ModelSelector(data_loader, feature_engineering)
    except Exception as e:
        st.error(f"Error initializing model selector: {str(e)}")
        return
    
    # Render selected page
    try:
        if selected_page == "Supply Chain Overview":
            supply_chain_overview()
        elif selected_page == "Demand Forecasting":
            demand_forecasting_tab()  # Use our new improved tab
        elif selected_page == "Inventory Optimization":
            inventory_optimization()
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")
        st.error("Please check the console for more details.")

def supply_chain_overview():
    """
    Display the supply chain overview dashboard
    """
    # Initialize data loader with error handling
    try:
        data_loader = DataLoader()
        data_loader.load_all_datasets()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Create inventory optimizer
    try:
        inventory_optimizer = InventoryOptimizer(data_loader)
    except Exception as e:
        st.error(f"Error initializing inventory optimizer: {str(e)}")
        return
    
    # Display the overview dashboard
    display_overview_dashboard(data_loader, inventory_optimizer, category_filter="All Categories")

def inventory_optimization():
    """
    Display the inventory optimization dashboard
    """
    # Initialize data loader with error handling
    try:
        data_loader = DataLoader()
        data_loader.load_all_datasets()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Create inventory optimizer
    try:
        inventory_optimizer = InventoryOptimizer(data_loader)
    except Exception as e:
        st.error(f"Error initializing inventory optimizer: {str(e)}")
        return
    
    # Display the inventory optimization dashboard
    display_inventory_optimization_dashboard(data_loader, inventory_optimizer)

def display_forecasting_dashboard(data_loader, model_selector):
    """
    Display demand forecasting dashboard
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of DataLoader class
    model_selector : ModelSelector
        Instance of ModelSelector class
    """
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("## Demand Forecasting")
    
    # Get available products with demand data
    product_options = data_loader.get_products_with_demand()
    
    if not product_options:
        st.error("No products with demand data found. Please check your data.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create layout with columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Product selection and forecast parameters
        st.markdown("### Select Product")
        
        # Product dropdown
        selected_product = st.selectbox(
            "Product",
            options=product_options,
            format_func=lambda x: data_loader.get_product_name(x),
            key="forecast_product_select"
        )
        
        # Forecast parameters
        st.markdown("### Forecast Settings")
        
        forecast_periods = st.number_input(
            "Forecast Horizon (Days)",
            min_value=7,
            max_value=365,
            value=30,
            step=1,
            key="forecast_periods"
        )
        
        # Model selection
        st.markdown("### Model Selection")
        
        available_models = ["Best Model", "Linear Regression", "Random Forest", "Gradient Boosting", "Prophet"]
        selected_model_type = st.radio(
            "Model Type",
            options=available_models,
            key="forecast_model_type"
        )
        
        # Train model button
        train_button = st.button("Train & Forecast", key="train_forecast_button")
    
    with col2:
        # Help text
        st.info("""
        **How to use this dashboard:**
        1. Select a product from the dropdown
        2. Choose forecast horizon (number of days to forecast)
        3. Select a model type
        4. Click 'Train & Forecast' to generate forecast
        """)
        
        # Generate forecast when button is clicked
        if train_button:
            with st.spinner("Training models and generating forecast..."):
                try:
                    # Train models for the selected product
                    models_dict = train_models(model_selector, selected_product)
                    
                    # Attempt to use the selected model type
                    model = None
                    forecast_df = None
                    
                    if models_dict:
                        # Try to get the appropriate model based on selection
                        if selected_model_type == "Best Model" and "best_model" in models_dict:
                            model = models_dict["best_model"]
                            st.success("Using best trained model for forecast")
                        
                        elif selected_model_type == "Linear Regression":
                            # Try different possible keys for linear regression
                            for key in ["linear_regression", "LinearRegression"]:
                                if key in models_dict:
                                    model = models_dict[key]
                                    st.success(f"Using Linear Regression model for forecast")
                                    break
                        
                        elif selected_model_type == "Random Forest":
                            # Try different possible keys for random forest
                            for key in ["random_forest", "RandomForestRegressor"]:
                                if key in models_dict:
                                    model = models_dict[key]
                                    st.success(f"Using Random Forest model for forecast")
                                    break
                        
                        elif selected_model_type == "Gradient Boosting":
                            # Try different possible keys for gradient boosting
                            for key in ["gradient_boosting", "GradientBoostingRegressor"]:
                                if key in models_dict:
                                    model = models_dict[key]
                                    st.success(f"Using Gradient Boosting model for forecast")
                                    break
                        
                        elif selected_model_type == "Prophet":
                            if "prophet" in models_dict:
                                model = models_dict["prophet"]
                                st.success(f"Using Prophet model for forecast")
                        
                        # If selected model not found, try to use any available model
                        if model is None:
                            available_keys = list(models_dict.keys())
                            if available_keys:
                                first_key = available_keys[0]
                                model = models_dict[first_key]
                                st.warning(f"Selected model not available. Using {first_key} as fallback model.")
                            else:
                                st.error("No models available for this product.")
                    else:
                        st.error("Failed to train models for this product.")
                    
                    # Generate forecast if we have a valid model
                    if model is not None:
                        # Check if model is actually a valid object with predict method
                        if hasattr(model, 'predict'):
                            forecast_df = generate_forecast(model_selector, model, selected_product, periods=forecast_periods)
                        else:
                            st.error(f"Invalid model object (type: {type(model)}). Cannot generate forecast.")
                            # Create a fallback forecast based on historical data
                            historical_data = data_loader.get_demand_by_product(selected_product)
                            if not historical_data.empty:
                                # Calculate historical mean and std
                                hist_mean = historical_data['demand'].mean()
                                hist_std = historical_data['demand'].std() 
                                
                                # Generate future dates
                                last_date = historical_data['date'].max()
                                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                                
                                # Generate random forecast values based on historical patterns
                                np.random.seed(42)  # For reproducibility
                                forecast_values = hist_mean * (0.8 + 0.4 * np.random.random(len(future_dates)))
                                
                                # Create forecast dataframe
                                forecast_df = pd.DataFrame({
                                    'date': future_dates,
                                    'product_id': selected_product,
                                    'forecast': forecast_values,
                                    'lower_ci': forecast_values * 0.8,
                                    'upper_ci': forecast_values * 1.2
                                })
                                st.warning("Using fallback forecast based on historical patterns")
                    
                    # Display the forecast if available
                    if forecast_df is not None and not forecast_df.empty:
                        # Display forecast chart
                        st.markdown("### Demand Forecast")
                        
                        # Get historical data for chart
                        historical_data = model_selector.data_loader.get_demand_by_product(selected_product)
                        
                        if not historical_data.empty:
                            # Create tabs for different views
                            forecast_tabs = st.tabs(["Forecast Chart", "Forecast Details", "Forecast Quality", "Inventory Implications"])
                            
                            with forecast_tabs[0]:
                                # Add time period filters
                                st.markdown("#### Time Period")
                                time_filter = st.radio(
                                    "Select time range to display:",
                                    ["Last 3 Months + Forecast", "Last Month + Forecast", "Last 6 Months + Forecast", "Last Year + Forecast", "All Data"],
                                    horizontal=True,
                                    index=0  # Default to Last 3 Months + Forecast
                                )
                                
                                # Apply time filters
                                filtered_historical = historical_data.copy()
                                
                                # Current date for filtering
                                current_date = pd.Timestamp(time.time(), unit='s')
                                
                                if time_filter == "Last Month + Forecast":
                                    start_date = current_date - pd.Timedelta(days=30)
                                    filtered_historical = filtered_historical[filtered_historical['date'] >= start_date]
                                elif time_filter == "Last 3 Months + Forecast":
                                    start_date = current_date - pd.Timedelta(days=90)
                                    filtered_historical = filtered_historical[filtered_historical['date'] >= start_date]
                                elif time_filter == "Last 6 Months + Forecast":
                                    start_date = current_date - pd.Timedelta(days=180)
                                    filtered_historical = filtered_historical[filtered_historical['date'] >= start_date]
                                elif time_filter == "Last Year + Forecast":
                                    start_date = current_date - pd.Timedelta(days=365)
                                    filtered_historical = filtered_historical[filtered_historical['date'] >= start_date]
                                
                                # Plot forecast with confidence intervals
                                fig = go.Figure()
                                
                                # Add historical data
                                fig.add_trace(go.Scatter(
                                    x=filtered_historical['date'],
                                    y=filtered_historical['demand'],
                                    mode='lines',
                                    name='Historical Demand',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Add forecast data - only if we have forecast data
                                if forecast_df is not None and not forecast_df.empty and 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=forecast_df['date'],
                                        y=forecast_df['forecast'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='#ff7f0e', width=3)
                                    ))
                                    
                                    # Add confidence intervals
                                    if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                                            y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
                                            fill='toself',
                                            fillcolor='rgba(255,127,14,0.2)',
                                            line=dict(color='rgba(255,127,14,0)'),
                                            name='Confidence Interval'
                                        ))
                                
                                # Update layout for a clean, modern look
                                fig.update_layout(
                                    title=f"Demand Forecast for {data_loader.get_product_name(selected_product)}",
                                    xaxis_title="Date",
                                    yaxis_title="Demand",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="center",
                                        x=0.5
                                    ),
                                    template="plotly_white",
                                    margin=dict(l=10, r=10, t=60, b=10),
                                    height=500
                                )
                                
                                # Display the chart
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add a vertical line to mark forecast start
                                fig.add_shape(
                                    type="line",
                                    x0=forecast_df['date'].min(),
                                    y0=0,
                                    x1=forecast_df['date'].min(),
                                    y1=1,
                                    yref="paper",
                                    line=dict(color="gray", width=1, dash="dash"),
                                )
                                
                                # Add annotation for forecast start
                                fig.add_annotation(
                                    x=forecast_df['date'].min(),
                                    y=1.05,
                                    yref="paper",
                                    text="Forecast Start",
                                    showarrow=False,
                                    font=dict(color="gray")
                                )
                    else:
                        st.error("Failed to generate forecast. Please try again with a different product or more historical data.")
                
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            # Show placeholder when no forecast is generated
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6c757d;">
                <h3>Demand Forecast</h3>
                <p>Select a product and click "Train & Forecast" to generate a demand forecast.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_current_inventory(data_loader, product_id):
    """
    Get current inventory level for a product
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of DataLoader class
    product_id : str
        Product ID
        
    Returns:
    --------
    float: Current inventory level
    """
    inventory_data = data_loader.get_inventory_by_product(product_id)
    
    if inventory_data.empty:
        return 0
    
    # Get latest date
    latest_date = inventory_data['date'].max()
    
    # Get inventory on latest date
    latest_inventory = inventory_data[inventory_data['date'] == latest_date]
    
    if latest_inventory.empty:
        return 0
    
    return latest_inventory.iloc[0]['ending_inventory']

def train_models(model_selector, product_id):
    """
    Train demand forecasting models for a product
    
    Parameters:
    -----------
    model_selector : ModelSelector
        Instance of ModelSelector class
    product_id : str
        Product ID
        
    Returns:
    --------
    dict: Dictionary containing trained models
    """
    st.info(f"Training forecasting models for product {product_id}. This may take a minute...")
    
    # Show a spinner while training
    with st.spinner("Training models..."):
        # First check if we have meaningful data to train on
        hist_data = model_selector.data_loader.get_demand_by_product(product_id)
        
        if hist_data.empty:
            st.error(f"No historical data found for product {product_id}")
            return None
            
        # Check data quality
        if len(hist_data) < 10:
            st.warning(f"Limited data available ({len(hist_data)} points). Results may not be reliable.")
        
        # Check the variability and data patterns
        hist_mean = hist_data['demand'].mean()
        hist_std = hist_data['demand'].std()
        zero_pct = (hist_data['demand'] == 0).mean() * 100
        
        if zero_pct > 70:
            st.warning(f"Data contains {zero_pct:.1f}% zero values. This may affect forecast accuracy.")
        
        # Choose training approach based on data characteristics
        if hist_std / hist_mean < 0.1 and len(hist_data) > 30:
            # Very stable demand - simpler models work better
            st.info("Data shows stable demand patterns. Using specialized model selection.")
            # Only train simpler models for stable demand
            available_models = ["linear_regression", "gradient_boosting"]
            noise_level = 0.02  # Less noise for stable data
        else:
            # More variable demand - use all models
            available_models = None  # Use all available models
            noise_level = 0.05  # Regular noise level
        
        try:
            # Train models with more robust settings
            results = model_selector.train_demand_forecasting_models(
                product_id, 
                models=available_models, 
                test_size=0.2,  # Use 20% of data for testing
                noise_level=noise_level  # Add slight noise to prevent overfitting
            )
            
            # Ensure we're getting a dictionary with models
            if not isinstance(results, dict):
                st.error(f"Expected a dictionary of models but got {type(results)}")
                return None
                
            # Check if there are any valid models in the results
            valid_models = {}
            
            # Process the results to make sure we're returning actual model objects
            for model_name, model_data in results.items():
                if isinstance(model_data, dict) and 'model' in model_data:
                    # Extract the actual model object
                    model_obj = model_data['model']
                    
                    # Ensure it's a valid model object with a predict method
                    if hasattr(model_obj, 'predict'):
                        valid_models[model_name] = model_obj
                        
                        # Add historical stats to the model for reference
                        if hasattr(model_obj, 'mean_target'):
                            model_obj.historical_mean = hist_mean
                            model_obj.historical_std = hist_std
                            model_obj.min_forecast = max(1.0, hist_mean * 0.3)
                    else:
                        st.warning(f"Model {model_name} doesn't have a predict method")
            
            # Determine best model
            if 'best_model' in results and isinstance(results['best_model'], str):
                # If best_model is a string, it's the name of the best model
                best_model_name = results['best_model']
                if best_model_name in valid_models:
                    valid_models['best_model'] = valid_models[best_model_name]
                else:
                    st.warning(f"Best model '{best_model_name}' not found in valid models")
                    
                    # If best model isn't available, use the first valid model as best
                    if valid_models:
                        first_model_name = next(iter(valid_models))
                        valid_models['best_model'] = valid_models[first_model_name]
                        st.info(f"Using {first_model_name} as best model")
            
            if not valid_models:
                st.error("No valid models found. Check model training process.")
                return None
            
            return valid_models
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def generate_forecast(model_selector, model, product_id, periods=30):
    """
    Wrapper to generate forecast and handle errors
    
    Parameters:
    -----------
    model_selector : ModelSelector
        Instance of ModelSelector class
    model : BaseForecastModel or Prophet
        Trained model
    product_id : str
        Product ID
    periods : int
        Number of periods to forecast
        
    Returns:
    --------
    pandas.DataFrame: Forecast DataFrame or None on error
    """
    try:
        # Generate forecast
        forecast = model_selector.forecast_demand(model, product_id, periods=periods)
        
        if forecast is None or forecast.empty:
            st.error("Failed to generate forecast")
            return None
            
        # Ensure forecast has proper date format to avoid Timestamp arithmetic issues
        if 'date' in forecast.columns:
            forecast['date'] = pd.to_datetime(forecast['date'])
        
        # Check and standardize column names
        # Sometimes the forecast column may be named differently
        if 'forecast' not in forecast.columns:
            if 'demand_forecast' in forecast.columns:
                forecast['forecast'] = forecast['demand_forecast']
            elif 'yhat' in forecast.columns:
                forecast['forecast'] = forecast['yhat']
            
        # Check if forecast values are reasonable (not all zeros or very small values)
        if 'forecast' in forecast.columns:
            # Get historical data to compare
            historical_data = model_selector.data_loader.get_demand_by_product(product_id)
            hist_mean = historical_data['demand'].mean() if not historical_data.empty else 1.0
            hist_min = historical_data['demand'].min() if not historical_data.empty else 0.0
            hist_max = historical_data['demand'].max() if not historical_data.empty else 10.0
            hist_median = historical_data['demand'].median() if not historical_data.empty else 1.0
            
            # Calculate reasonable range for forecasts based on historical data
            reasonable_min = max(1.0, hist_min * 0.7)
            reasonable_max = hist_max * 1.2
            
            mean_forecast = forecast['forecast'].mean() 
            
            # Calculate scaling factors more intelligently
            forecast_range = forecast['forecast'].max() - forecast['forecast'].min()
            hist_range = hist_max - hist_min
            
            # MAJOR FIX: Check if forecast is significantly different from historical patterns
            if mean_forecast < hist_mean * 0.5 or mean_forecast > hist_mean * 1.5:
                st.warning(f"Forecast values are different from historical average. Historical: {hist_mean:.2f}, Forecast: {mean_forecast:.2f}")
                
                # Scaling approach 1: Direct mean-based scaling
                scaling_factor_1 = hist_mean / max(mean_forecast, 0.1)  # Avoid division by zero
                
                # Scaling approach 2: Range-preserving scaling
                if forecast_range > 0:
                    baseline = hist_mean - (mean_forecast * hist_range / forecast_range)
                    scaling_factor_2 = hist_range / forecast_range
                else:
                    baseline = hist_mean * 0.7
                    scaling_factor_2 = 1.0
                
                # Choose best scaling approach
                if abs(scaling_factor_1 - 1.0) < abs(scaling_factor_2 - 1.0) and forecast_range > 0:
                    # Simple scaling is less extreme, use it
                    scaling_factor = min(max(scaling_factor_1, 0.5), 10.0)  # Limit scaling between 0.5x and 10x
                    forecast['forecast'] = forecast['forecast'] * scaling_factor
                    st.info(f"Adjusted forecast using mean-scaling (factor: {scaling_factor:.2f})")
                else:
                    # Use range-preserving scaling for more dynamic patterns
                    forecast['forecast'] = baseline + (forecast['forecast'] * scaling_factor_2)
                    st.info(f"Adjusted forecast using range-preserving approach")
                
                # If still too low, use historical pattern-based correction
                if forecast['forecast'].mean() < hist_mean * 0.5:
                    historical_pattern = np.array(historical_data['demand'])
                    # Get a sample pattern from history
                    if len(historical_pattern) > len(forecast):
                        # Choose a random segment of the right length
                        start = np.random.randint(0, len(historical_pattern) - len(forecast))
                        pattern = historical_pattern[start:start+len(forecast)]
                    else:
                        # Sample with replacement to get enough data
                        pattern = np.random.choice(historical_pattern, size=len(forecast))
                    
                    # Blend the forecast with historical pattern
                    blend_ratio = 0.7  # 70% historical pattern, 30% model forecast
                    forecast['forecast'] = (blend_ratio * pattern) + ((1-blend_ratio) * forecast['forecast'].values)
                    st.info(f"Applied historical pattern blending to create more realistic forecast")
            
            # Check for unreasonably low values
            if forecast['forecast'].min() < reasonable_min:
                # Adjust the minimum values
                baseline = max(1.0, hist_median * 0.5)
                forecast['forecast'] = forecast['forecast'].clip(lower=baseline)
            
            # Ensure that forecast values stay positive
            forecast['forecast'] = forecast['forecast'].clip(lower=0)
            
            # Add confidence intervals if missing
            if 'lower_ci' not in forecast.columns or 'upper_ci' not in forecast.columns:
                if not historical_data.empty:
                    # Calculate historical volatility to set confidence interval width
                    historical_std = historical_data['demand'].std()
                    
                    if hist_mean > 0:
                        cv = historical_std / hist_mean
                        # Higher volatility = wider bounds, with min 15% and max 50%
                        confidence_level = min(0.5, max(0.15, cv))
                    else:
                        confidence_level = 0.2  # Default 20% bounds
                    
                    forecast['lower_ci'] = (forecast['forecast'] * (1 - confidence_level)).clip(0)
                    forecast['upper_ci'] = forecast['forecast'] * (1 + confidence_level)
                else:
                    # Use default bounds of ±20%
                    forecast['lower_ci'] = (forecast['forecast'] * 0.8).clip(0)
                    forecast['upper_ci'] = forecast['forecast'] * 1.2
        
        return forecast
        
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def display_inventory_optimization_dashboard(data_loader, inventory_optimizer):
    """
    Display inventory optimization dashboard
    
    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader instance
    inventory_optimizer : InventoryOptimizer
        InventoryOptimizer instance
    """
    st.header("Inventory Optimization")
    
    # Create a larger gap
    st.write("")
    
    st.subheader("Optimization Settings")
    
    # Load list of products
    products_data = data_loader.load_products_data()
    product_options = [(row['product_id'], f"{row['name']} ({row['product_id']})") 
                      for _, row in products_data.iterrows()]
    
    # Create a 2-column layout for better alignment
    col1, col2 = st.columns([1, 1])
        
        with col1:
        # Select product - fix the empty label warning
        selected_product = st.selectbox(
                "Select Product",
            options=[pid for pid, _ in product_options],
            format_func=lambda x: next((name for pid, name in product_options if pid == x), x)
        )
    
    # Create some space between the select box and the button
    st.write("")
    
    # Create a button to run optimization
    optimize_button = st.button("Optimize Inventory", type="primary", use_container_width=True)
    
    # Remove the "Getting Started" section
    
    # Create a larger gap
    st.write("")
    st.write("")
    
    # Add batch optimization section
    st.subheader("Batch Optimization")
    st.write("Optimize inventory for all products at once.")
    
    optimize_all_button = st.button("Optimize All Products", key="optimize_all")
    
    # Check if optimize button was clicked
    if optimize_button:
        with st.spinner("Optimizing inventory..."):
            try:
                # Get current inventory data
                inventory_data = get_current_inventory(data_loader, selected_product)
                
                # Get product info
                product_info = None
                if not products_data[products_data['product_id'] == selected_product].empty:
                    product_info = products_data[products_data['product_id'] == selected_product].iloc[0].to_dict()
                
                # Get cost and price from product data
                unit_cost = product_info['cost'] if product_info and 'cost' in product_info else 10
                unit_price = product_info['price'] if product_info and 'price' in product_info else 20
                
                # Optimize inventory
                optimization_result = optimize_inventory(inventory_optimizer, selected_product)
                
                if optimization_result:
                    # Extract results
                    eoq = optimization_result.get('eoq', 0)
                    reorder_point = optimization_result.get('reorder_point', 0)
                    service_level = optimization_result.get('service_level', 0.95)
                    safety_stock = optimization_result.get('safety_stock', 0)
                    annual_order_cost = optimization_result.get('annual_order_cost', 0)
                    annual_holding_cost = optimization_result.get('annual_holding_cost', 0)
                    stockout_prob = optimization_result.get('stockout_probability', 0)
                    
                    # Display results in an attractive layout with metrics and cards
                    st.subheader(f"Optimization Results for {product_info['name'] if product_info else selected_product}")
                    
                    # Create three columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Economic Order Quantity (EOQ)", f"{eoq:.0f} units")
                    
                    with col2:
                        st.metric("Reorder Point (ROP)", f"{reorder_point:.0f} units")
                        
                    with col3:
                        st.metric("Safety Stock", f"{safety_stock:.0f} units")
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
            
            with col1:
                        st.metric("Service Level", f"{service_level*100:.1f}%")
                    
                    with col2:
                        st.metric("Stockout Probability", f"{stockout_prob*100:.1f}%")
                        
                    with col3:
                        inventory_turnover = optimization_result.get('inventory_turnover', 0)
                        st.metric("Inventory Turnover", f"{inventory_turnover:.1f}x per year")
                    
                    # Cost analysis
                    st.subheader("Cost Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Annual Ordering Cost", f"${annual_order_cost:.2f}")
                    
                    with col2:
                        st.metric("Annual Holding Cost", f"${annual_holding_cost:.2f}")
                        
                    with col3:
                        total_cost = annual_order_cost + annual_holding_cost
                        st.metric("Total Annual Inventory Cost", f"${total_cost:.2f}")
                    
                    # Visualization of the EOQ model
                    st.subheader("Inventory Cost Visualization")
                    
                    # Create EOQ visualization
                    order_quantities = np.linspace(max(1, eoq * 0.2), eoq * 2, 100)
                    
                    # Calculate costs for different order quantities
                    annual_demand = optimization_result.get('annual_demand', 1000)
                    ordering_cost = optimization_result.get('ordering_cost', 50)
                    holding_cost_pct = optimization_result.get('holding_cost_pct', 0.25)
                    
                    ordering_costs = []
                    holding_costs = []
                    total_costs = []
                    
                    for q in order_quantities:
                        annual_ordering_cost = (annual_demand / q) * ordering_cost
                        annual_holding_cost = (q / 2) * unit_cost * holding_cost_pct
                        ordering_costs.append(annual_ordering_cost)
                        holding_costs.append(annual_holding_cost)
                        total_costs.append(annual_ordering_cost + annual_holding_cost)
                    
                    # Create plot with better styling
                fig = go.Figure()
                
                    fig.add_trace(go.Scatter(
                        x=order_quantities,
                        y=ordering_costs,
                        name='Ordering Cost',
                        line=dict(color='#4285F4', width=2.5),
                        hovertemplate='Order Quantity: %{x:.0f}<br>Ordering Cost: $%{y:.2f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=order_quantities,
                        y=holding_costs,
                        name='Holding Cost',
                        line=dict(color='#EA4335', width=2.5),
                        hovertemplate='Order Quantity: %{x:.0f}<br>Holding Cost: $%{y:.2f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=order_quantities,
                        y=total_costs,
                        name='Total Cost',
                        line=dict(color='#34A853', width=3.5),
                        hovertemplate='Order Quantity: %{x:.0f}<br>Total Cost: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Highlight the EOQ point
                    eoq_cost = total_costs[np.abs(order_quantities - eoq).argmin()]
                    
                    fig.add_trace(go.Scatter(
                        x=[eoq],
                        y=[eoq_cost],
                        mode='markers',
                        marker=dict(
                            color='#FBBC05',
                            size=12,
                            line=dict(
                                color='black',
                                width=2
                            )
                        ),
                        name='EOQ Optimal Point',
                        hovertemplate='EOQ: %{x:.0f} units<br>Minimum Cost: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add vertical line at EOQ
                    fig.add_shape(
                        type='line',
                        x0=eoq,
                        y0=0,
                        x1=eoq,
                        y1=max(total_costs)*1.1,
                        line=dict(color='#FBBC05', width=2, dash='dash')
                    )
                    
                    # Add annotation with better positioning
                    fig.add_annotation(
                        x=eoq,
                        y=max(total_costs)*0.9,
                        text=f"EOQ = {eoq:.0f} units",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='#FBBC05',
                        ax=40,
                        ay=-40,
                        font=dict(
                            size=14,
                            color='black',
                            family="Arial, sans-serif"
                        ),
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='#FBBC05',
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                    )
                    
                    # Add a note explaining EOQ
                    min_cost_point = f"${min(total_costs):.2f}"
                    fig.add_annotation(
                        x=0.5,
                        y=1.12,
                        xref="paper",
                        yref="paper",
                        text=f"The Economic Order Quantity (EOQ) of {eoq:.0f} units minimizes total inventory costs at {min_cost_point}",
                        showarrow=False,
                        font=dict(size=12),
                        align="center",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="lightgrey",
                        borderwidth=1,
                        borderpad=4
                    )
                    
                fig.update_layout(
                        title={
                            'text': "Economic Order Quantity (EOQ) Cost Analysis",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=20)
                        },
                        xaxis_title={
                            'text': "Order Quantity (units)",
                            'font': dict(size=14)
                        },
                        yaxis_title={
                            'text': "Annual Cost ($)",
                            'font': dict(size=14)
                        },
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=12)
                        ),
                        margin=dict(l=40, r=40, t=100, b=60),
                        plot_bgcolor='rgba(250,250,250,0.9)',
                        hovermode="x unified"
                    )
                    
                    # Add grid lines for better readability
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211,211,211,0.5)',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='lightgrey'
                    )
                    
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211,211,211,0.5)',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='lightgrey'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
                    # Add explanatory text about EOQ model
                    with st.expander("What is EOQ and how is it calculated?"):
                st.markdown(f"""
                        **The Economic Order Quantity (EOQ)** is the order quantity that minimizes the total costs of inventory management, balancing:
                        
                        * **Ordering costs**: The costs incurred when placing an order (fixed costs like processing, shipping, receiving).
                        * **Holding costs**: The costs of storing inventory (warehouse space, insurance, obsolescence, opportunity cost).
                        
                        The EOQ formula is:
                        
                        $EOQ = \\sqrt{{\\frac{{2 \\times D \\times S}}{{H}}}}$
                        
                        Where:
                        * D = Annual demand
                        * S = Fixed cost per order
                        * H = Annual holding cost per unit
                        
                        For this product:
                        * Annual Demand: {annual_demand:.0f} units
                        * Order Cost: ${ordering_cost:.2f} per order
                        * Holding Cost: {holding_cost_pct*100:.1f}% of unit cost (${unit_cost*holding_cost_pct:.2f} per unit per year)
                        * Calculated EOQ: {eoq:.0f} units
                        """)
                    
                    # Inventory projection with enhanced visuals
                    st.subheader("Inventory Projection")
                    
                    # Get demand forecast
                    feature_engineering = FeatureEngineering(data_loader)
                    model_selector = ModelSelector(data_loader, feature_engineering)
                    forecast_data = advanced_demand_forecast(
                        model_selector, 
                        selected_product, 
                        periods=90,  # 3 months forecast
                        confidence=0.95
                    )
                    
                    if forecast_data is not None:
                        # Get current inventory level
                        current_inventory = get_current_inventory(data_loader, selected_product)
                        
                        # Create inventory projection
                        projection = create_inventory_projection(
                            current_inventory,
                            forecast_data,
                            eoq,
                            reorder_point,
                            lead_time=optimization_result.get('lead_time', 7)
                        )
                        
                        if projection is not None:
                            # Plot inventory projection with enhanced visuals
            fig = go.Figure()
            
                            # Add inventory level line
            fig.add_trace(go.Scatter(
                                x=projection['date'],
                                y=projection['inventory_level'],
                name='Projected Inventory',
                                line=dict(color='#4285F4', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(66, 133, 244, 0.1)',
                                hovertemplate='Date: %{x|%Y-%m-%d}<br>Inventory: %{y:.0f} units<extra></extra>'
            ))
            
            # Add reorder point line
            fig.add_trace(go.Scatter(
                                x=projection['date'],
                                y=[reorder_point] * len(projection),
                name='Reorder Point',
                                line=dict(color='#EA4335', width=2, dash='dash'),
                                hovertemplate='Reorder Point: %{y:.0f} units<extra></extra>'
            ))
            
            # Add safety stock line
            fig.add_trace(go.Scatter(
                                x=projection['date'],
                                y=[safety_stock] * len(projection),
                name='Safety Stock',
                                line=dict(color='#FBBC05', width=2, dash='dot'),
                                hovertemplate='Safety Stock: %{y:.0f} units<extra></extra>'
                            ))
                            
                            # Highlight critical inventory zones
                            danger_zone = safety_stock * 0.8
                            fig.add_shape(
                                type="rect",
                                x0=projection['date'].min(),
                                y0=0,
                                x1=projection['date'].max(),
                                y1=danger_zone,
                                fillcolor="rgba(234, 67, 53, 0.1)",
                                line=dict(width=0),
                                layer="below"
                            )
                            
                            # Add order points with better visuals
                            order_dates = projection[projection['place_order']]['date']
                            order_levels = projection[projection['place_order']]['inventory_level']
                            
                fig.add_trace(go.Scatter(
                                x=order_dates,
                                y=order_levels,
                    mode='markers',
                                name='Place Order',
                    marker=dict(
                                    color='#EA4335',
                        size=12,
                                    symbol='triangle-up',
                                    line=dict(
                                        color='white',
                                        width=1
                                    )
                                ),
                                hovertemplate='Date: %{x|%Y-%m-%d}<br>Place order of %{text} units<br>Current Level: %{y:.0f} units<extra></extra>',
                                text=[f"{eoq:.0f}"] * len(order_dates)
                            ))
                            
                            # Add receive points with better visuals
                            receive_dates = projection[projection['receive_order']]['date']
                            receive_levels = projection[projection['receive_order']]['inventory_level']
                            
                            if not receive_dates.empty:
                                fig.add_trace(go.Scatter(
                                    x=receive_dates,
                                    y=receive_levels,
                                    mode='markers',
                                    name='Receive Order',
                                    marker=dict(
                                        color='#34A853',
                                        size=12,
                                        symbol='circle',
                                        line=dict(
                                            color='white',
                                            width=1
                                        )
                                    ),
                                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Receive order of %{text} units<br>New Level: %{y:.0f} units<extra></extra>',
                                    text=[f"{eoq:.0f}"] * len(receive_dates)
                                ))
                            
                            # Add annotations for first order and receipt
                            if not order_dates.empty:
                                first_order_date = order_dates.iloc[0]
                                first_order_level = order_levels.iloc[0]
                                
                                fig.add_annotation(
                                    x=first_order_date,
                                    y=first_order_level,
                                    text="First Order",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="#EA4335",
                                    arrowsize=1,
                                    arrowwidth=2,
                                    ax=-40,
                                    ay=-40,
                                    font=dict(color="#EA4335", size=12),
                                    bgcolor="white",
                                    bordercolor="#EA4335",
                                    borderpad=3,
                                    borderwidth=1,
                                )
                            
                            if not receive_dates.empty:
                                first_receive_date = receive_dates.iloc[0]
                                first_receive_level = receive_levels.iloc[0]
                                
                                fig.add_annotation(
                                    x=first_receive_date,
                                    y=first_receive_level,
                                    text="First Delivery",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="#34A853",
                                    arrowsize=1,
                                    arrowwidth=2,
                                    ax=40,
                                    ay=-40,
                                    font=dict(color="#34A853", size=12),
                                    bgcolor="white",
                                    bordercolor="#34A853",
                                    borderpad=3,
                                    borderwidth=1,
                                )
                            
                            # Add summary statistics
                            total_orders = len(order_dates)
                            avg_inventory = projection['inventory_level'].mean()
                            
                            fig.add_annotation(
                                x=0.5,
                                y=1.13,
                                xref="paper",
                                yref="paper",
                                text=f"Orders needed: {total_orders} | Average inventory level: {avg_inventory:.0f} units | Lead time: {optimization_result.get('lead_time', 7)} days",
                                showarrow=False,
                                font=dict(size=12),
                                align="center",
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="lightgrey",
                                borderwidth=1,
                                borderpad=4
                            )
                            
            fig.update_layout(
                                title={
                                    'text': "90-Day Inventory Projection with Order Schedule",
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=20)
                                },
                                xaxis_title={
                                    'text': "Date",
                                    'font': dict(size=14)
                                },
                                yaxis_title={
                                    'text': "Inventory Level (units)",
                                    'font': dict(size=14)
                                },
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    font=dict(size=12)
                ),
                                margin=dict(l=40, r=40, t=120, b=60),
                                plot_bgcolor='rgba(250,250,250,0.9)',
                hovermode="x unified"
            )
                            
                            # Add grid lines for better readability
                            fig.update_xaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(211,211,211,0.5)',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='lightgrey',
                                tickformat='%b %d'
                            )
                            
                            fig.update_yaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(211,211,211,0.5)',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='lightgrey'
                            )
            
            st.plotly_chart(fig, use_container_width=True)
            
                            # Add explanatory text about inventory projections
                            with st.expander("Understanding the Inventory Projection"):
                                st.markdown(f"""
                                This chart shows the projected inventory levels over the next 90 days, based on:
                                
                                * **Current inventory**: Starting level of {current_inventory:.0f} units
                                * **Forecasted demand**: Daily demand projections from the demand forecasting model
                                * **Reorder point**: When inventory reaches {reorder_point:.0f} units, a new order is placed
                                * **Order quantity**: Each order is for {eoq:.0f} units (the EOQ)
                                * **Lead time**: {optimization_result.get('lead_time', 7):.0f} days from order placement to receipt
                                * **Safety stock**: {safety_stock:.0f} units to protect against demand variability
                                
                                The red triangles show when orders are placed, and green circles show when inventory is received.
                                """)
                            
                            # Display order schedule with enhanced styling
                            st.subheader("Recommended Order Schedule")
                            
                            # Filter only rows where orders are placed
                            order_schedule = projection[projection['place_order']].copy()
                            
                            if not order_schedule.empty:
                                # Calculate expected receipt date (order date + lead time)
                                lead_time = optimization_result.get('lead_time', 7)
                                
                                order_schedule['receipt_date'] = order_schedule['date'].apply(
                                    lambda x: x + pd.Timedelta(days=lead_time)
                                )
                                
                                # Format dates for display
                                order_schedule['order_date'] = order_schedule['date'].dt.strftime('%Y-%m-%d')
                                order_schedule['receipt_date'] = order_schedule['receipt_date'].dt.strftime('%Y-%m-%d')
                                
                                # Add expected inventory at order time and order amount (EOQ)
                                order_schedule['inventory_at_order'] = order_schedule['inventory_level'].round(0).astype(int)
                                order_schedule['order_quantity'] = eoq
                                order_schedule['expected_cost'] = eoq * unit_cost
                                
                                # Display as table
                                display_cols = [
                                    'order_date', 
                                    'inventory_at_order', 
                                    'order_quantity', 
                                    'expected_cost',
                                    'receipt_date'
                                ]
                                
                                display_names = {
                                    'order_date': 'Order Date',
                                    'inventory_at_order': 'Inventory Level',
                                    'order_quantity': 'Order Quantity',
                                    'expected_cost': 'Order Cost ($)',
                                    'receipt_date': 'Expected Receipt'
                                }
                                
                                order_display = order_schedule[display_cols].rename(columns=display_names)
                                
                                # Add order urgency indicator
                                def get_urgency(days_until):
                                    if days_until <= 7:
                                        return "🔴 Urgent"
                                    elif days_until <= 14:
                                        return "🟠 Soon"
    else:
                                        return "🟢 Planned"
                                
                                # Calculate days from today
                                today = pd.Timestamp.now().date()
                                order_display['Order Date'] = pd.to_datetime(order_display['Order Date'])
                                order_display['Days Until Order'] = (order_display['Order Date'].dt.date - today).dt.days
                                order_display['Order Urgency'] = order_display['Days Until Order'].apply(get_urgency)
                                
                                # Sort by order date
                                order_display = order_display.sort_values('Order Date')
                                
                                # Reformat date after calculations
                                order_display['Order Date'] = order_display['Order Date'].dt.strftime('%Y-%m-%d')
                                
                                # Display the enhanced table
                                st.dataframe(order_display, use_container_width=True)
                                
                                # Calculate total orders and cost
                                total_orders = len(order_schedule)
                                total_order_cost = total_orders * eoq * unit_cost
                                
                                # Enhanced info display with HTML
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #4285F4;">
                                    <h4 style="margin-top: 0; color: #4285F4;">Order Summary</h4>
                                    <p><strong>Total orders needed:</strong> {total_orders} orders over 90 days</p>
                                    <p><strong>Total cost:</strong> ${total_order_cost:.2f}</p>
                                    <p><strong>Average order frequency:</strong> {90/max(1, total_orders):.1f} days between orders</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Download order schedule with better styling
                                col1, col2 = st.columns([3, 1])
                                with col1:
        st.markdown("""
                                    <div style="padding: 5px; color: #666;">
                                    Download the complete order schedule to integrate with your purchasing system.
        </div>
        """, unsafe_allow_html=True)
        
                                with col2:
                                    csv = order_display.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "📥 Download Schedule",
                                        csv,
                                        f"order_schedule_{selected_product}.csv",
                                        "text/csv",
                                        key='download-orders',
                                        use_container_width=True
                                    )
                            else:
                                # Enhanced message when no orders needed
                                st.markdown(f"""
                                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #34A853;">
                                    <h4 style="margin-top: 0; color: #34A853;">No Orders Required</h4>
                                    <p>Current inventory levels are sufficient to meet forecasted demand for the next 90 days.</p>
                                    <p>Current inventory: <strong>{current_inventory:.0f} units</strong></p>
                                    <p>Projected minimum in next 90 days: <strong>{projection['inventory_level'].min():.0f} units</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.error("Optimization failed. Please try another product.")
            except Exception as e:
                st.error(f"Error optimizing inventory: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Check if optimize all button was clicked
    if optimize_all_button:
            with st.spinner("Optimizing inventory for all products..."):
            try:
                results = optimize_inventory_for_all_products(
                    inventory_optimizer,
                    [pid for pid, _ in product_options]
                )
                
                if results and len(results) > 0:
                    st.subheader("Batch Optimization Results")
                    
                    # Create a DataFrame for display
                    results_df = pd.DataFrame(results)
                    
                    # Add product names
                    product_names = {row['product_id']: row['name'] 
                                   for _, row in products_data.iterrows()}
                    
                    results_df['Product Name'] = results_df['product_id'].map(product_names)
                    
                    # Format decimal columns
                    decimal_cols = ['service_level', 'stockout_probability']
                    for col in decimal_cols:
                        if col in results_df.columns:
                            results_df[col] = results_df[col] * 100
                    
                    # Rename columns for display
                    display_names = {
                        'product_id': 'Product ID',
                        'eoq': 'EOQ',
                        'reorder_point': 'Reorder Point',
                        'safety_stock': 'Safety Stock',
                        'service_level': 'Service Level (%)',
                        'stockout_probability': 'Stockout Probability (%)',
                        'annual_order_cost': 'Annual Ordering Cost ($)',
                        'annual_holding_cost': 'Annual Holding Cost ($)',
                        'total_cost': 'Total Annual Cost ($)'
                    }
                    
                    results_df = results_df.rename(columns=display_names)
                    
                    # Reorder columns
                    col_order = [
                        'Product ID', 'Product Name', 'EOQ', 'Reorder Point',
                        'Safety Stock', 'Service Level (%)', 'Stockout Probability (%)',
                        'Annual Ordering Cost ($)', 'Annual Holding Cost ($)', 'Total Annual Cost ($)'
                    ]
                    
                    display_df = results_df[[col for col in col_order if col in results_df.columns]]
                    
                    # Round numeric columns
                    numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
                    display_df[numeric_cols] = display_df[numeric_cols].round(2)
                    
                    # Display as interactive table
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download results
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Optimization Results",
                        csv,
                        "inventory_optimization_results.csv",
                        "text/csv",
                        key='download-optimization'
                    )
                else:
                    st.error("Batch optimization failed to produce results.")
            except Exception as e:
                st.error(f"Error in batch optimization: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

def optimize_inventory_for_all_products(inventory_optimizer, product_ids):
    """
    Optimize inventory for all products
    
    Parameters:
    -----------
    inventory_optimizer : InventoryOptimizer
        InventoryOptimizer instance
    product_ids : list
        List of product IDs to optimize
        
    Returns:
    --------
    list: List of optimization results for all products
    """
    results = []
    
    for product_id in product_ids:
        try:
            # Optimize inventory for product
            result = optimize_inventory(inventory_optimizer, product_id)
            
            if result:
                # Add product ID to result
                result['product_id'] = product_id
                
                # Calculate total cost
                total_cost = result.get('annual_order_cost', 0) + result.get('annual_holding_cost', 0)
                result['total_cost'] = total_cost
                
                # Add to results
                results.append(result)
                
        except Exception as e:
            print(f"Error optimizing inventory for product {product_id}: {str(e)}")
            continue
    
    return results

def optimize_inventory(inventory_optimizer, product_id):
    """
    Optimize inventory for a single product
    
    Parameters:
    -----------
    inventory_optimizer : InventoryOptimizer
        Instance of InventoryOptimizer class
    product_id : str
        Product ID
        
    Returns:
    --------
    dict: Dictionary containing optimization results
    """
    try:
        # Call the optimizer's optimize method
        result = inventory_optimizer.optimize_inventory(product_id)
        
        # If result is empty or None, create a basic structure
        if not result:
            # Get product data
            product_data = inventory_optimizer.data_loader.datasets['products']
            product_info = product_data[product_data['product_id'] == product_id]
            
            if product_info.empty:
                return None
            
            # Get current inventory level
            current_inventory = get_current_inventory(inventory_optimizer.data_loader, product_id)
            
            # Get historical demand
            demand_data = inventory_optimizer.data_loader.get_demand_by_product(product_id)
            avg_daily_demand = demand_data['demand'].mean() if not demand_data.empty else 0
            
            # Create basic result
            result = {
                'product_id': product_id,
                'current_inventory': current_inventory,
                'daily_demand': avg_daily_demand,
                'days_to_stockout': current_inventory / max(0.1, avg_daily_demand),
                'eoq': 0,
                'reorder_point': 0,
                'safety_stock': 0,
                'should_order': False,
                'quantity_to_order': 0,
                'order_cost': 0,
                'pending_orders': []
            }
        
        return result
    except Exception as e:
        print(f"Error in optimize_inventory: {str(e)}")
        return None

def display_model_training():
    """Display model training interface and functionality"""
    st.markdown("## Demand Forecasting Model Training")
    
    # Get available products and categories
    available_data = load_model_data()
    
    # Load product data
    products_df = data_loader.get_products()
    
    # Define tabs for different training options
    tab1, tab2, tab3 = st.tabs(["Train Single Product Model", "Train Category Models", "Train All Products"])
    
    with tab1:
        st.subheader("Train model for a single product")
        
        # Select product
        product_options = [(f"{p['product_name']} (ID: {p['product_id']})", p['product_id']) 
                           for _, p in products_df.iterrows()]
        
        selected_product_tuple = st.selectbox(
            "Select a product",
            options=product_options,
            format_func=lambda x: x[0]
        )
        
        if selected_product_tuple:
            selected_product_id = selected_product_tuple[1]
            
            # Select model type
            model_type = st.selectbox(
                "Select model type",
                options=["Auto (Best)", "Linear Regression", "Random Forest", "Gradient Boosting", "Prophet"],
                index=0
            )
            
            # Training parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
                
            with col2:
                noise_level = st.slider("Noise level (to prevent overfitting)", 
                                        min_value=0, max_value=10, value=5, step=1) / 100
            
            # Add a button to train the model
            if st.button("Train Model", key="train_single"):
                with st.spinner("Training model..."):
                    # Train model
                    model_selector = ModelSelector(data_loader)
                    
                    try:
                        # Convert UI selection to model object
                        if model_type == "Linear Regression":
                            model_obj = LinearRegression()
                        elif model_type == "Random Forest":
                            model_obj = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Gradient Boosting":
                            model_obj = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Prophet":
                            model_obj = "Prophet"  # Special case for Prophet
                        else:
                            model_obj = None  # Auto select
                            
                        # Train the model with selected parameters
                        result = model_selector.train_demand_forecasting_models(
                            product_ids=[selected_product_id],
                            models=[model_obj] if model_obj else None,
                            test_size=test_size,
                            noise_level=noise_level
                        )
                        
                        if result and len(result) > 0:
                            st.success(f"Model trained successfully for product {selected_product_id}")
                            
                            # Display evaluation metrics
                            metrics = result[selected_product_id]['metrics']
                            best_model = result[selected_product_id]['best_model']
                            
                            st.markdown("### Model Evaluation Metrics")
                            
                            # Create metrics display
                            metrics_df = pd.DataFrame(metrics).transpose()
                            st.dataframe(metrics_df)
                            
                            # Show best model
                            st.markdown(f"**Best Model:** {best_model}")
                            
                            # Plot actual vs predicted for test set
                            if 'test_predictions' in result[selected_product_id]:
                                test_preds = result[selected_product_id]['test_predictions']
                                if isinstance(test_preds, dict) and len(test_preds) > 0:
                                    # Get the first (and only) key if it's a dict
                                    first_key = list(test_preds.keys())[0]
                                    test_preds_df = test_preds[first_key]
                                    
                                    if not test_preds_df.empty and 'actual' in test_preds_df.columns and 'predicted' in test_preds_df.columns:
                                        st.markdown("### Actual vs Predicted (Test Set)")
                                        
                                        fig = px.scatter(
                                            test_preds_df, 
                                            x='actual', 
                                            y='predicted',
                                            title='Actual vs Predicted',
                                            labels={'actual': 'Actual Values', 'predicted': 'Predicted Values'}
                                        )
                                        
                                        # Add a 45-degree line for reference
                                        max_val = max(test_preds_df['actual'].max(), test_preds_df['predicted'].max())
                                        min_val = min(test_preds_df['actual'].min(), test_preds_df['predicted'].min())
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=[min_val, max_val],
                                                y=[min_val, max_val],
                                                mode='lines',
                                                name='Perfect Prediction',
                                                line=dict(color='red', dash='dash')
                                            )
                                        )
                                        
                                        st.plotly_chart(fig)
                                        
                                        # Show data leakage warning if accuracy is suspiciously high
                                        if 'r2' in metrics[best_model] and metrics[best_model]['r2'] > 0.98:
                                            st.warning(
                                                "⚠️ The model's accuracy is suspiciously high, which may indicate data leakage. "
                                                "Consider reviewing your features or increasing the noise level."
                                            )
                                    else:
                                        st.info("No test predictions available to display")
                                else:
                                    st.info("No test predictions available to display")
                        else:
                            st.error("Failed to train model. Check logs for details.")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("Train models by product category")
        
        # Get unique categories
        categories = products_df['category'].unique().tolist()
        
        # Select category
        selected_category = st.selectbox(
            "Select a category",
            options=categories
        )
        
        if selected_category:
            # Get products in selected category
            category_products = products_df[products_df['category'] == selected_category]
            st.info(f"Training will include {len(category_products)} products in the {selected_category} category")
            
            # Select model type
            model_type = st.selectbox(
                "Select model type",
                options=["Auto (Best)", "Linear Regression", "Random Forest", "Gradient Boosting", "Prophet"],
                index=0,
                key="cat_model"
            )
            
            # Training parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test set size (%)", 
                                     min_value=10, max_value=50, value=20, step=5,
                                     key="cat_test_size") / 100
                
            with col2:
                noise_level = st.slider("Noise level (to prevent overfitting)", 
                                       min_value=0, max_value=10, value=5, step=1,
                                       key="cat_noise") / 100
            
            # Add a button to train the models
            if st.button("Train Category Models", key="train_category"):
                with st.spinner(f"Training models for {len(category_products)} products in {selected_category} category..."):
                    # Get product IDs for the selected category
                    product_ids = category_products['product_id'].tolist()
                    
                    # Train models
                    model_selector = ModelSelector(data_loader)
                    
                    try:
                        # Convert UI selection to model object
                        if model_type == "Linear Regression":
                            model_obj = LinearRegression()
                        elif model_type == "Random Forest":
                            model_obj = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Gradient Boosting":
                            model_obj = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Prophet":
                            model_obj = "Prophet"  # Special case for Prophet
                        else:
                            model_obj = None  # Auto select
                        
                        # Train models for all products in the category
                        result = model_selector.train_demand_forecasting_models(
                            product_ids=product_ids,
                            models=[model_obj] if model_obj else None,
                            test_size=test_size,
                            noise_level=noise_level
                        )
                        
                        if result and len(result) > 0:
                            st.success(f"Models trained successfully for {len(result)} products in {selected_category} category")
                            
                            # Display summary of results
                            summary_data = []
                            for product_id, res in result.items():
                                product_name = next((p['product_name'] for _, p in products_df.iterrows() 
                                                  if p['product_id'] == product_id), product_id)
                                                  
                                best_model = res.get('best_model', 'N/A')
                                metrics = res.get('metrics', {})
                                best_metrics = metrics.get(best_model, {})
                                
                                rmse = best_metrics.get('rmse', float('nan'))
                                r2 = best_metrics.get('r2', float('nan'))
                                
                                summary_data.append({
                                    'Product': product_name,
                                    'Best Model': best_model,
                                    'RMSE': rmse,
                                    'R²': r2
                                })
                            
                            if summary_data:
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df)
                                
                                # Show histogram of R² values
                                r2_values = [d['R²'] for d in summary_data if not math.isnan(d['R²'])]
                                if r2_values:
                                    fig = px.histogram(
                                        x=r2_values,
                                        nbins=10,
                                        title=f'Distribution of R² Scores for {selected_category} Products',
                                        labels={'x': 'R² Score', 'y': 'Count'}
                                    )
                                    st.plotly_chart(fig)
                                    
                                    # Show data leakage warning if many models have suspiciously high accuracy
                                    high_r2_count = sum(1 for r2 in r2_values if r2 > 0.98)
                                    if high_r2_count > len(r2_values) * 0.3:  # If more than 30% have very high R²
                                        st.warning(
                                            f"⚠️ {high_r2_count} models have suspiciously high accuracy (R² > 0.98), "
                                            "which may indicate data leakage. Consider reviewing your features "
                                            "or increasing the noise level."
                                        )
                        else:
                            st.error("Failed to train models. Check logs for details.")
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
                        st.code(traceback.format_exc())
    
    with tab3:
        st.subheader("Train models for all products")
        st.info(f"Training will include all {len(products_df)} products")
        
        # Select model type
        model_type = st.selectbox(
            "Select model type",
            options=["Auto (Best)", "Linear Regression", "Random Forest", "Gradient Boosting", "Prophet"],
            index=0,
            key="all_model"
        )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size (%)", 
                                 min_value=10, max_value=50, value=20, step=5,
                                 key="all_test_size") / 100
            
        with col2:
            noise_level = st.slider("Noise level (to prevent overfitting)", 
                                   min_value=0, max_value=10, value=5, step=1,
                                   key="all_noise") / 100
        
        # Add a button to train the models
        if st.button("Train All Models", key="train_all"):
            with st.spinner(f"Training models for all {len(products_df)} products..."):
                # Get all product IDs
                all_product_ids = products_df['product_id'].tolist()
                
                # If there are many products, let user know this will take time
                if len(all_product_ids) > 10:
                    st.warning(f"Training models for {len(all_product_ids)} products may take a while. Please be patient.")
                
                # Train models
                model_selector = ModelSelector(data_loader)
                
                try:
                    # Convert UI selection to model object
                    if model_type == "Linear Regression":
                        model_obj = LinearRegression()
                    elif model_type == "Random Forest":
                        model_obj = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_type == "Gradient Boosting":
                        model_obj = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    elif model_type == "Prophet":
                        model_obj = "Prophet"  # Special case for Prophet
                    else:
                        model_obj = None  # Auto select
                    
                    # Train models for all products
                    result = model_selector.train_demand_forecasting_models(
                        product_ids=all_product_ids,
                        models=[model_obj] if model_obj else None,
                        test_size=test_size,
                        noise_level=noise_level
                    )
                    
                    if result and len(result) > 0:
                        st.success(f"Models trained successfully for {len(result)} products")
                        
                        # Display summary of results by category
                        categories_summary = {}
                        for product_id, res in result.items():
                            try:
                                product_info = products_df[products_df['product_id'] == product_id].iloc[0]
                                category = product_info['category']
                                
                                if category not in categories_summary:
                                    categories_summary[category] = {
                                        'count': 0,
                                        'r2_sum': 0,
                                        'rmse_sum': 0,
                                        'successful': 0,
                                        'models': {}
                                    }
                                
                                best_model = res.get('best_model', 'N/A')
                                if best_model != 'N/A':
                                    categories_summary[category]['count'] += 1
                                    
                                    if best_model not in categories_summary[category]['models']:
                                        categories_summary[category]['models'][best_model] = 0
                                    categories_summary[category]['models'][best_model] += 1
                                    
                                    metrics = res.get('metrics', {})
                                    best_metrics = metrics.get(best_model, {})
                                    
                                    rmse = best_metrics.get('rmse', float('nan'))
                                    r2 = best_metrics.get('r2', float('nan'))
                                    
                                    if not math.isnan(r2):
                                        categories_summary[category]['r2_sum'] += r2
                                        categories_summary[category]['successful'] += 1
                                    
                                    if not math.isnan(rmse):
                                        categories_summary[category]['rmse_sum'] += rmse
                            except:
                                # Skip this product if there's an issue
                                continue
                        
                        # Create summary dataframe
                        summary_data = []
                        for category, data in categories_summary.items():
                            avg_r2 = data['r2_sum'] / data['successful'] if data['successful'] > 0 else float('nan')
                            avg_rmse = data['rmse_sum'] / data['successful'] if data['successful'] > 0 else float('nan')
                            
                            # Get the most common model
                            most_common_model = max(data['models'].items(), key=lambda x: x[1])[0] if data['models'] else 'N/A'
                            
                            summary_data.append({
                                'Category': category,
                                'Products': data['count'],
                                'Successful': data['successful'],
                                'Avg R²': avg_r2,
                                'Avg RMSE': avg_rmse,
                                'Most Common Model': most_common_model
                            })
                        
                        if summary_data:
                            st.markdown("### Training Results by Category")
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df)
                            
                            # Plot average R² by category
                            valid_data = [d for d in summary_data if not math.isnan(d['Avg R²'])]
                            if valid_data:
                                fig = px.bar(
                                    valid_data,
                                    x='Category',
                                    y='Avg R²',
                                    title='Average R² Score by Product Category',
                                    color='Most Common Model'
                                )
                                st.plotly_chart(fig)
                                
                                # Show data leakage warning if many categories have suspiciously high accuracy
                                high_r2_count = sum(1 for d in valid_data if d['Avg R²'] > 0.95)
                                if high_r2_count > len(valid_data) * 0.3:  # If more than 30% have very high R²
                                    st.warning(
                                        f"⚠️ {high_r2_count} categories have suspiciously high average accuracy (R² > 0.95), "
                                        "which may indicate data leakage. Consider reviewing your features "
                                        "or increasing the noise level."
                                    )
                    else:
                        st.error("Failed to train models. Check logs for details.")
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.code(traceback.format_exc())

def advanced_demand_forecast(model_selector, product_id, periods=30, freq='D', confidence=0.95):
    """
    Generate an advanced demand forecast using the ensemble model
    
    Parameters:
    -----------
    model_selector : ModelSelector
        ModelSelector instance
    product_id : str
        Product ID
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of forecast ('D' for daily, 'W' for weekly, etc.)
    confidence : float
        Confidence level for prediction intervals (0.0-1.0)
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with forecast and confidence intervals
    """
    try:
        with st.spinner("Generating advanced ensemble forecast..."):
            # Create cache key based on parameters
            cache_key = f"{product_id}_{periods}_{freq}_{confidence}"
            
            # Check if we already have this forecast in standard cache (not session state)
            # This is for short-term caching while on the same page
            if cache_key in st.session_state and st.session_state[cache_key] is not None:
                return st.session_state[cache_key]
            
            # Get historical data for scaling reference
            historical_data = model_selector.data_loader.get_demand_by_product(product_id)
            
            # Check if data is empty
            if historical_data.empty:
                st.error(f"No historical data available for product {product_id}")
                return None
                
            hist_mean = historical_data['demand'].mean()
            
            # Create ensemble model with multiple forecasting methods
            ensemble_model, metrics = model_selector.create_ensemble_model(
                product_id,
                model_types=['linear', 'rf', 'gbm', 'arima'],  # Removed prophet as it's causing issues
                test_size=0.2
            )
            
            if ensemble_model is None:
                st.error("Failed to create ensemble model")
                return None
                
            # Generate future features for prediction
            future_features = model_selector.generate_future_features(
                product_id,
                periods=periods,
                freq=freq
            )
            
            if future_features is None or future_features.empty:
                st.error("Failed to generate future features")
                return None
                
            # Set up dates for forecasting - fix Timestamp arithmetic
            last_date = historical_data['date'].max()
            
            # Ensure last_date is a pandas Timestamp object
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.Timestamp(last_date)
                
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            
            # Get forecast and prediction intervals
            point_forecast = ensemble_model.predict(future_features)
            lower_bound, upper_bound = ensemble_model.predict_interval(future_features, confidence=confidence)
            
            # Create forecast DataFrame
            forecast = pd.DataFrame({
                'date': future_dates,
                'product_id': product_id,
                'forecast': point_forecast,
                'lower_ci': lower_bound,
                'upper_ci': upper_bound
            })
            
            # Perform sanity checks on forecast values
            if hist_mean is not None:
                forecast_mean = forecast['forecast'].mean()
                # If forecast mean is suspiciously low compared to historical data
                if forecast_mean < 0.7 * hist_mean:
                    st.warning(f"Forecast mean ({forecast_mean:.2f}) is much lower than historical mean ({hist_mean:.2f}). Adjusting forecast values.")
                    adjustment_factor = hist_mean / forecast_mean if forecast_mean > 0 else 1.2
                    # Cap adjustment to reasonable values
                    adjustment_factor = min(adjustment_factor, 1.5)
                    
                    # Apply adjustment
                    forecast['forecast'] = forecast['forecast'] * adjustment_factor
                    forecast['lower_ci'] = forecast['lower_ci'] * adjustment_factor  
                    forecast['upper_ci'] = forecast['upper_ci'] * adjustment_factor
                elif forecast_mean > 1.5 * hist_mean:
                    st.warning(f"Forecast mean ({forecast_mean:.2f}) is much higher than historical mean ({hist_mean:.2f}). Adjusting forecast values.")
                    adjustment_factor = hist_mean / forecast_mean if forecast_mean > 0 else 0.8
                    # Ensure adjustment is reasonable
                    adjustment_factor = max(adjustment_factor, 0.7)
                    
                    # Apply adjustment
                    forecast['forecast'] = forecast['forecast'] * adjustment_factor
                    forecast['lower_ci'] = forecast['lower_ci'] * adjustment_factor
                    forecast['upper_ci'] = forecast['upper_ci'] * adjustment_factor
            
            # Apply seasonality patterns from historical data
            if not historical_data.empty and len(historical_data) >= 30:
                try:
                    # Extract day of week patterns
                    historical_data['dayofweek'] = pd.to_datetime(historical_data['date']).dt.dayofweek
                    dow_factors = historical_data.groupby('dayofweek')['demand'].mean() / historical_data['demand'].mean()
                    
                    # Apply day of week seasonality to forecast
                    forecast['dayofweek'] = pd.to_datetime(forecast['date']).dt.dayofweek
                    for day, factor in dow_factors.items():
                        # Only apply if factor is reasonable (not extreme)
                        if 0.5 <= factor <= 2.0:
                            day_mask = forecast['dayofweek'] == day
                            forecast.loc[day_mask, 'forecast'] = forecast.loc[day_mask, 'forecast'] * factor
                            forecast.loc[day_mask, 'lower_ci'] = forecast.loc[day_mask, 'lower_ci'] * factor
                            forecast.loc[day_mask, 'upper_ci'] = forecast.loc[day_mask, 'upper_ci'] * factor
                    
                    # Remove the helper column
                    forecast = forecast.drop(columns=['dayofweek'])
                except Exception as e:
                    print(f"Error applying seasonality: {str(e)}")
            
            # Ensure all values are non-negative
            forecast['forecast'] = forecast['forecast'].clip(lower=0)
            forecast['lower_ci'] = forecast['lower_ci'].clip(lower=0)
            forecast['upper_ci'] = forecast['upper_ci'].clip(lower=0)
            
            # Ensure forecast has some reasonable variation
            if forecast['forecast'].std() < 0.1 * forecast['forecast'].mean():
                # Add slight variation based on historical patterns
                if not historical_data.empty:
                    cv = historical_data['demand'].std() / historical_data['demand'].mean() if historical_data['demand'].mean() > 0 else 0.2
                    # Add noise with similar coefficient of variation
                    noise = np.random.normal(0, forecast['forecast'].mean() * cv * 0.5, size=len(forecast))
                    forecast['forecast'] = forecast['forecast'] + noise
                    # Ensure non-negative values
                    forecast['forecast'] = forecast['forecast'].clip(lower=0)
            
            # Add metadata about the forecast
            forecast.attrs['generated_at'] = str(datetime.now())
            forecast.attrs['product_id'] = product_id
            forecast.attrs['periods'] = periods
            forecast.attrs['confidence'] = confidence
            
            # Store result in standard cache (not session state)
            # This is for short-term caching while on the same page
            st.session_state[cache_key] = forecast
            
            return forecast
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        print(f"Detailed error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_demand_forecast_with_confidence(historical_data, forecast_data, product_info=None, display_days=None):
    """
    Create an interactive forecast plot with confidence intervals
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Historical demand data
    forecast_data : pandas.DataFrame
        Forecast data with confidence intervals
    product_info : dict, optional
        Product information for display
    display_days : int, optional
        Number of recent days to display
    
    Returns:
    --------
    plotly.graph_objects.Figure: Interactive forecast plot
    """
    # Basic error checking
    if historical_data is None or forecast_data is None:
        st.error("No data available for plotting")
        return None
    
    try:
        # Create a copy of the data to avoid modifying originals
        historical = historical_data.copy()
        forecast = forecast_data.copy()
        
        # Ensure dates are datetime objects
        historical['date'] = pd.to_datetime(historical['date'])
        forecast['date'] = pd.to_datetime(forecast['date'])
        
        # Sort by date
        historical = historical.sort_values('date')
        forecast = forecast.sort_values('date')
        
        # Calculate forecast start date for plotting
        forecast_start = forecast['date'].min()
        
        # Filter historical data to only show recent history if display_days is specified
        if display_days is not None and not historical.empty:
            # Fix the timestamp arithmetic error by using Timedelta properly
            cutoff_date = forecast_start - pd.Timedelta(days=int(display_days))
            historical = historical[historical['date'] >= cutoff_date]
        
        # Create figure
        fig = go.Figure()
        
        # Add historical demand
        if not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['demand'],
                mode='lines',
                name='Historical Demand',
                line=dict(color='#1F77B4', width=2)
            ))
        
        # Format the date where forecast starts
        forecast_start_str = forecast_start.strftime('%Y-%m-%d')
            
        # Add forecast
        if not forecast.empty:
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Add confidence intervals as a filled area
            if 'lower_ci' in forecast.columns and 'upper_ci' in forecast.columns:
                # Add upper bound
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['upper_ci'],
                    mode='lines',
                    name='95% Confidence Interval',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                # Add lower bound with fill to upper bound
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['lower_ci'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    name='95% Confidence Interval'
                ))
        
        # Add a vertical line to mark forecast start - use add_shape instead of add_vline
        fig.add_shape(
            type="line",
            x0=forecast_start,
            y0=0,
            x1=forecast_start,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        # Add annotation for forecast start
        fig.add_annotation(
            x=forecast_start,
            y=1.05,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="gray")
        )
        
        # Update layout
        product_name = product_info['name'] if product_info and 'name' in product_info else ""
        product_id = product_info['product_id'] if product_info and 'product_id' in product_info else ""
        
        title = f"Demand Forecast for {product_name} - Advanced Ensemble"
        if product_id:
            title += f" ({product_id})"
            
        # Calculate useful metrics for annotation
        if not historical.empty:
            hist_mean = historical['demand'].mean()
            hist_max = historical['demand'].max()
        else:
            hist_mean = forecast['forecast'].mean() if not forecast.empty else 0
            hist_max = forecast['forecast'].max() if not forecast.empty else 0
            
        if not forecast.empty:
            forecast_mean = forecast['forecast'].mean()
            forecast_max = forecast['forecast'].max()
            mean_diff_pct = ((forecast_mean - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0
        else:
            forecast_mean = 0
            forecast_max = 0
            mean_diff_pct = 0
            
        # Format plot annotations
        annotations = []
        annotations.append(
            dict(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=f"Historical Mean: {hist_mean:.1f} | Forecast Mean: {forecast_mean:.1f} | Change: {mean_diff_pct:.1f}%",
                showarrow=False,
                font=dict(size=12)
            )
        )
        
        y_upper_limit = max(hist_max, forecast_max) * 1.2 if max(hist_max, forecast_max) > 0 else 100
            
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Demand",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=annotations,
            hovermode="x unified",
            yaxis=dict(range=[0, y_upper_limit]),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating forecast plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Update the existing demand forecasting tab to include the advanced option
def demand_forecasting_tab():
    """Display demand forecasting tab"""
    st.header("Demand Forecasting")
    
    # Initialize ModelSelector
    data_loader = DataLoader()
    features = FeatureEngineering(data_loader)
    model_selector = ModelSelector(data_loader, features)
    
    # Get list of products
    products_data = data_loader.load_products_data()
    product_options = [(row['product_id'], f"{row['name']} ({row['product_id']})") 
                      for _, row in products_data.iterrows()]
    
    # UI controls - first row
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_product = st.selectbox(
            "Select Product", 
            [pid for pid, _ in product_options],
            format_func=lambda x: next((name for pid, name in product_options if pid == x), x)
        )
        
    with col2:
        forecast_type = st.radio(
            "Forecast Type", 
            ["Standard Forecast", "Advanced Ensemble Forecast"],
            index=1  # Default to advanced
        )
    
    # UI controls - second row: Forecast parameters
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Add time horizon selection options
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["1 Month (30 days)", "3 Months (90 days)", "6 Months (180 days)", "1 Year (365 days)", "Custom"],
            index=0
        )
        
        # If custom is selected, show the slider
        if forecast_horizon == "Custom":
            forecast_days = st.slider("Forecast Days", 7, 365, 30)
        else:
            # Extract days from selected option
            forecast_days = {
                "1 Month (30 days)": 30,
                "3 Months (90 days)": 90,
                "6 Months (180 days)": 180,
                "1 Year (365 days)": 365
            }[forecast_horizon]
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level", 
            min_value=0.8, 
            max_value=0.99, 
            value=0.95, 
            format="%0.0f%%",
            step=0.01,
            help="Confidence level for prediction intervals"
        )
        
    with col3:
        # Add a scaling factor for forecast adjustment
        scaling_factor = st.slider(
            "Forecast Scaling",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Adjust forecast values up or down (1.0 = no adjustment)"
        )
    
    # Get product info
    product_info = products_data[products_data['product_id'] == selected_product].iloc[0].to_dict() if not products_data[products_data['product_id'] == selected_product].empty else None
    
    # Load historical data
    historical_data = data_loader.get_demand_by_product(selected_product)
    
    # Generate a unique key for this forecast configuration
    forecast_key = f"{selected_product}_{forecast_type}_{forecast_days}_{confidence_level:.2f}_{scaling_factor:.2f}"
    
    # Check if forecast already exists in session state
    forecast_exists = forecast_key in st.session_state.forecasts
    
    # Create forecast button
    forecast_col1, forecast_col2 = st.columns([3, 1])
    with forecast_col1:
        forecast_clicked = st.button("Generate Forecast", use_container_width=True)
    
    with forecast_col2:
        # Add a Clear Cache button
        if st.button("Clear Cache", help="Clear saved forecast data"):
            # Clear the forecasts from session state
            st.session_state.forecasts = {}
            st.success("Forecast cache cleared!")
            st.experimental_rerun()
    
    # Display historical data summary
    if not historical_data.empty:
        st.subheader("Historical Data Summary")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Historical Mean", f"{historical_data['demand'].mean():.1f}")
        
        with col2:
            st.metric("Historical Max", f"{historical_data['demand'].max():.1f}")
            
        with col3:
            # Calculate coefficient of variation
            cv = historical_data['demand'].std() / historical_data['demand'].mean() if historical_data['demand'].mean() > 0 else 0
            st.metric("Variability (CV)", f"{cv:.2f}")
            
        with col4:
            # Calculate recent trend (last 30 days vs previous 30 days)
            if len(historical_data) >= 60:
                recent = historical_data.sort_values('date').tail(30)['demand'].mean()
                previous = historical_data.sort_values('date').iloc[-60:-30]['demand'].mean()
                trend_pct = ((recent - previous) / previous) * 100 if previous > 0 else 0
                st.metric("30-Day Trend", f"{trend_pct:.1f}%", delta=f"{trend_pct:.1f}%")
            else:
                st.metric("30-Day Trend", "Insufficient Data")
    
    # Generate or retrieve forecast data
    forecast = None
    
    # If we've got a cached forecast, use it
    if forecast_exists and not forecast_clicked:
        forecast = st.session_state.forecasts[forecast_key]
        st.info(f"Displaying cached forecast. Click 'Generate Forecast' to refresh.")
    
    # Generate new forecast when button is clicked
    if forecast_clicked or (not forecast_exists and not forecast):
        if forecast_type == "Advanced Ensemble Forecast":
            # Use the advanced forecast
            forecast = advanced_demand_forecast(
                model_selector, 
                selected_product, 
                periods=forecast_days,
                confidence=confidence_level
            )
            
            # Apply scaling factor if it's not 1.0
            if scaling_factor != 1.0 and forecast is not None:
                forecast['forecast'] = forecast['forecast'] * scaling_factor
                forecast['lower_ci'] = forecast['lower_ci'] * scaling_factor
                forecast['upper_ci'] = forecast['upper_ci'] * scaling_factor
                st.info(f"Forecast values scaled by factor {scaling_factor}")
                
        else:
            # Use the standard forecast
            # Select a model type (default to random forest)
            model_type = "RandomForestModel"
            
            # Load or train a model
            model_path = os.path.join('models', f'{model_type}_{selected_product}.joblib')
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    print(f"Loaded existing {model_type} model for {selected_product}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    model = None
            else:
                model = None
            
            if model is None:
                # Train a new model
                with st.spinner("Training model..."):
                    model_results = model_selector.train_demand_forecasting_models(selected_product)
                    if 'best_model' in model_results:
                        model = model_results['best_model']
                    else:
                        st.error("Failed to train a model. Please try another product.")
                        return
            
            # Generate forecast
            forecast = generate_demand_forecast(model_selector, model, selected_product, forecast_days)
            
            # Apply scaling factor if it's not 1.0
            if scaling_factor != 1.0 and forecast is not None:
                forecast['forecast'] = forecast['forecast'] * scaling_factor
                forecast['lower_ci'] = forecast['lower_ci'] * scaling_factor
                forecast['upper_ci'] = forecast['upper_ci'] * scaling_factor
                st.info(f"Forecast values scaled by factor {scaling_factor}")
        
        # Save forecast to session state if it was successfully generated
        if forecast is not None:
            # Convert dates to strings for JSON serialization
            forecast_for_storage = forecast.copy()
            forecast_for_storage['date'] = forecast_for_storage['date'].astype(str)
            st.session_state.forecasts[forecast_key] = forecast_for_storage
    
    # Restore date column back to datetime if loaded from session state
    if forecast is not None and isinstance(forecast['date'].iloc[0], str):
        forecast['date'] = pd.to_datetime(forecast['date'])
    
    # Display forecast data if available
    if forecast is not None:
        # Create plot tabs
        plot_tabs = st.tabs(["Interactive Forecast", "Forecast Details", "Data Table"])
        
        with plot_tabs[0]:
            # Plot forecast with confidence intervals
            fig = plot_demand_forecast_with_confidence(
                historical_data, 
                forecast, 
                product_info, 
                display_days=min(90, len(historical_data))
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create forecast plot")
        
        with plot_tabs[1]:
            # Show additional forecast details
            st.subheader("Forecast Details")
            
            # Show forecast statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average Forecast", 
                    f"{forecast['forecast'].mean():.1f}",
                    delta=f"{(forecast['forecast'].mean() - historical_data['demand'].mean()) / historical_data['demand'].mean() * 100:.1f}%" 
                    if not historical_data.empty and historical_data['demand'].mean() > 0 else None
                )
            
            with col2:
                st.metric(
                    "Peak Forecast", 
                    f"{forecast['forecast'].max():.1f}",
                    delta=f"{(forecast['forecast'].max() - historical_data['demand'].max()) / historical_data['demand'].max() * 100:.1f}%"
                    if not historical_data.empty and historical_data['demand'].max() > 0 else None
                )
            
            with col3:
                # Calculate average width of confidence interval
                if 'lower_ci' in forecast.columns and 'upper_ci' in forecast.columns:
                    avg_width = ((forecast['upper_ci'] - forecast['lower_ci']) / forecast['forecast']).mean() * 100
                    st.metric("Avg. Confidence Interval", f"±{avg_width:.1f}%")
                else:
                    st.metric("Avg. Confidence Interval", "Not available")
            
            # Add forecast accuracy assessment
            st.subheader("Forecast Accuracy Assessment")
            
            # Compare forecast mean with historical mean
            historical_mean = historical_data['demand'].mean()
            forecast_mean = forecast['forecast'].mean()
            mean_diff_pct = ((forecast_mean - historical_mean) / historical_mean) * 100 if historical_mean > 0 else 0
            
            # Make a simple assessment based on the difference
            if abs(mean_diff_pct) <= 10:
                accuracy_status = "✓ Good"
                accuracy_color = "green"
            elif abs(mean_diff_pct) <= 25:
                accuracy_status = "⚠️ Fair"
                accuracy_color = "orange"
            else:
                accuracy_status = "❌ Poor"
                accuracy_color = "red"
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                <h4 style="margin-top: 0;">Forecast vs Historical Comparison</h4>
                <p>Historical Mean: <b>{historical_mean:.2f}</b> | Forecast Mean: <b>{forecast_mean:.2f}</b></p>
                <p>Difference: <b>{mean_diff_pct:.1f}%</b> | Status: <span style="color: {accuracy_color}; font-weight: bold;">{accuracy_status}</span></p>
                <p style="margin-bottom: 0; font-size: 0.9em;">Note: High differences may indicate significant trend changes or possible forecast issues.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show forecast by time period (weekly)
            st.subheader("Weekly Forecast Summary")
            
            # Add week number to forecast
            forecast_with_week = forecast.copy()
            # Properly extract the week number from the isocalendar method
            try:
                # For pandas >= 1.1.0
                forecast_with_week['week'] = forecast_with_week['date'].dt.isocalendar().week.astype(int)
            except AttributeError:
                # For older pandas versions
                forecast_with_week['week'] = forecast_with_week['date'].apply(lambda x: x.isocalendar()[1]).astype(int)
            
            # Group by week
            weekly_forecast = forecast_with_week.groupby('week').agg({
                'forecast': 'sum',
                'lower_ci': 'sum' if 'lower_ci' in forecast.columns else 'mean',
                'upper_ci': 'sum' if 'upper_ci' in forecast.columns else 'mean'
            }).reset_index()
            
            # Add week start date
            week_dates = {}
            for _, row in forecast_with_week.iterrows():
                week = row['week']
                if week not in week_dates or row['date'] < week_dates[week]:
                    week_dates[week] = row['date']
            
            weekly_forecast['start_date'] = weekly_forecast['week'].map(week_dates)
            weekly_forecast = weekly_forecast.sort_values('start_date')
            
            # Plot weekly forecast
            fig_weekly = go.Figure()
            
            fig_weekly.add_trace(go.Bar(
                x=weekly_forecast['start_date'],
                y=weekly_forecast['forecast'],
                name='Weekly Forecast',
                marker_color='#1a5fb4'
            ))
            
            # Add error bars for confidence intervals
            if 'lower_ci' in weekly_forecast.columns and 'upper_ci' in weekly_forecast.columns:
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_forecast['start_date'],
                    y=weekly_forecast['upper_ci'],
                    mode='lines',
                    name='Upper CI',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_forecast['start_date'],
                    y=weekly_forecast['lower_ci'],
                    mode='lines',
                    name='Lower CI',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(26, 95, 180, 0.2)',
                    showlegend=False
                ))
            
            fig_weekly.update_layout(
                title='Weekly Demand Forecast',
                xaxis_title='Week Starting',
                yaxis_title='Weekly Demand',
                hovermode='x unified',
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Monthly forecast summary (for longer forecasts)
            if forecast_days >= 60:
                st.subheader("Monthly Forecast Summary")
                
                # Add month information to forecast
                forecast_with_month = forecast.copy()
                forecast_with_month['month'] = forecast_with_month['date'].dt.month
                forecast_with_month['month_name'] = forecast_with_month['date'].dt.strftime('%b %Y')
                
                # Group by month
                monthly_forecast = forecast_with_month.groupby(['month', 'month_name']).agg({
                    'forecast': 'sum',
                    'lower_ci': 'sum',
                    'upper_ci': 'sum'
                }).reset_index()
                
                # Add month start date
                month_dates = {}
                for _, row in forecast_with_month.iterrows():
                    month = row['month']
                    if month not in month_dates or row['date'] < month_dates[month]:
                        month_dates[month] = row['date']
                
                monthly_forecast['start_date'] = monthly_forecast['month'].map(month_dates)
                monthly_forecast = monthly_forecast.sort_values('start_date')
                
                # Plot monthly forecast
                fig_monthly = go.Figure()
                
                fig_monthly.add_trace(go.Bar(
                    x=monthly_forecast['month_name'],
                    y=monthly_forecast['forecast'],
                    name='Monthly Forecast',
                    marker_color='#26a69a'
                ))
                
                # Add error bars for confidence intervals
                if 'lower_ci' in monthly_forecast.columns and 'upper_ci' in monthly_forecast.columns:
                    fig_monthly.add_trace(go.Scatter(
                        x=monthly_forecast['month_name'],
                        y=monthly_forecast['upper_ci'],
                        mode='lines',
                        name='Upper CI',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig_monthly.add_trace(go.Scatter(
                        x=monthly_forecast['month_name'],
                        y=monthly_forecast['lower_ci'],
                        mode='lines',
                        name='Lower CI',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(38, 166, 154, 0.2)',
                        showlegend=False
                    ))
                
                fig_monthly.update_layout(
                    title='Monthly Demand Forecast',
                    xaxis_title='Month',
                    yaxis_title='Monthly Demand',
                    hovermode='x unified',
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with plot_tabs[2]:
            # Show raw forecast data
            st.subheader("Forecast Data")
            
            # Format the DataFrame for display
            display_forecast = forecast.copy()
            display_forecast['date'] = display_forecast['date'].dt.date
            display_forecast = display_forecast.round(1)
            
            st.dataframe(display_forecast, use_container_width=True)
            
            # Add download button using our helper function
            download_forecast_csv(forecast, selected_product)

# Define the download_forecast_csv function 
def download_forecast_csv(forecast, selected_product):
    """Generate and enable download of forecast CSV"""
    # Format the DataFrame for display
    display_forecast = forecast.copy()
    
    # Convert date column to datetime if it's in string format (from session state)
    if 'date' in display_forecast.columns and len(display_forecast) > 0:
        if isinstance(display_forecast['date'].iloc[0], str):
            display_forecast['date'] = pd.to_datetime(display_forecast['date'])
    
    # Format dates for display
    display_forecast['date'] = display_forecast['date'].dt.date
    display_forecast = display_forecast.round(1)
    
    # Add download button
    csv = display_forecast.to_csv(index=False).encode('utf-8')
    timestamp = generate_timestamp()
    
    # Generate a unique key for this download button
    key = f"download-forecast-{selected_product}-{timestamp}"
    
    st.download_button(
        "Download Forecast CSV",
        csv,
        f"forecast_{selected_product}_{timestamp}.csv",
        "text/csv",
        key=key
    )

def create_inventory_projection(current_inventory, forecast_data, eoq, reorder_point, lead_time=7):
    """
    Create an inventory projection based on forecast data
    
    Parameters:
    -----------
    current_inventory : float
        Current inventory level
    forecast_data : pandas.DataFrame
        Forecast data with dates and demand forecast
    eoq : float
        Economic Order Quantity
    reorder_point : float
        Reorder Point
    lead_time : int, optional
        Lead time in days
        
    Returns:
    --------
    pandas.DataFrame: Projected inventory levels with order indicators
    """
    try:
        # Create a copy of the forecast data
        projection = forecast_data.copy()
        
        # Ensure date column is datetime
        projection['date'] = pd.to_datetime(projection['date'])
        
        # Sort by date
        projection = projection.sort_values('date')
        
        # Initialize inventory level with current inventory
        inventory_level = current_inventory
        
        # Initialize order tracking
        pending_orders = []  # List of (order_date, receive_date, quantity)
        
        # Initialize columns
        projection['inventory_level'] = 0.0
        projection['place_order'] = False
        projection['receive_order'] = False
        
        # Process each day
        for i, row in projection.iterrows():
            # Get forecasted demand for this day
            daily_demand = row['forecast']
            
            # Check if any pending orders are received today
            received_quantity = 0
            new_pending_orders = []
            
            for order_date, receive_date, quantity in pending_orders:
                if receive_date <= row['date']:
                    # Order received
                    received_quantity += quantity
                    # Mark this day as receiving an order
                    projection.loc[i, 'receive_order'] = True
                else:
                    # Order still pending
                    new_pending_orders.append((order_date, receive_date, quantity))
            
            # Update pending orders
            pending_orders = new_pending_orders
            
            # Add received quantity to inventory
            inventory_level += received_quantity
            
            # Subtract demand from inventory
            inventory_level = max(0, inventory_level - daily_demand)
            
            # Store inventory level
            projection.loc[i, 'inventory_level'] = inventory_level
            
            # Check if we need to place an order
            if inventory_level <= reorder_point and not any(order_date == row['date'] for order_date, _, _ in pending_orders):
                # Place an order
                projection.loc[i, 'place_order'] = True
                
                # Calculate receive date
                receive_date = row['date'] + pd.Timedelta(days=lead_time)
                
                # Add to pending orders
                pending_orders.append((row['date'], receive_date, eoq))
        
        return projection
    
    except Exception as e:
        print(f"Error creating inventory projection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 