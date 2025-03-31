"""
Intelligent Demand Forecasting and Inventory Optimization System

A comprehensive solution for predicting product demand and optimizing
inventory levels to improve supply chain efficiency.
"""

__version__ = '0.1.0'

# Import main components for easier access
from .data import DataLoader, FeatureEngineering
from .models import (
    BaseForecastModel, 
    LinearRegressionModel,
    RandomForestModel, 
    GradientBoostingModel,
    ARIMAModel,
    ExponentialSmoothingModel,
    ProphetModel,
    EnsembleForecastModel,
    ModelSelector,
    InventoryOptimizer,
    optimize_inventory_for_all_products
)
from .utils import (
    format_currency,
    format_number,
    calculate_percentage_change,
    load_config,
    save_config,
    update_config,
    get_setting,
    config
)
from .visualization import (
    create_dashboard
)

# Helper function for advanced forecasting
def create_advanced_forecast(product_id, periods=30, confidence=0.95):
    """
    Create an advanced ensemble forecast for a product
    
    Parameters:
    -----------
    product_id : str
        Product ID
    periods : int
        Number of periods to forecast
    confidence : float
        Confidence level for prediction intervals
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with forecast and confidence intervals
    """
    from .data import DataLoader
    from .data import FeatureEngineering
    from .models import ModelSelector
    
    # Create required objects
    data_loader = DataLoader()
    feature_engineering = FeatureEngineering(data_loader)
    model_selector = ModelSelector(data_loader, feature_engineering)
    
    # Generate advanced forecast
    return model_selector.advanced_forecast(
        product_id=product_id,
        periods=periods,
        confidence=confidence
    )

__all__ = [
    'DataLoader',
    'FeatureEngineering',
    'BaseForecastModel',
    'LinearRegressionModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'ARIMAModel',
    'ExponentialSmoothingModel',
    'ProphetModel',
    'EnsembleForecastModel',
    'ModelSelector',
    'InventoryOptimizer',
    'optimize_inventory_for_all_products',
    'format_currency',
    'format_number',
    'calculate_percentage_change',
    'load_config',
    'save_config',
    'update_config',
    'get_setting',
    'config',
    'create_dashboard',
    'create_advanced_forecast'
] 