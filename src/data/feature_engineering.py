import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class FeatureEngineering:
    """
    Class for creating features for demand forecasting from supply chain data
    """
    def __init__(self, data_loader):
        """
        Initialize feature engineering with data loader
        
        Parameters:
        -----------
        data_loader : DataLoader
            Instance of DataLoader class with loaded datasets
        """
        self.data_loader = data_loader
        self.datasets = data_loader.datasets if data_loader.datasets else data_loader.load_all_datasets()
    
    def create_time_features(self, df, date_column='date'):
        """
        Create time-based features from date column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing date column
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional time features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Ensure date column is datetime
        if date_column in result_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
                result_df[date_column] = pd.to_datetime(result_df[date_column])
        else:
            print(f"Warning: Date column '{date_column}' not found in DataFrame")
            return result_df
            
        # Extract date components
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['day'] = result_df[date_column].dt.day
        result_df['day_of_week'] = result_df[date_column].dt.dayofweek
        result_df['day_of_year'] = result_df[date_column].dt.dayofyear
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
        result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
        
        # Create cyclical features for month, day of week, etc.
        # These capture the cyclical nature of time features better than linear values
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_year_sin'] = np.sin(2 * np.pi * result_df['day_of_year'] / 365)
        result_df['day_of_year_cos'] = np.cos(2 * np.pi * result_df['day_of_year'] / 365)
        
        # Add day of month
        result_df['day_of_month'] = result_df[date_column].dt.day
        
        # Special period indicators
        # First week of month
        result_df['is_first_week'] = (result_df['day_of_month'] <= 7).astype(int)
        # Last week of month (less precise but useful approximation)
        result_df['is_last_week'] = (result_df['day_of_month'] >= 23).astype(int)
        # First/last day of month
        result_df['is_first_day'] = (result_df['day_of_month'] == 1).astype(int)
        result_df['is_last_day'] = (result_df['day_of_month'] >= 28).astype(int)
        
        # Add trend feature (days since first date in dataset)
        min_date = result_df[date_column].min()
        result_df['days_since_start'] = (result_df[date_column] - min_date).dt.days
        
        return result_df
    
    def create_lagged_features(self, df, target_column, lags=[1, 7, 14, 30], group_column=None):
        """
        Create lagged features from target column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing target column
        target_column : str
            Name of the target column to create lags from
        lags : list
            List of lag values (in days)
        group_column : str or list, optional
            Column(s) to group by before creating lags (e.g., product_id)
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional lagged features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Ensure the DataFrame is sorted by date
        if group_column:
            result_df = result_df.sort_values([group_column, 'date'])
        else:
            result_df = result_df.sort_values('date')
        
        # Create lagged features
        for lag in lags:
            if group_column:
                lag_values = result_df.groupby(group_column)[target_column].shift(lag)
            else:
                lag_values = result_df[target_column].shift(lag)
            
            result_df[f'{target_column}_lag_{lag}'] = lag_values
        
        # Drop rows with NaN values due to lagging
        result_df = result_df.dropna()
        
        return result_df
    
    def create_rolling_aggregates(self, df, target_column, windows=[7, 14, 30], aggs=['mean', 'std'], group_column=None):
        """
        Create rolling window aggregation features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing target column
        target_column : str
            Name of the target column for rolling aggregations
        windows : list
            List of rolling window sizes (in days)
        aggs : list
            List of aggregation functions to apply
        group_column : str or list, optional
            Column(s) to group by before creating rolling features
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional rolling aggregate features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Ensure the DataFrame is sorted by date
        if group_column:
            result_df = result_df.sort_values([group_column, 'date'])
        else:
            result_df = result_df.sort_values('date')
        
        # Create rolling aggregate features
        for window in windows:
            for agg in aggs:
                if group_column:
                    grouped = result_df.groupby(group_column)[target_column]
                    
                    if agg == 'mean':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = grouped.transform(lambda x: x.rolling(window).mean())
                    elif agg == 'std':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = grouped.transform(lambda x: x.rolling(window).std())
                    elif agg == 'min':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = grouped.transform(lambda x: x.rolling(window).min())
                    elif agg == 'max':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = grouped.transform(lambda x: x.rolling(window).max())
                else:
                    if agg == 'mean':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = result_df[target_column].rolling(window).mean()
                    elif agg == 'std':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = result_df[target_column].rolling(window).std()
                    elif agg == 'min':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = result_df[target_column].rolling(window).min()
                    elif agg == 'max':
                        result_df[f'{target_column}_rolling_{window}d_{agg}'] = result_df[target_column].rolling(window).max()
        
        # Replace NaN values with column means
        for col in result_df.columns:
            if col.startswith(f'{target_column}_rolling_'):
                result_df[col] = result_df[col].fillna(result_df[col].mean())
        
        return result_df
    
    def add_product_features(self, df, product_id_column='product_id'):
        """
        Add product features to the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing product_id column
        product_id_column : str
            Name of the product ID column
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional product features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Only process if we have product_id column and products data
        if product_id_column in result_df.columns and 'products' in self.datasets:
            # Get products data
            products_data = self.datasets['products']
            
            # Check if all product IDs exist in products_data
            unique_products = result_df[product_id_column].unique()
            missing_products = [pid for pid in unique_products if pid not in products_data['product_id'].values]
            
            if missing_products:
                print(f"Warning: {len(missing_products)} product IDs not found in products data: {missing_products[:5]}")
                # For missing products, we'll create default values
                for missing_pid in missing_products:
                    # Create a row with default values for missing product
                    default_product = {
                        'product_id': missing_pid,
                        'category': 'Unknown',
                        'base_price': products_data['base_price'].mean() if len(products_data) > 0 else 100,
                        'base_cost': products_data['base_cost'].mean() if len(products_data) > 0 else 70,
                        'lead_time_days': products_data['lead_time_days'].mean() if len(products_data) > 0 else 7
                    }
                    # Add to products_data
                    products_data = pd.concat([products_data, pd.DataFrame([default_product])], ignore_index=True)
            
            # Merge product features with the main DataFrame
            try:
                result_df = pd.merge(
                    result_df,
                    products_data[['product_id', 'category', 'base_price', 'base_cost', 'lead_time_days']],
                    left_on=product_id_column,
                    right_on='product_id',
                    how='left'
                )
                
                # Create price-related features
                result_df['margin'] = result_df['base_price'] - result_df['base_cost']
                result_df['margin_percentage'] = (result_df['margin'] / result_df['base_price']) * 100
                
                # Create one-hot encoded features for product category
                category_dummies = pd.get_dummies(result_df['category'], prefix='category')
                result_df = pd.concat([result_df, category_dummies], axis=1)
            except Exception as e:
                print(f"Error adding product features: {str(e)}")
        else:
            print(f"Skipping add_product_features: product_id_column={product_id_column} not found or products data missing")
        
        return result_df
    
    def add_seasonal_factors(self, df, date_column='date'):
        """
        Add seasonal factors from external data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing date column
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional seasonal factors
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Get external factors data
        if 'external_factors' in self.datasets:
            external_factors = self.datasets['external_factors']
            
            # Add seasonal ratings if available
            if 'seasonal_factors' in external_factors:
                seasonal_factors = external_factors['seasonal_factors']
                
                # Create a month column for joining
                result_df['month_year'] = result_df[date_column].dt.strftime('%Y-%m')
                seasonal_factors['month_year'] = seasonal_factors['month'].dt.strftime('%Y-%m')
                
                # Prepare seasonal ratings for each category
                for df_row in result_df.itertuples():
                    category = df_row.category.lower() if hasattr(df_row, 'category') else None
                    month_year = df_row.month_year
                    
                    # Find matching seasonal factor
                    for sf_row in seasonal_factors.itertuples():
                        if sf_row.month_year == month_year:
                            ratings = sf_row.ratings
                            
                            # Add rating if category exists in ratings
                            if category and category in ratings:
                                result_df.at[df_row.Index, f'seasonal_rating_{category}'] = ratings[category]
                            elif 'electronics' in ratings and category == 'electronics':
                                result_df.at[df_row.Index, 'seasonal_rating_electronics'] = ratings['electronics']
                            elif 'clothing' in ratings and category == 'clothing':
                                result_df.at[df_row.Index, 'seasonal_rating_clothing'] = ratings['clothing']
                            elif 'home goods' in ratings and category == 'home goods':
                                result_df.at[df_row.Index, 'seasonal_rating_home_goods'] = ratings['home goods']
            
            # Add holiday flags if available
            if 'complete' in external_factors and 'major_holidays' in external_factors['complete']:
                holidays = external_factors['complete']['major_holidays']
                
                # Convert to DataFrame for easier processing
                holidays_df = pd.DataFrame(holidays)
                holidays_df['date'] = pd.to_datetime(holidays_df['date'])
                
                # Add holiday flag
                result_df['is_holiday'] = 0
                
                for holiday in holidays:
                    holiday_date = pd.to_datetime(holiday['date'])
                    result_df.loc[result_df[date_column] == holiday_date, 'is_holiday'] = 1
                    
                    # Add impact level for specific categories if available
                    if 'impact_level' in holiday:
                        for category, impact in holiday['impact_level'].items():
                            column_name = f'holiday_impact_{category}'
                            if column_name not in result_df.columns:
                                result_df[column_name] = 0
                            result_df.loc[result_df[date_column] == holiday_date, column_name] = impact
            
            # Add promotion flags if available
            if 'complete' in external_factors and 'promotional_events' in external_factors['complete']:
                promotions = external_factors['complete']['promotional_events']
                
                # Add promotion flag
                result_df['is_promotion'] = 0
                result_df['promotion_discount'] = 0.0
                
                for promotion in promotions:
                    start_date = pd.to_datetime(promotion['start_date'])
                    end_date = pd.to_datetime(promotion['end_date'])
                    affected_categories = promotion['affected_categories']
                    discount_rate = promotion['discount_rate']
                    
                    # Mark dates within promotion period
                    mask = (result_df[date_column] >= start_date) & (result_df[date_column] <= end_date)
                    
                    # Apply only to affected categories if category column exists
                    if 'category' in result_df.columns:
                        for category in affected_categories:
                            category_mask = mask & (result_df['category'].str.lower() == category.lower())
                            result_df.loc[category_mask, 'is_promotion'] = 1
                            result_df.loc[category_mask, 'promotion_discount'] = discount_rate
                    else:
                        # If no category column, apply to all rows in the date range
                        result_df.loc[mask, 'is_promotion'] = 1
                        result_df.loc[mask, 'promotion_discount'] = discount_rate
        
        # Drop temporary columns
        if 'month_year' in result_df.columns:
            result_df = result_df.drop('month_year', axis=1)
        
        return result_df
    
    def add_economic_indicators(self, df, date_column='date'):
        """
        Add economic indicators from external data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing date column
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with additional economic indicators
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Get economic indicators data
        if 'external_factors' in self.datasets and 'economic_indicators' in self.datasets['external_factors']:
            indicators = self.datasets['external_factors']['economic_indicators']
            
            # Create a month column for joining
            result_df['month_year'] = result_df[date_column].dt.strftime('%Y-%m')
            indicators['month_year'] = indicators['month'].dt.strftime('%Y-%m')
            
            # Add economic indicators
            for indicator_col in ['consumer_confidence_index', 'inflation_rate']:
                if indicator_col in indicators.columns:
                    # Create a mapping from month_year to indicator value
                    indicator_map = dict(zip(indicators['month_year'], indicators[indicator_col]))
                    
                    # Add indicator values to result DataFrame
                    result_df[indicator_col] = result_df['month_year'].map(indicator_map)
                    
                    # Fill missing values with forward filling (or mean if no prior values)
                    result_df[indicator_col] = result_df[indicator_col].ffill()
                    result_df[indicator_col] = result_df[indicator_col].bfill()
                    result_df[indicator_col] = result_df[indicator_col].fillna(result_df[indicator_col].mean())
        
        # Drop temporary columns
        if 'month_year' in result_df.columns:
            result_df = result_df.drop('month_year', axis=1)
        
        return result_df
    
    def add_supply_chain_disruptions(self, df, date_column='date'):
        """
        Add supply chain disruption flags
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing date column
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with disruption indicators
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Get supply chain disruptions data
        if ('external_factors' in self.datasets and 'complete' in self.datasets['external_factors'] and
            'supply_chain_disruptions' in self.datasets['external_factors']['complete']):
            
            disruptions = self.datasets['external_factors']['complete']['supply_chain_disruptions']
            
            # Add disruption flag and severity
            result_df['has_disruption'] = 0
            result_df['disruption_severity'] = 0
            
            for disruption in disruptions:
                start_date = pd.to_datetime(disruption['date'])
                end_date = start_date + timedelta(days=disruption['duration_days'])
                affected_categories = disruption['affected_categories']
                severity = disruption['severity']
                
                # Convert severity to numeric value
                severity_value = 1 if severity == 'low' else 2 if severity == 'medium' else 3
                
                # Mark dates within disruption period
                mask = (result_df[date_column] >= start_date) & (result_df[date_column] <= end_date)
                
                # Apply only to affected categories if category column exists
                if 'category' in result_df.columns:
                    for category in affected_categories:
                        category_mask = mask & (result_df['category'].str.lower() == category.lower())
                        result_df.loc[category_mask, 'has_disruption'] = 1
                        result_df.loc[category_mask, 'disruption_severity'] = severity_value
                else:
                    # If no category column, apply to all rows in the date range
                    result_df.loc[mask, 'has_disruption'] = 1
                    result_df.loc[mask, 'disruption_severity'] = severity_value
        
        return result_df
    
    def create_features(self, df):
        """
        Create comprehensive feature set for any input DataFrame
        
        This is a more general method that applies appropriate feature engineering
        based on the columns available in the input DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame to create features for
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with generated features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        try:
            # Create time features if date column exists
            if 'date' in result_df.columns:
                result_df = self.create_time_features(result_df)
                
                # Add seasonal factors and economic indicators
                result_df = self.add_seasonal_factors(result_df)
                result_df = self.add_economic_indicators(result_df)
                result_df = self.add_supply_chain_disruptions(result_df)
            
            # Add product features if product_id column exists
            if 'product_id' in result_df.columns:
                result_df = self.add_product_features(result_df)
            
            # Create lagged features and rolling aggregates if demand column exists
            # and we have enough data (more than 30 rows)
            if 'demand' in result_df.columns and len(result_df) > 30:
                group_by = 'product_id' if 'product_id' in result_df.columns else None
                
                # For smaller datasets, use smaller lags to avoid losing too many rows
                if len(result_df) < 60:
                    lags = [1, 7, 14]
                    windows = [7, 14]
                else:
                    lags = [1, 7, 14, 30]
                    windows = [7, 14, 30]
                
                # Only create lags if we have enough data
                if len(result_df) > max(lags) + 5:
                    try:
                        result_df = self.create_lagged_features(result_df, 'demand', lags=lags, group_column=group_by)
                    except Exception as e:
                        print(f"Warning: Error creating lagged features: {str(e)}")
                
                # Only create rolling features if we have enough data
                if len(result_df) > max(windows) + 5:
                    try:
                        result_df = self.create_rolling_aggregates(result_df, 'demand', windows=windows, group_column=group_by)
                    except Exception as e:
                        print(f"Warning: Error creating rolling aggregates: {str(e)}")
            
            # Fill any remaining NaN values
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
            
            # Fill categorical columns with 'Unknown'
            cat_cols = result_df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                result_df[cat_cols] = result_df[cat_cols].fillna('Unknown')
                
            return result_df
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            # Return original dataframe if feature creation failed
            return df
    
    def create_features_for_demand_forecasting(self, demand_data=None, product_id=None):
        """
        Create comprehensive feature set for demand forecasting
        
        Parameters:
        -----------
        demand_data : pandas.DataFrame, optional
            Demand data; if not provided, it will be loaded from data_loader
        product_id : str, optional
            Product ID to filter demand data
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with all features for demand forecasting
        """
        # Get demand data if not provided
        if demand_data is None:
            demand_data = self.data_loader.get_demand_by_product(product_id)
        elif product_id is not None:
            demand_data = demand_data[demand_data['product_id'] == product_id]
        
        # Create time features
        result_df = self.create_time_features(demand_data)
        
        # Add product features if product_id column exists
        if 'product_id' in result_df.columns:
            result_df = self.add_product_features(result_df)
        
        # Create lagged features and rolling aggregates
        if 'demand' in result_df.columns:
            target_column = 'demand'
            group_by = 'product_id' if 'product_id' in result_df.columns else None
            
            result_df = self.create_lagged_features(result_df, target_column, group_column=group_by)
            result_df = self.create_rolling_aggregates(result_df, target_column, group_column=group_by)
        
        # Add seasonal factors
        result_df = self.add_seasonal_factors(result_df)
        
        # Add economic indicators
        result_df = self.add_economic_indicators(result_df)
        
        # Add supply chain disruptions
        result_df = self.add_supply_chain_disruptions(result_df)
        
        # Fill any remaining NaN values
        result_df = result_df.fillna(0)
        
        return result_df
    
    def generate_features_for_prediction(self, product_id, future_demand, last_date=None):
        """
        Generate features for future prediction dates without data leakage
        
        Parameters:
        -----------
        product_id : str
            Product ID
        future_demand : pandas.DataFrame
            DataFrame with future dates and dummy demand values
        last_date : datetime, optional
            Last date from historical data (for proper feature generation)
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with features for prediction
        """
        try:
            # Ensure the future_demand dataframe has the required columns
            if 'date' not in future_demand.columns:
                print("Error: future_demand must contain a 'date' column")
                return None
                
            if 'product_id' not in future_demand.columns:
                future_demand['product_id'] = product_id
                
            if 'demand' not in future_demand.columns:
                # Add dummy demand value
                future_demand['demand'] = 0
            
            # Get historical data for this product to generate proper time-dependent features
            historical_data = self.data_loader.get_demand_by_product(product_id)
            
            if historical_data.empty:
                print(f"Warning: No historical data found for product {product_id}")
                # Create basic features without historical context
                basic_features = self.create_time_features(future_demand)
                if 'product_id' in basic_features.columns:
                    basic_features = self.add_product_features(basic_features)
                basic_features = self.add_seasonal_factors(basic_features)
                basic_features = self.add_economic_indicators(basic_features)
                basic_features = self.add_supply_chain_disruptions(basic_features)
                return basic_features.drop(columns=['date', 'demand'], errors='ignore')
            
            # Determine last date if not provided
            if last_date is None:
                last_date = historical_data['date'].max()
            
            # For proper feature generation that includes time-dependent features,
            # we need to combine historical and future data, then filter out only future dates
            
            # Prepare historical data to get needed lag periods
            # Calculate the maximum lag period needed
            max_lag_period = 30  # Default maximum window for lagged features
            
            # Get historical data from last_date - max_lag_period
            if last_date is not None:
                from datetime import timedelta
                start_date = last_date - timedelta(days=max_lag_period)
                historical_subset = historical_data[historical_data['date'] >= start_date].copy()
            else:
                # Take the last max_lag_period days from historical data
                historical_subset = historical_data.sort_values('date').tail(max_lag_period).copy()
            
            # Ensure both dataframes have the same columns before concatenating
            common_cols = list(set(historical_subset.columns) & set(future_demand.columns))
            historical_for_concat = historical_subset[common_cols].copy()
            future_for_concat = future_demand[common_cols].copy()
            
            # Combine historical and future data
            combined_data = pd.concat([historical_for_concat, future_for_concat], ignore_index=True)
            combined_data = combined_data.sort_values('date')
            
            # Create all features on the combined dataset
            # First, create time features
            feature_df = self.create_time_features(combined_data)
            
            # Add product features
            feature_df = self.add_product_features(feature_df)
            
            # Create lagged features and rolling aggregates
            if 'demand' in feature_df.columns:
                group_by = 'product_id' if 'product_id' in feature_df.columns else None
                
                # Use smaller lag and window sizes suitable for prediction
                feature_df = self.create_lagged_features(feature_df, 'demand', 
                                                        lags=[1, 3, 7], 
                                                        group_column=group_by)
                
                feature_df = self.create_rolling_aggregates(feature_df, 'demand', 
                                                           windows=[3, 7, 14], 
                                                           group_column=group_by)
            
            # Add seasonal factors
            feature_df = self.add_seasonal_factors(feature_df)
            
            # Add economic indicators
            feature_df = self.add_economic_indicators(feature_df)
            
            # Add supply chain disruptions
            feature_df = self.add_supply_chain_disruptions(feature_df)
            
            # Filter out only the future dates
            min_future_date = future_demand['date'].min()
            future_features = feature_df[feature_df['date'] >= min_future_date].copy()
            
            # Drop non-feature columns
            future_features = future_features.drop(columns=['date', 'demand'], errors='ignore')
            
            # Fill any remaining NaN values
            future_features = future_features.fillna(0)
            
            # Ensure all features are numeric
            numeric_cols = future_features.select_dtypes(include=['number']).columns
            future_features = future_features[numeric_cols]
            
            return future_features
            
        except Exception as e:
            print(f"Error generating features for prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    
    data_loader = DataLoader()
    data_loader.load_all_datasets()
    
    feature_engineering = FeatureEngineering(data_loader)
    
    # Get demand data for a sample product
    product_id = 'ELE-001'  # Example product ID
    sample_demand = data_loader.get_demand_by_product(product_id)
    
    if not sample_demand.empty:
        # Create features for demand forecasting
        features_df = feature_engineering.create_features_for_demand_forecasting(sample_demand)
        
        print(f"Features created for {product_id}:")
        print(f"Number of rows: {len(features_df)}")
        print(f"Number of features: {len(features_df.columns)}")
        print("\nSample of features:")
        print(features_df.head(3))
        print("\nFeature columns:")
        print(features_df.columns.tolist())
    else:
        print(f"No demand data found for {product_id}. Testing with all products.")
        all_demand = data_loader.get_demand_by_product()
        if not all_demand.empty:
            features_df = feature_engineering.create_features_for_demand_forecasting(all_demand)
            print(f"Features created for all products:")
            print(f"Number of rows: {len(features_df)}")
            print(f"Number of features: {len(features_df.columns)}")
            print("\nSample of features:")
            print(features_df.head(3)) 