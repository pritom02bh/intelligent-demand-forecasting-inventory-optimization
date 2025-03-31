import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class DataLoader:
    """
    Class for loading and preprocessing supply chain datasets
    """
    def __init__(self, data_path='supply_chain_datasets'):
        """
        Initialize data loader with path to datasets
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing datasets
        """
        self.data_path = data_path
        self.datasets = {}
    
    def load_all_datasets(self):
        """
        Load all available datasets
        
        Returns:
        --------
        dict: Dictionary containing all loaded datasets
        """
        # Load inventory data
        self.datasets['inventory'] = self.load_inventory_data()
        
        # Load products data
        self.datasets['products'] = self.load_products_data()
        
        # Load stores data
        self.datasets['stores'] = self.load_stores_data()
        
        # Load suppliers data
        self.datasets['suppliers'] = self.load_suppliers_data()
        
        # Load product-supplier mapping
        self.datasets['product_supplier_mapping'] = self.load_product_supplier_mapping()
        
        # Load pending orders
        self.datasets['pending_orders'] = self.load_pending_orders()
        
        # Load external factors
        self.datasets['external_factors'] = self.load_external_factors()
        
        # Try to load demand data
        try:
            self.datasets['demand'] = self.load_demand_data()
        except Exception as e:
            print(f"Warning: Could not load demand data: {e}")
        
        return self.datasets
    
    def load_inventory_data(self):
        """
        Load and preprocess inventory data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed inventory data
        """
        file_path = os.path.join(self.data_path, 'inventory.csv')
        inventory_data = pd.read_csv(file_path)
        
        # Convert date to datetime
        inventory_data['date'] = pd.to_datetime(inventory_data['date'])
        
        return inventory_data
    
    def load_products_data(self):
        """
        Load and preprocess products data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed products data
        """
        file_path = os.path.join(self.data_path, 'products.csv')
        products_data = pd.read_csv(file_path)
        
        return products_data
    
    def load_stores_data(self):
        """
        Load and preprocess stores data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed stores data
        """
        file_path = os.path.join(self.data_path, 'stores.csv')
        stores_data = pd.read_csv(file_path)
        
        # Convert opening_date to datetime
        stores_data['opening_date'] = pd.to_datetime(stores_data['opening_date'])
        
        return stores_data
    
    def load_suppliers_data(self):
        """
        Load and preprocess suppliers data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed suppliers data
        """
        file_path = os.path.join(self.data_path, 'suppliers.csv')
        suppliers_data = pd.read_csv(file_path)
        
        return suppliers_data
    
    def load_product_supplier_mapping(self):
        """
        Load and preprocess product-supplier mapping data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed product-supplier mapping data
        """
        file_path = os.path.join(self.data_path, 'product_supplier_mapping.csv')
        mapping_data = pd.read_csv(file_path)
        
        return mapping_data
    
    def load_pending_orders(self):
        """
        Load and preprocess pending orders data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed pending orders data
        """
        file_path = os.path.join(self.data_path, 'pending_orders.csv')
        pending_orders = pd.read_csv(file_path)
        
        # Convert date columns to datetime
        pending_orders['order_date'] = pd.to_datetime(pending_orders['order_date'])
        pending_orders['arrival_date'] = pd.to_datetime(pending_orders['arrival_date'])
        
        return pending_orders
    
    def load_external_factors(self):
        """
        Load and preprocess external factors data
        
        Returns:
        --------
        dict: Dictionary containing different external factors
        """
        external_factors = {}
        
        # Load economic indicators
        try:
            file_path = os.path.join(self.data_path, 'external_economic_indicators.csv')
            economic_indicators = pd.read_csv(file_path)
            economic_indicators['month'] = pd.to_datetime(economic_indicators['month'])
            external_factors['economic_indicators'] = economic_indicators
        except Exception as e:
            print(f"Warning: Could not load economic indicators: {e}")
        
        # Load seasonal factors
        try:
            file_path = os.path.join(self.data_path, 'external_seasonal_factors.csv')
            seasonal_factors = pd.read_csv(file_path)
            seasonal_factors['month'] = pd.to_datetime(seasonal_factors['month'])
            # Convert string representations of dictionaries to actual dictionaries
            seasonal_factors['ratings'] = seasonal_factors['ratings'].apply(eval)
            external_factors['seasonal_factors'] = seasonal_factors
        except Exception as e:
            print(f"Warning: Could not load seasonal factors: {e}")
        
        # Load complete external factors (JSON)
        try:
            file_path = os.path.join(self.data_path, 'external_factors_complete.json')
            with open(file_path, 'r') as f:
                external_factors_complete = json.load(f)
            
            # Process dates in the JSON
            for category in external_factors_complete:
                for item in external_factors_complete[category]:
                    if 'date' in item:
                        item['date'] = pd.to_datetime(item['date'])
                    if 'month' in item:
                        item['month'] = pd.to_datetime(item['month'])
                    if 'start_date' in item:
                        item['start_date'] = pd.to_datetime(item['start_date'])
                    if 'end_date' in item:
                        item['end_date'] = pd.to_datetime(item['end_date'])
            
            external_factors['complete'] = external_factors_complete
        except Exception as e:
            print(f"Warning: Could not load complete external factors: {e}")
        
        return external_factors
    
    def load_demand_data(self):
        """
        Load and preprocess demand data
        
        Returns:
        --------
        pandas.DataFrame: Preprocessed demand data
        """
        file_path = os.path.join(self.data_path, 'demand.csv')
        demand_data = pd.read_csv(file_path)
        
        # If demand data exists, process it
        if 'date' in demand_data.columns:
            demand_data['date'] = pd.to_datetime(demand_data['date'])
        
        return demand_data
    
    def prepare_demand_from_inventory(self):
        """
        Derive demand data from inventory transactions if demand.csv is not available
        
        Returns:
        --------
        pandas.DataFrame: Processed demand data
        """
        if 'inventory' not in self.datasets:
            self.datasets['inventory'] = self.load_inventory_data()
        
        inventory_data = self.datasets['inventory']
        
        # Filter only consumption transactions (negative quantities)
        consumption_data = inventory_data[inventory_data['transaction_type'] == 'consumption'].copy()
        
        # Convert consumption to positive demand values
        consumption_data['demand'] = -consumption_data['quantity']
        
        # Group by date and product_id to get daily demand
        demand_data = consumption_data.groupby(['date', 'product_id'])['demand'].sum().reset_index()
        
        return demand_data
    
    def get_demand_by_product(self, product_id=None):
        """
        Get demand data for a specific product or all products
        
        Parameters:
        -----------
        product_id : str, optional
            Product ID to filter demand data
            
        Returns:
        --------
        pandas.DataFrame: Demand data for specified product(s)
        """
        # Check if demand data is already loaded
        if 'demand' not in self.datasets:
            try:
                self.datasets['demand'] = self.load_demand_data()
            except Exception:
                # If demand data file is not available, derive from inventory
                self.datasets['demand'] = self.prepare_demand_from_inventory()
        
        demand_data = self.datasets['demand']
        
        if product_id:
            return demand_data[demand_data['product_id'] == product_id]
        else:
            return demand_data
    
    def get_products_with_demand(self):
        """
        Get list of products that have demand data
        
        Returns:
        --------
        list: List of product IDs that have demand data
        """
        if 'demand' not in self.datasets or self.datasets['demand'].empty:
            return []
        
        # Get unique product IDs from the demand data
        return self.datasets['demand']['product_id'].unique().tolist()
    
    def get_inventory_by_product(self, product_id, date=None):
        """
        Get inventory data for a specific product
        
        Parameters:
        -----------
        product_id : str
            Product ID
        date : datetime, optional
            Date to filter inventory data
            
        Returns:
        --------
        pandas.DataFrame: Inventory data for the product
        """
        if 'inventory' not in self.datasets or self.datasets['inventory'].empty:
            return pd.DataFrame()
        
        inventory_data = self.datasets['inventory'][self.datasets['inventory']['product_id'] == product_id].copy()
        
        if date is not None:
            inventory_data = inventory_data[inventory_data['date'] <= date]
        
        return inventory_data.sort_values('date')
    
    def get_product_name(self, product_id):
        """
        Get the name of a product given its ID
        
        Parameters:
        -----------
        product_id : str
            Product ID
            
        Returns:
        --------
        str: Product name or product_id if not found
        """
        if 'products' not in self.datasets or self.datasets['products'].empty:
            return str(product_id)
        
        product_data = self.datasets['products'][self.datasets['products']['product_id'] == product_id]
        
        if product_data.empty:
            return str(product_id)
        
        return f"{product_id} - {product_data.iloc[0]['name']}"


if __name__ == "__main__":
    # Test data loading
    data_loader = DataLoader()
    datasets = data_loader.load_all_datasets()
    
    print("Available datasets:")
    for key in datasets.keys():
        print(f"- {key}")
    
    # Print sample of each dataset
    for key, dataset in datasets.items():
        if isinstance(dataset, pd.DataFrame):
            print(f"\nSample of {key} dataset:")
            print(dataset.head(3))
        elif isinstance(dataset, dict):
            print(f"\n{key} contains multiple datasets:")
            for subkey in dataset.keys():
                print(f"- {subkey}") 