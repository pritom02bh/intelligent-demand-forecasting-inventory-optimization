import os
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_product_categories():
    """Generate realistic product categories and products using OpenAI."""
    
    prompt = """
    Generate a JSON structure of product categories and products for a supply chain dataset.
    Create 5 main product categories, each with 3-5 products.
    For each product, include:
    - product_id (format: CAT-XXX where CAT is category code and XXX is a number)
    - name
    - base_price (realistic wholesale price in USD)
    - base_cost (60-80% of base_price)
    - lead_time_days (typical manufacturing/shipping lead time, 7-45 days)
    - shelf_life_days (if applicable, otherwise null)
    - unit (e.g., "each", "kg", "liter")
    - min_order_quantity
    - package_size (how many units per standard package)
    
    Format the response as a valid JSON array of objects.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    
    content = response.choices[0].message.content
    
    # Fix common JSON issues that might occur in API responses
    try:
        products_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in product data: {e}")
        print("Attempting to fix malformed JSON...")
        
        # Fix common trailing comma issue
        content = content.replace(',}', '}').replace(',]', ']')
        
        # Try to clean the JSON by removing potential problematic characters
        content = content.strip()
        if content.startswith('```json'):
            content = content.split('```json', 1)[1]
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        
        content = content.strip()
        
        try:
            products_data = json.loads(content)
            print("JSON fixed successfully.")
        except json.JSONDecodeError as e:
            print(f"Could not fix JSON: {e}")
            print("Using a simplified default structure instead.")
            # Provide a minimal default structure with a few basic products
            products_data = [
                {
                    "product_id": "ELEC-001",
                    "name": "Basic Smartphone",
                    "base_price": 300,
                    "base_cost": 210,
                    "lead_time_days": 14,
                    "shelf_life_days": None,
                    "unit": "each",
                    "min_order_quantity": 10,
                    "package_size": 1
                },
                {
                    "product_id": "FOOD-001",
                    "name": "Organic Apple",
                    "base_price": 1.2,
                    "base_cost": 0.8,
                    "lead_time_days": 7,
                    "shelf_life_days": 14,
                    "unit": "kg",
                    "min_order_quantity": 50,
                    "package_size": 10
                }
            ]
    
    # Flatten the products data if it has a nested structure
    flat_products = []
    
    # Check if the structure is a nested dictionary with categories
    if isinstance(products_data, dict) and 'categories' in products_data:
        # Nested categories format
        for category in products_data['categories']:
            if 'products' in category and isinstance(category['products'], list):
                for product in category['products']:
                    if 'product_id' in product:
                        # Add category for reference
                        product_copy = product.copy()
                        product_copy['category'] = category.get('name', '')
                        flat_products.append(product_copy)
    elif isinstance(products_data, list):
        # Check if each item in the list is a category with products
        has_nested_structure = False
        for item in products_data:
            if isinstance(item, dict) and 'products' in item and isinstance(item['products'], list):
                has_nested_structure = True
                for product in item['products']:
                    if 'product_id' in product:
                        product_copy = product.copy()
                        product_copy['category'] = item.get('name', item.get('category', ''))
                        flat_products.append(product_copy)
        
        # If not a nested structure, assume it's already flat
        if not has_nested_structure:
            flat_products = [p for p in products_data if isinstance(p, dict) and 'product_id' in p]
    
    # If we couldn't extract any products, use default
    if not flat_products:
        print("Warning: Could not extract product data properly. Using default product list.")
        flat_products = [
            {
                "product_id": "ELEC-001",
                "name": "Basic Smartphone",
                "base_price": 300,
                "base_cost": 210,
                "lead_time_days": 14,
                "shelf_life_days": None,
                "unit": "each",
                "min_order_quantity": 10,
                "package_size": 1,
                "category": "Electronics"
            },
            {
                "product_id": "FOOD-001",
                "name": "Organic Apple",
                "base_price": 1.2,
                "base_cost": 0.8,
                "lead_time_days": 7,
                "shelf_life_days": 14,
                "unit": "kg",
                "min_order_quantity": 50,
                "package_size": 10,
                "category": "Food"
            }
        ]
    
    return flat_products

def generate_external_factors(start_date, end_date):
    """Generate external factors that might influence demand."""
    
    prompt = f"""
    Generate a JSON array of objects representing external factors that might influence product demand 
    for a supply chain from {start_date} to {end_date}. Include:
    
    1. Major holidays with dates and expected impact level on different product categories
    2. Promotional events with start date, end date, and affected product categories
    3. Economic indicators by month (consumer confidence index, inflation rate)
    4. Seasonal factors by month (1-10 rating for each product category)
    5. Supply chain disruptions (date, duration in days, affected product categories, severity)
    
    Format as valid JSON with separate arrays for each factor type.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    
    content = response.choices[0].message.content
    
    # Fix common JSON issues that might occur in API responses
    try:
        factors_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Attempting to fix malformed JSON...")
        
        # Fix common trailing comma issue
        content = content.replace(',}', '}').replace(',]', ']')
        
        # Try to clean the JSON by removing potential problematic characters
        content = content.strip()
        if content.startswith('```json'):
            content = content.split('```json', 1)[1]
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        
        content = content.strip()
        
        try:
            factors_data = json.loads(content)
            print("JSON fixed successfully.")
        except json.JSONDecodeError as e:
            print(f"Could not fix JSON: {e}")
            print("Using a simplified default structure instead.")
            # Provide a minimal default structure
            factors_data = {
                "holidays": [],
                "promotions": [],
                "economic_indicators": [],
                "seasonal_factors": [],
                "disruptions": []
            }
    
    return factors_data

def generate_store_locations():
    """Generate store locations data using OpenAI."""
    
    prompt = """
    Generate a JSON array of 10 store locations for a retail company. For each store include:
    - store_id
    - name
    - city
    - state
    - country
    - region (Northeast, Southeast, Midwest, West, Southwest)
    - size_category (small, medium, large)
    - opening_date (realistic date in the past 5 years)
    
    Format the response as a valid JSON array of objects.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    
    content = response.choices[0].message.content
    
    # Fix common JSON issues that might occur in API responses
    try:
        stores_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in store locations: {e}")
        print("Attempting to fix malformed JSON...")
        
        # Fix common trailing comma issue
        content = content.replace(',}', '}').replace(',]', ']')
        
        # Try to clean the JSON by removing potential problematic characters
        content = content.strip()
        if content.startswith('```json'):
            content = content.split('```json', 1)[1]
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        
        content = content.strip()
        
        try:
            stores_data = json.loads(content)
            print("JSON fixed successfully.")
        except json.JSONDecodeError as e:
            print(f"Could not fix JSON: {e}")
            print("Using a simplified default structure instead.")
            # Provide a minimal default structure with a few stores
            stores_data = [
                {
                    "store_id": "S001",
                    "name": "Downtown Flagship",
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "region": "Northeast",
                    "size_category": "large",
                    "opening_date": "2020-03-15"
                },
                {
                    "store_id": "S002",
                    "name": "Westside Mall",
                    "city": "Los Angeles",
                    "state": "CA",
                    "country": "USA",
                    "region": "West",
                    "size_category": "medium",
                    "opening_date": "2021-06-20"
                },
                {
                    "store_id": "S003",
                    "name": "Suburb Center",
                    "city": "Chicago",
                    "state": "IL",
                    "country": "USA",
                    "region": "Midwest",
                    "size_category": "small",
                    "opening_date": "2022-01-10"
                }
            ]
    
    return stores_data

def apply_seasonal_pattern(base, month, product_seasonality, product_id, external_factors):
    """Apply seasonal patterns to base demand."""
    # Extract seasonality factor for this product and month
    seasonality = next((s['rating'] for s in product_seasonality 
                        if s['month'] == month and s['product_id'] == product_id), 5) / 5
    
    # Apply seasonality
    result = base * (0.5 + seasonality)
    
    # Check for holidays in this month
    # First check if holidays exist in external_factors
    if 'holidays' in external_factors and external_factors['holidays']:
        try:
            month_holidays = [h for h in external_factors['holidays'] 
                            if datetime.strptime(h['date'], '%Y-%m-%d').month == month]
            
            for holiday in month_holidays:
                # Find if this product category is affected
                product_category = product_id.split('-')[0]
                # Check if impact field exists and has the right structure
                if 'impact' in holiday and isinstance(holiday['impact'], list):
                    impact = next((i for i in holiday['impact'] if i['category'] == product_category), None)
                    if impact and 'factor' in impact:
                        # Apply holiday impact (multiplicative)
                        impact_factor = impact['factor']
                        result *= impact_factor
        except (KeyError, ValueError) as e:
            # Skip holiday effects if data is malformed
            print(f"Warning: Could not process holiday effects: {e}")
    
    return result

def apply_promotions(demand, date, product_id, promotions):
    """Apply promotional effects to demand."""
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if promotions is a valid list
        if not isinstance(promotions, list):
            return demand
            
        # Check if date falls within any promotion period
        for promo in promotions:
            # Skip invalid promotions
            if not isinstance(promo, dict) or 'start_date' not in promo or 'end_date' not in promo:
                continue
                
            try:
                start = datetime.strptime(promo['start_date'], '%Y-%m-%d')
                end = datetime.strptime(promo['end_date'], '%Y-%m-%d')
                
                if start <= date_obj <= end:
                    # Check if this product category is affected
                    product_category = product_id.split('-')[0]
                    
                    # Check if affected_categories exists and is a list
                    if 'affected_categories' in promo and isinstance(promo['affected_categories'], list):
                        if product_category in promo['affected_categories']:
                            # Apply promotion impact (multiplicative)
                            impact_factor = promo.get('impact_factor', 0.2)  # Default 20% boost
                            demand *= (1 + impact_factor)
            except (ValueError, TypeError) as e:
                # Skip this promotion if dates are invalid
                continue
    except Exception as e:
        print(f"Warning: Error in apply_promotions: {e}")
    
    return demand

def apply_economic_indicators(demand, date, indicators):
    """Apply economic indicator effects to demand."""
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month_year = date_obj.strftime('%Y-%m')
        
        # Check if indicators is a valid list
        if not isinstance(indicators, list):
            return demand
            
        # Find indicators for this month
        month_indicators = next((i for i in indicators if isinstance(i, dict) and 
                               i.get('month') == month_year), None)
        
        if month_indicators:
            # Get indicators with default values if missing
            try:
                # Consumer confidence influence (higher confidence, higher demand)
                cci = month_indicators.get('consumer_confidence_index', 50)
                cci_factor = 0.8 + (cci / 100) * 0.4  # Scale to range 0.8-1.2
                
                # Inflation influence (higher inflation, lower demand)
                inflation = month_indicators.get('inflation_rate', 2)
                inflation_factor = 1.1 - (inflation / 10) * 0.2  # Scale to range 0.9-1.1
                
                # Apply both factors
                demand *= cci_factor * inflation_factor
            except (TypeError, ValueError) as e:
                # Skip economic effects if values are invalid
                print(f"Warning: Invalid economic indicator values: {e}")
    except Exception as e:
        print(f"Warning: Error in apply_economic_indicators: {e}")
    
    return demand

def apply_disruptions(demand, date, product_id, disruptions):
    """Apply supply chain disruptions to demand fulfillment."""
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if disruptions is a valid list
        if not isinstance(disruptions, list):
            return demand
            
        # Check if date falls within any disruption period
        for disruption in disruptions:
            # Skip invalid disruptions
            if not isinstance(disruption, dict) or 'date' not in disruption:
                continue
                
            try:
                start = datetime.strptime(disruption['date'], '%Y-%m-%d')
                duration = disruption.get('duration_days', 7)  # Default to 1 week
                end = start + timedelta(days=duration)
                
                if start <= date_obj <= end:
                    # Check if this product category is affected
                    product_category = product_id.split('-')[0]
                    
                    # Check if affected_categories exists and is a list
                    if 'affected_categories' in disruption and isinstance(disruption['affected_categories'], list):
                        if product_category in disruption['affected_categories']:
                            # Apply disruption severity (multiplicative reduction)
                            # More severe disruptions lead to more unfulfilled demand
                            severity = disruption.get('severity', 5)  # Default moderate severity
                            severity_factor = 1 - (severity / 10)
                            demand *= severity_factor
            except (ValueError, TypeError) as e:
                # Skip this disruption if dates are invalid
                continue
    except Exception as e:
        print(f"Warning: Error in apply_disruptions: {e}")
    
    return demand

def generate_demand_data(products, stores, external_factors, start_date, end_date):
    """Generate realistic daily demand data with patterns and external factor effects."""
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = [(start + timedelta(days=i)).strftime('%Y-%m-%d') 
                  for i in range((end - start).days + 1)]
    
    # Extract relevant factors with error handling
    # Ensure external_factors has required keys
    if not isinstance(external_factors, dict):
        print("Warning: external_factors is not a dict. Using empty default.")
        external_factors = {
            "holidays": [],
            "promotions": [],
            "economic_indicators": [],
            "seasonal_factors": [],
            "disruptions": []
        }
    
    # Extract factors with fallbacks to empty lists if missing
    promotions = external_factors.get('promotions', [])
    economic_indicators = external_factors.get('economic_indicators', [])
    disruptions = external_factors.get('disruptions', [])
    seasonal_factors = external_factors.get('seasonal_factors', [])
    
    # Flatten products data if nested
    if isinstance(products, dict) and 'categories' in products:
        flat_products = []
        for category in products['categories']:
            for product in category['products']:
                product['category'] = category['name']
                flat_products.append(product)
        products = flat_products
    
    # Validate products list
    if not products or not isinstance(products, list):
        print("Warning: products list is empty or invalid. Creating default products.")
        products = [
            {
                "product_id": "ELEC-001",
                "name": "Basic Smartphone",
                "base_price": 300,
                "base_cost": 210
            },
            {
                "product_id": "FOOD-001",
                "name": "Organic Apple",
                "base_price": 1.2,
                "base_cost": 0.8
            }
        ]
    
    # Validate stores list
    if not stores or not isinstance(stores, list):
        print("Warning: stores list is empty or invalid. Creating default stores.")
        stores = [
            {
                "store_id": "S001",
                "name": "Main Store",
                "size_category": "medium"
            }
        ]
    
    data = []
    
    # Generate base demand profiles for each product and store
    product_store_profiles = {}
    for product in products:
        # Skip invalid products
        if 'product_id' not in product:
            continue
            
        product_id = product['product_id']
        for store in stores:
            # Skip invalid stores
            if 'store_id' not in store or 'size_category' not in store:
                continue
                
            store_id = store['store_id']
            
            # Base demand depends on store size and random factors
            size_multiplier = {'small': 0.7, 'medium': 1.0, 'large': 1.5}
            # Use default multiplier if size_category not recognized
            store_size = store.get('size_category', 'medium')
            multiplier = size_multiplier.get(store_size, 1.0)
            
            base_demand = random.randint(10, 30) * multiplier
            
            # Some randomness in popularity across stores
            store_popularity = random.uniform(0.7, 1.3)
            
            product_store_profiles[(product_id, store_id)] = base_demand * store_popularity
    
    # Generate daily demand for each product and store
    for date in date_range:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month = date_obj.month
        weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend effect
        weekend_factor = 1.2 if weekday >= 5 else 1.0
        
        for product in products:
            # Skip invalid products
            if 'product_id' not in product:
                continue
                
            product_id = product['product_id']
            
            for store in stores:
                # Skip invalid stores
                if 'store_id' not in store:
                    continue
                    
                store_id = store['store_id']
                
                # Skip if profile doesn't exist
                if (product_id, store_id) not in product_store_profiles:
                    continue
                
                # Get base demand for this product-store combination
                base_demand = product_store_profiles[(product_id, store_id)]
                
                # Apply seasonal pattern
                demand = apply_seasonal_pattern(base_demand, month, seasonal_factors, product_id, external_factors)
                
                # Apply weekend effect
                demand *= weekend_factor
                
                # Apply promotions
                demand = apply_promotions(demand, date, product_id, promotions)
                
                # Apply economic indicators
                demand = apply_economic_indicators(demand, date, economic_indicators)
                
                # Apply random daily fluctuation (±15%)
                demand *= random.uniform(0.85, 1.15)
                
                # Round to whole units
                demand = max(0, round(demand))
                
                # Record actual demand (what customers wanted)
                actual_demand = demand
                
                # Apply supply disruptions (what was actually fulfilled)
                fulfilled_demand = apply_disruptions(demand, date, product_id, disruptions)
                fulfilled_demand = max(0, round(fulfilled_demand))
                
                # Add to dataset
                data.append({
                    'date': date,
                    'product_id': product_id,
                    'store_id': store_id,
                    'demand': actual_demand,
                    'sales': fulfilled_demand,
                    'unfulfilled_demand': actual_demand - fulfilled_demand
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def generate_inventory_data(demand_df, products, start_date, lead_times=None):
    """Generate inventory data based on demand data and product information."""
    
    # Check if demand_df is valid and has required columns
    if demand_df.empty:
        print("Warning: Demand DataFrame is empty, generating default demand data")
        # Create a minimal demand dataset
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [(start_date_obj + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]  # 30 days
        
        # Get product IDs or use default
        if isinstance(products, list) and products:
            product_ids = [p.get('product_id') for p in products if 'product_id' in p]
            if not product_ids:
                product_ids = ['ELEC-001', 'FOOD-001']
        else:
            product_ids = ['ELEC-001', 'FOOD-001']
        
        # Create default demand data
        default_demand = []
        for date in dates:
            for product_id in product_ids:
                default_demand.append({
                    'date': date,
                    'product_id': product_id,
                    'demand': random.randint(5, 20)
                })
        demand_df = pd.DataFrame(default_demand)
    
    # Check if required columns exist
    required_columns = ['date', 'product_id', 'demand']
    missing_columns = [col for col in required_columns if col not in demand_df.columns]
    
    if missing_columns:
        print(f"Warning: Demand DataFrame missing required columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            if col == 'date':
                demand_df['date'] = start_date
            elif col == 'product_id':
                demand_df['product_id'] = 'UNKNOWN-001'
            elif col == 'demand':
                demand_df['demand'] = random.randint(5, 20)
    
    # Group by product and date to get total daily demand
    daily_demand = demand_df.groupby(['date', 'product_id'])['demand'].sum().reset_index()
    
    # Create a mapping of product_id to product details
    product_map = {}
    if isinstance(products, list):
        for p in products:
            if isinstance(p, dict) and 'product_id' in p:
                product_map[p['product_id']] = p
    
    # If no valid products, create default product data
    if not product_map:
        print("Warning: No valid products found. Using default product data.")
        for product_id in daily_demand['product_id'].unique():
            product_map[product_id] = {
                'product_id': product_id,
                'base_cost': 10,
                'lead_time_days': 14,
                'min_order_quantity': 10,
                'package_size': 1
            }
    
    inventory_data = []
    
    # Define random starting inventory levels for each product
    # Aim for ~30-60 days of average demand
    product_inventory = {}
    for product_id in daily_demand['product_id'].unique():
        avg_daily_demand = daily_demand[daily_demand['product_id'] == product_id]['demand'].mean()
        product_inventory[product_id] = max(30, int(avg_daily_demand * random.randint(30, 60)))
    
    # Track when orders are placed and when they arrive
    pending_orders = {}
    
    # Process each day
    all_dates = sorted(daily_demand['date'].unique())
    
    for date in all_dates:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Process arriving orders
        arrivals = [order_id for order_id, details in pending_orders.items() 
                   if details['arrival_date'] == date]
        
        for order_id in arrivals:
            order = pending_orders[order_id]
            product_inventory[order['product_id']] += order['quantity']
            # Add receiving record
            inventory_data.append({
                'date': date,
                'product_id': order['product_id'],
                'transaction_type': 'receiving',
                'quantity': order['quantity'],
                'ending_inventory': product_inventory[order['product_id']],
                'order_id': order_id
            })
            # Remove from pending
            del pending_orders[order_id]
        
        # Process daily demand and check inventory levels
        day_demand = daily_demand[daily_demand['date'] == date]
        
        for _, row in day_demand.iterrows():
            product_id = row['product_id']
            demand = row['demand']
            
            # Deduct from inventory
            if product_id in product_inventory:
                fulfilled = min(demand, product_inventory[product_id])
                product_inventory[product_id] -= fulfilled
                
                # Record consumption
                inventory_data.append({
                    'date': date,
                    'product_id': product_id,
                    'transaction_type': 'consumption',
                    'quantity': -fulfilled,
                    'ending_inventory': product_inventory[product_id],
                    'order_id': None
                })
                
                # Check if we need to place an order
                # Skip if product not in product_map
                if product_id not in product_map:
                    continue
                    
                product = product_map[product_id]
                
                # Get lead time (either from provided dict or from product info)
                if lead_times and product_id in lead_times:
                    lead_time = lead_times[product_id]
                else:
                    lead_time = product.get('lead_time_days', 14)
                
                # Calculate reorder point: Cover lead time demand + safety stock
                avg_daily_demand = day_demand[day_demand['product_id'] == product_id]['demand'].mean()
                safety_stock = avg_daily_demand * 7  # 1 week safety stock
                reorder_point = avg_daily_demand * lead_time + safety_stock
                
                # If below reorder point and no pending orders, place order
                pending_for_product = sum(order['quantity'] for order in pending_orders.values() 
                                         if order['product_id'] == product_id)
                
                if (product_inventory[product_id] + pending_for_product) < reorder_point:
                    # Calculate order quantity (aim to cover 30 days + lead time)
                    target_days = 30 + lead_time
                    order_quantity = max(
                        avg_daily_demand * target_days - product_inventory[product_id] - pending_for_product,
                        product.get('min_order_quantity', 1)
                    )
                    
                    # Round to package size if applicable
                    if 'package_size' in product and product['package_size'] > 1:
                        packages = max(1, round(order_quantity / product['package_size']))
                        order_quantity = packages * product['package_size']
                    
                    # Create order ID
                    order_id = f"ORD-{product_id}-{date.replace('-', '')}"
                    
                    # Calculate arrival date
                    arrival_date = (date_obj + timedelta(days=lead_time)).strftime('%Y-%m-%d')
                    
                    # Record order
                    inventory_data.append({
                        'date': date,
                        'product_id': product_id,
                        'transaction_type': 'order',
                        'quantity': order_quantity,
                        'ending_inventory': product_inventory[product_id],
                        'order_id': order_id
                    })
                    
                    # Add to pending orders
                    pending_orders[order_id] = {
                        'product_id': product_id,
                        'quantity': order_quantity,
                        'order_date': date,
                        'arrival_date': arrival_date
                    }
    
    # Convert to DataFrame
    inventory_df = pd.DataFrame(inventory_data)
    
    # Add costs and financial information (with error handling)
    if not inventory_df.empty:
        inventory_df['unit_cost'] = inventory_df['product_id'].apply(
            lambda x: product_map.get(x, {}).get('base_cost', 10))
        
        inventory_df['transaction_value'] = inventory_df.apply(
            lambda row: abs(row['quantity']) * row['unit_cost'], axis=1)
    else:
        # Create empty DataFrame with necessary columns
        inventory_df = pd.DataFrame(columns=['date', 'product_id', 'transaction_type', 
                                             'quantity', 'ending_inventory', 'order_id',
                                             'unit_cost', 'transaction_value'])
    
    # Create pending orders dataframe
    pending_orders_list = [
        {**order, 'status': 'pending' if order['arrival_date'] > all_dates[-1] else 'received'} 
        for order_id, order in pending_orders.items()
    ]
    pending_orders_df = pd.DataFrame(pending_orders_list) if pending_orders_list else pd.DataFrame()
    
    return inventory_df, pending_orders_df

def generate_supplier_data(products):
    """Generate supplier data for the products."""
    
    prompt = """
    Generate a JSON array of 15 suppliers for a supply chain dataset. For each supplier include:
    - supplier_id
    - name
    - country
    - reliability_score (1-10)
    - average_lead_time_days
    - payment_terms (e.g., "Net 30", "Net 60")
    - specialties (array of product categories they supply)
    
    Format the response as a valid JSON array of objects.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    
    content = response.choices[0].message.content
    
    # Fix common JSON issues that might occur in API responses
    try:
        suppliers_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in supplier data: {e}")
        print("Attempting to fix malformed JSON...")
        
        # Fix common trailing comma issue
        content = content.replace(',}', '}').replace(',]', ']')
        
        # Try to clean the JSON by removing potential problematic characters
        content = content.strip()
        if content.startswith('```json'):
            content = content.split('```json', 1)[1]
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        
        content = content.strip()
        
        try:
            suppliers_data = json.loads(content)
            print("JSON fixed successfully.")
        except json.JSONDecodeError as e:
            print(f"Could not fix JSON: {e}")
            print("Using a simplified default structure instead.")
            # Provide a minimal default structure with a few suppliers
            suppliers_data = [
                {
                    "supplier_id": "SUP001",
                    "name": "Global Electronics",
                    "country": "China",
                    "reliability_score": 8,
                    "average_lead_time_days": 25,
                    "payment_terms": "Net 45",
                    "specialties": ["ELEC"]
                },
                {
                    "supplier_id": "SUP002",
                    "name": "Organic Farms Co",
                    "country": "USA",
                    "reliability_score": 9,
                    "average_lead_time_days": 7,
                    "payment_terms": "Net 30",
                    "specialties": ["FOOD"]
                },
                {
                    "supplier_id": "SUP003",
                    "name": "Fashion Textiles Ltd",
                    "country": "India",
                    "reliability_score": 7,
                    "average_lead_time_days": 30,
                    "payment_terms": "Net 60",
                    "specialties": ["CLTH"]
                }
            ]
    
    # Check if products is empty or invalid
    if not products or not isinstance(products, list):
        print("Warning: Invalid product data. Creating sample product list.")
        products = [
            {
                "product_id": "ELEC-001",
                "name": "Basic Smartphone",
                "base_price": 300,
                "base_cost": 210
            },
            {
                "product_id": "FOOD-001",
                "name": "Organic Apple",
                "base_price": 1.2,
                "base_cost": 0.8
            }
        ]
    
    # Create product-supplier mapping
    product_categories = set()
    for product in products:
        # Check if product has product_id, if not skip
        if 'product_id' not in product:
            print(f"Warning: Found product without product_id: {product}")
            continue
            
        category = product['product_id'].split('-')[0]
        product_categories.add(category)
    
    # Make sure we have at least one product category
    if not product_categories:
        print("Warning: No valid product categories found. Using default categories.")
        product_categories = {"ELEC", "FOOD"}
    
    # Make sure suppliers have specialties for our categories
    for supplier in suppliers_data:
        if 'specialties' not in supplier or not supplier['specialties']:
            supplier['specialties'] = random.sample(list(product_categories), 
                                                  min(len(product_categories), random.randint(1, 3)))
    
    product_supplier_mappings = []
    
    for product in products:
        # Skip products without product_id
        if 'product_id' not in product:
            continue
            
        product_id = product['product_id']
        parts = product_id.split('-')
        if len(parts) < 2:
            print(f"Warning: Invalid product_id format: {product_id}, using default category")
            category = "DEFAULT"
        else:
            category = parts[0]
        
        # Find suppliers that supply this category
        matching_suppliers = [s for s in suppliers_data if category in s['specialties']]
        
        # If no suppliers match this category, assign a default supplier
        if not matching_suppliers:
            # Find the first supplier or create a default one
            if suppliers_data:
                default_supplier = suppliers_data[0]
                # Add this category to the supplier's specialties
                if 'specialties' in default_supplier:
                    default_supplier['specialties'].append(category)
                else:
                    default_supplier['specialties'] = [category]
                matching_suppliers = [default_supplier]
            else:
                # Create a default supplier if none exist
                default_supplier = {
                    "supplier_id": f"SUP-{category}",
                    "name": f"{category} Default Supplier",
                    "country": "USA",
                    "reliability_score": 7,
                    "average_lead_time_days": 14,
                    "payment_terms": "Net 30",
                    "specialties": [category]
                }
                suppliers_data.append(default_supplier)
                matching_suppliers = [default_supplier]
        
        # Assign 1-3 suppliers per product
        num_suppliers = min(len(matching_suppliers), random.randint(1, 3))
        selected_suppliers = random.sample(matching_suppliers, num_suppliers)
        
        for supplier in selected_suppliers:
            lead_time_variation = random.randint(-5, 5)
            
            # Ensure product has base_cost
            base_cost = product.get('base_cost', 10)
            
            product_supplier_mappings.append({
                'product_id': product_id,
                'supplier_id': supplier['supplier_id'],
                'unit_cost': base_cost * random.uniform(0.9, 1.1),  # Variation in cost
                'lead_time_days': supplier.get('average_lead_time_days', 14) + lead_time_variation,
                'is_primary': selected_suppliers.index(supplier) == 0,
                'min_order_quantity': product.get('min_order_quantity', 1) * random.randint(1, 3)
            })
    
    # Convert supplier_data to DataFrame
    suppliers_df = pd.DataFrame(suppliers_data)
    
    # Handle specialties column for CSV format
    if 'specialties' in suppliers_df.columns:
        suppliers_df['specialties'] = suppliers_df['specialties'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    
    return suppliers_df, pd.DataFrame(product_supplier_mappings)

def plot_demand_patterns(demand_df, products, output_folder):
    """Create visualizations of demand patterns for validation."""
    
    # Check if demand_df is valid and has required columns
    if demand_df.empty:
        print("Warning: Cannot create visualizations - demand DataFrame is empty")
        return
        
    # Check if required columns exist
    required_columns = ['date', 'product_id', 'demand']
    missing_columns = [col for col in required_columns if col not in demand_df.columns]
    
    if missing_columns:
        print(f"Warning: Cannot create visualizations - demand DataFrame missing columns: {missing_columns}")
        return
    
    # Continue with visualization creation
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Aggregate daily demand to weekly for better visualization
        demand_df['week'] = pd.to_datetime(demand_df['date']).dt.to_period('W').astype(str)
        weekly_demand = demand_df.groupby(['week', 'product_id'])['demand'].sum().reset_index()
        
        # Create product_id to name mapping
        product_map = {}
        
        # Handle different product structures
        if isinstance(products, dict) and 'categories' in products:
            # Handle nested product structure
            for category in products['categories']:
                if 'products' in category and isinstance(category['products'], list):
                    for product in category['products']:
                        if 'product_id' in product and 'name' in product:
                            product_map[product['product_id']] = product['name']
        elif isinstance(products, list):
            # Handle flat product structure
            for p in products:
                if isinstance(p, dict) and 'product_id' in p:
                    product_map[p['product_id']] = p.get('name', p['product_id'])
        
        # Plot for each product category
        categories = set(pid.split('-')[0] for pid in demand_df['product_id'].unique() if '-' in pid)
        
        # If no valid categories found, use all product IDs as a single category
        if not categories:
            categories = ['all']
            product_by_category = {'all': demand_df['product_id'].unique()}
        else:
            product_by_category = {cat: [pid for pid in demand_df['product_id'].unique() 
                                       if pid.startswith(cat)] for cat in categories}
        
        for category in categories:
            category_products = product_by_category[category]
            if not category_products:
                continue
                
            plt.figure(figsize=(12, 8))
            
            for product_id in category_products:
                product_data = weekly_demand[weekly_demand['product_id'] == product_id]
                if not product_data.empty:
                    plt.plot(product_data['week'], product_data['demand'], 
                            marker='o', linestyle='-', label=product_map.get(product_id, product_id))
            
            plt.title(f'Weekly Demand Patterns - {category} Category')
            plt.xlabel('Week')
            plt.ylabel('Total Demand')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_folder, f'demand_pattern_{category}.png'))
            plt.close()
        
        # Plot inventory levels if data is available
        if not demand_df.empty and all(col in demand_df.columns for col in ['transaction_type', 'ending_inventory']):
            inventory_df = demand_df[demand_df['transaction_type'].isin(['consumption', 'receiving'])]
            
            if not inventory_df.empty:
                sample_products = random.sample(list(demand_df['product_id'].unique()), 
                                              min(5, len(demand_df['product_id'].unique())))
                
                plt.figure(figsize=(12, 8))
                
                for product_id in sample_products:
                    product_inv = inventory_df[inventory_df['product_id'] == product_id]
                    if not product_inv.empty:
                        plt.plot(product_inv['date'], product_inv['ending_inventory'], 
                                marker='.', linestyle='-', label=product_map.get(product_id, product_id))
                
                plt.title('Daily Ending Inventory Levels - Sample Products')
                plt.xlabel('Date')
                plt.ylabel('Inventory Level')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_folder, 'inventory_levels.png'))
                plt.close()
    except Exception as e:
        print(f"Warning: Error creating visualizations: {e}")
        # Create a placeholder visualization to avoid errors
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Visualization not available - data format error", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.savefig(os.path.join(output_folder, 'visualization_error.png'))
        plt.close()

def save_datasets(datasets, output_folder):
    """Save all generated datasets to CSV files."""
    os.makedirs(output_folder, exist_ok=True)
    
    for name, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(os.path.join(output_folder, f'{name}.csv'), index=False)
            print(f"Saved {name}.csv")
        elif isinstance(data, list):
            # Convert list to DataFrame and save as CSV
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_folder, f'{name}.csv'), index=False)
            print(f"Saved {name}.csv")
            # Also save as JSON for reference
            with open(os.path.join(output_folder, f'{name}.json'), 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, dict):
            # Try to convert dict to DataFrame if possible
            try:
                if 'categories' in data:
                    # Handle nested product categories
                    flat_products = []
                    for category in data['categories']:
                        for product in category['products']:
                            product['category'] = category['name']
                            flat_products.append(product)
                    df = pd.DataFrame(flat_products)
                else:
                    # For flat dictionaries
                    df = pd.DataFrame([data])
                df.to_csv(os.path.join(output_folder, f'{name}.csv'), index=False)
                print(f"Saved {name}.csv")
            except Exception as e:
                print(f"Warning: Could not convert {name} to CSV: {e}")
                # Save as JSON if conversion fails
                with open(os.path.join(output_folder, f'{name}.json'), 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved {name}.json instead")

def main():
    """Main function to generate all datasets."""
    try:
        # Set date range for the dataset (e.g., 2 years of data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        print(f"Generating supply chain datasets from {start_date} to {end_date}")
        
        # Create output folders
        os.makedirs("supply_chain_datasets", exist_ok=True)
        os.makedirs("supply_chain_visualizations", exist_ok=True)
        
        datasets = {}
        
        # Generate datasets with error handling for each step
        try:
            print("Generating product categories...")
            products = generate_product_categories()
            datasets['products'] = products
        except Exception as e:
            print(f"Error generating products: {e}")
            products = [
                {
                    "product_id": "ELEC-001",
                    "name": "Basic Smartphone",
                    "base_price": 300,
                    "base_cost": 210,
                    "category": "Electronics"
                },
                {
                    "product_id": "FOOD-001",
                    "name": "Organic Apple",
                    "base_price": 1.2,
                    "base_cost": 0.8,
                    "category": "Food"
                }
            ]
            datasets['products'] = products
        
        try:
            print("Generating store locations...")
            stores = generate_store_locations()
            datasets['stores'] = stores
        except Exception as e:
            print(f"Error generating stores: {e}")
            stores = [
                {
                    "store_id": "S001",
                    "name": "Main Store",
                    "city": "New York",
                    "state": "NY",
                    "size_category": "medium"
                }
            ]
            datasets['stores'] = stores
        
        try:
            print("Generating external factors...")
            external_factors = generate_external_factors(start_date, end_date)
            
            # Extract external factors into separate DataFrames
            external_factors_dfs = {}
            for factor_type in ['holidays', 'promotions', 'economic_indicators', 'seasonal_factors', 'disruptions']:
                if factor_type in external_factors and isinstance(external_factors[factor_type], list):
                    external_factors_dfs[f'external_{factor_type}'] = pd.DataFrame(external_factors[factor_type])
            
            # Add external factors DataFrames to datasets
            datasets.update(external_factors_dfs)
            
            # Save complete external factors as JSON for reference
            with open(os.path.join("supply_chain_datasets", "external_factors_complete.json"), 'w') as f:
                json.dump(external_factors, f, indent=2)
        except Exception as e:
            print(f"Error generating external factors: {e}")
            external_factors = {
                "holidays": [],
                "promotions": [],
                "economic_indicators": [],
                "seasonal_factors": [],
                "disruptions": []
            }
        
        try:
            print("Generating supplier data...")
            suppliers_df, product_supplier_mapping = generate_supplier_data(products)
            datasets['suppliers'] = suppliers_df
            datasets['product_supplier_mapping'] = product_supplier_mapping
        except Exception as e:
            print(f"Error generating supplier data: {e}")
            # Create simple supplier dataframe
            suppliers_df = pd.DataFrame([
                {
                    "supplier_id": "SUP001",
                    "name": "Default Supplier",
                    "country": "USA",
                    "reliability_score": 8,
                    "specialties": "ELEC,FOOD"
                }
            ])
            # Create simple mapping dataframe
            mappings = []
            for p in products:
                if 'product_id' in p:
                    mappings.append({
                        'product_id': p['product_id'],
                        'supplier_id': "SUP001",
                        'unit_cost': p.get('base_cost', 10),
                        'lead_time_days': 14,
                        'is_primary': True,
                        'min_order_quantity': 10
                    })
            product_supplier_mapping = pd.DataFrame(mappings)
            
            datasets['suppliers'] = suppliers_df
            datasets['product_supplier_mapping'] = product_supplier_mapping
        
        try:
            print("Generating demand data...")
            demand_df = generate_demand_data(products, stores, external_factors, start_date, end_date)
            datasets['demand'] = demand_df
        except Exception as e:
            print(f"Error generating demand data: {e}")
            # Create simple demand DataFrame
            demand_data = []
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            dates = [(start_date_obj + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
            
            for date in dates:
                for p in products:
                    if 'product_id' in p:
                        for s in stores:
                            if 'store_id' in s:
                                demand_data.append({
                                    'date': date,
                                    'product_id': p['product_id'],
                                    'store_id': s['store_id'],
                                    'demand': random.randint(5, 20),
                                    'sales': random.randint(3, 18),
                                    'unfulfilled_demand': random.randint(0, 5)
                                })
            demand_df = pd.DataFrame(demand_data)
            datasets['demand'] = demand_df
        
        try:
            print("Generating inventory data...")
            inventory_df, pending_orders_df = generate_inventory_data(demand_df, products, start_date)
            datasets['inventory'] = inventory_df
            datasets['pending_orders'] = pending_orders_df
        except Exception as e:
            print(f"Error generating inventory data: {e}")
            # Create minimal inventory and pending orders DataFrames
            inventory_df = pd.DataFrame(columns=['date', 'product_id', 'transaction_type', 
                                                'quantity', 'ending_inventory', 'order_id',
                                                'unit_cost', 'transaction_value'])
            pending_orders_df = pd.DataFrame(columns=['product_id', 'quantity', 'order_date', 
                                                     'arrival_date', 'status'])
            datasets['inventory'] = inventory_df
            datasets['pending_orders'] = pending_orders_df
        
        try:
            # Create visualizations for validation
            print("Creating validation visualizations...")
            output_folder = "supply_chain_visualizations"
            plot_demand_patterns(demand_df, products, output_folder)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        # Save all datasets
        print("Saving datasets to CSV files...")
        save_datasets(datasets, "supply_chain_datasets")
        
        print("Dataset generation completed successfully!")
        print("All data has been saved in CSV format in the 'supply_chain_datasets' folder.")
        return True
    
    except Exception as e:
        print(f"Critical error in data generation: {e}")
        print("Attempted to save any generated data.")
        return False

if __name__ == "__main__":
    main()