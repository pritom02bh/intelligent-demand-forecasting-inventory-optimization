import os
import json

# Default configuration
DEFAULT_CONFIG = {
    # Data paths
    "data_path": "supply_chain_datasets",
    "models_path": "models",
    
    # Dashboard settings
    "dashboard": {
        "title": "Supply Chain Analytics Dashboard",
        "theme": "light",
        "default_page": "Supply Chain Overview"
    },
    
    # Forecasting settings
    "forecasting": {
        "default_horizon": 30,
        "train_test_split": 0.8,
        "default_models": ["linear_regression", "random_forest", "gradient_boosting"]
    },
    
    # Inventory settings
    "inventory": {
        "default_order_cost": 50.0,
        "default_holding_cost_rate": 0.25,
        "default_service_level": 0.95,
        "max_days_to_stockout": 60
    }
}

# Path to the config file
CONFIG_FILE = "config.json"


def load_config():
    """
    Load configuration from file or create default config
    
    Returns:
    --------
    dict: Configuration settings
    """
    # Check if config file exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Ensure all default keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG
    else:
        # Create default config
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG


def save_config(config):
    """
    Save configuration to file
    
    Parameters:
    -----------
    config : dict
        Configuration settings
        
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def update_config(section, key, value):
    """
    Update a specific configuration setting
    
    Parameters:
    -----------
    section : str
        Configuration section
    key : str
        Configuration key
    value : any
        New value
        
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    config = load_config()
    
    if section in config:
        if isinstance(config[section], dict):
            config[section][key] = value
        else:
            config[section] = {key: value}
    else:
        config[section] = {key: value}
    
    return save_config(config)


def get_setting(section, key, default=None):
    """
    Get a specific configuration setting
    
    Parameters:
    -----------
    section : str
        Configuration section
    key : str
        Configuration key
    default : any
        Default value if setting is not found
        
    Returns:
    --------
    any: Configuration value
    """
    config = load_config()
    
    if section in config and key in config[section]:
        return config[section][key]
    else:
        return default


# Load config when module is imported
config = load_config() 