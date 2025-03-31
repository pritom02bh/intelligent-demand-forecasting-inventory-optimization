import os
import subprocess
import sys

def main():
    """Run the Streamlit dashboard for the Demand Forecasting and Inventory Optimization System"""
    # Check if models directory exists, create if not
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Set the command to run Streamlit
    command = [sys.executable, "-m", "streamlit", "run", "src/visualization/dashboard.py"]
    
    print("Starting Supply Chain Analytics Dashboard...")
    print("Press Ctrl+C to exit")
    
    # Run the Streamlit command
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        print("\nMake sure Streamlit is installed with: pip install streamlit")

if __name__ == "__main__":
    main() 