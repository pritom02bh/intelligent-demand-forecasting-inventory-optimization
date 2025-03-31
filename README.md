# IntelliStock: Intelligent Demand Forecasting & Inventory Optimization Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.2%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Methods-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

IntelliStock is an advanced analytics platform that combines AI-powered demand forecasting with inventory optimization to help businesses make data-driven supply chain decisions.

![Dashboard Screenshot](assets/dashboard_screenshot.png)

## 🚀 Features

- **Advanced Demand Forecasting**: Utilizes ensemble machine learning models to predict future demand with high accuracy
- **Inventory Optimization**: Automatically calculates optimal reorder points and economic order quantities (EOQ)
- **Interactive Dashboard**: Visualizes historical data, forecasts, and inventory projections
- **What-if Analysis**: Test different scenarios to optimize inventory levels and reduce costs
- **Batch Optimization**: Process all products simultaneously for efficient inventory management

## 📋 Requirements

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- pmdarima (for ARIMA models)

## 🔧 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/intellistock.git
cd intellistock
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the application
```bash
python run.py
```

2. Open your browser and navigate to http://localhost:8501

## 📊 Dashboard Sections

### 1. Supply Chain Overview
Provides a high-level summary of inventory status, demand patterns, and key performance indicators.

### 2. Demand Forecasting
- Select products to forecast
- Choose forecasting time horizons (1 month to 1 year)
- Adjust confidence levels and view prediction intervals
- Analyze seasonal patterns and trends

### 3. Inventory Optimization
- Calculate Economic Order Quantities (EOQ) and Reorder Points
- Visualize inventory projections
- Generate order schedules
- Optimize safety stock levels

## 📂 Project Structure

```
intellistock/
├── data/                   # Data directory
│   ├── raw/                # Raw input data
│   └── processed/          # Processed data
├── models/                 # Trained models
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # ML model implementations
│   └── visualization/      # Dashboard and visualization components
├── tests/                  # Test suite
├── .gitignore              # Git ignore file
├── LICENSE                 # License file
├── README.md               # Project readme
├── requirements.txt        # Python dependencies
└── run.py                  # Main entry point
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For any questions or feedback, please reach out to [your-email@example.com](mailto:your-email@example.com).

---

*Built with ❤️ using Python and Streamlit* 