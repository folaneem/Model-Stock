# Stock Market Prediction System

A predictive analytics system for stock market trends using LSTM neural networks.

## Project Overview

This project implements a stock market prediction system using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) architecture. The system provides:

- Real-time stock price prediction
- Interactive visualization dashboard
- Technical indicator analysis
- Multi-day forecasting capability

## Features

- Real-time stock price visualization
- Interactive prediction interface
- Multiple technical indicators
- Multi-day forecasting
- Exportable predictions
- User-friendly dashboard

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```
2. In the web interface:
   - Enter a stock ticker (e.g., AAPL, GOOGL)
   - Select prediction days
   - Click "Predict" to generate forecasts

## Project Structure

```
├── src/
│   ├── app.py           # Main Streamlit application
│   ├── models/
│   │   └── lstm_model.py # LSTM model implementation
│   └── utils/
│       └── data_processor.py # Data preprocessing utilities
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Technical Details

- **Model Architecture**: LSTM Neural Network
- **Data Processing**: Technical indicators (SMA, RSI, MACD)
- **Visualization**: Plotly and Streamlit
- **Data Source**: Yahoo Finance API

## Author

Neemat Folasayo OLAJIDE
LCU/UG/22/21837
Lead City University, Ibadan, Nigeria
