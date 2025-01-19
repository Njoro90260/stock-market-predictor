import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries
from decouple import config

configuration = {
    "alpha_vantage": {
        "key": config('API_KEY'),
        "symbol": "IBM",
        "outputsize": "compact",  # Changed to 'compact' for free-tier compatibility
        "key_close": "4. close",  # Key for close prices in `get_daily`
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 30,  # Adjusted for a smaller dataset
        "color_actual": "#001f3f",
    },
    "model": {
        "input_size": 1,  # Only using the close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def download_data(configuration):
    """Download daily stock data using Alpha Vantage API."""
    try:
        ts = TimeSeries(key=configuration["alpha_vantage"]["key"], output_format='json')
        data, meta_data = ts.get_daily(
            symbol=configuration["alpha_vantage"]["symbol"],
            outputsize=configuration["alpha_vantage"]["outputsize"]
        )
    except ValueError as e:
        print(f"Error fetching data: {e}")
        return None, None, None, None

    # Process the data
    data_date = list(data.keys())
    data_date.reverse()

    data_close_price = [
        float(data[date][configuration["alpha_vantage"]["key_close"]]) for date in data_date
    ]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = f"from {data_date[0]} to {data_date[-1]}"
    print("Number of data points:", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

def plot_data(data_date, data_close_price, num_data_points, display_date_range, configuration):
    """Plot the stock data."""
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=configuration["plots"]["color_actual"])

    xticks = [
        data_date[i] if (i % configuration["plots"]["xticks_interval"] == 0 or i == num_data_points - 1) else None
        for i in range(num_data_points)
    ]
    plt.xticks(np.arange(len(xticks)), xticks, rotation='vertical')
    plt.title(f"Daily close price for {configuration['alpha_vantage']['symbol']}, {display_date_range}")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.show()

# Fetch and plot the data
data_date, data_close_price, num_data_points, display_date_range = download_data(configuration)
if data_date:
    plot_data(data_date, data_close_price, num_data_points, display_date_range, configuration)

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x
    
    def inverse_transform(self, x):
        return(x*self.sd) + self.mu
    
scaler = Normalizer()
normalized_data_closed_price = scaler.fit_transform(data_close_price)

def prepare_data(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1 
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    #perform simple moving average
    #use the next day as label
    output = x[window_size:]
    return output