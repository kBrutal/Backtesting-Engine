import pandas as pd
import matplotlib.pyplot as plt
from utils import load_high_low, get_cfg
import plotly.graph_objects as go
import pandas as pd


def plot_btc_with_trades():
    # Convert Date columns to datetime if not already
    _, btc_data = load_high_low(get_cfg())
    trade_sheet = pd.read_csv("trade_sheet.csv")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=btc_data["datetime"],
            y=btc_data["close"],
            mode="lines",
            name="BTC Close Price",
            line=dict(color="blue"),
        )
    )

    # Add trade points for LONG and SHORT trades only
    for _, row in trade_sheet.iterrows():
        if row["order_status"] == "LONG":
            color = "green"
        elif row["order_status"] == "SHORT":
            color = "red"
        else:
            continue  # Skip if order_status is Squared_Off or any other status

        fig.add_trace(
            go.Scatter(
                x=[row["date_time"]],
                y=[row["executed_price"]],
                mode="markers",
                marker=dict(color=color, size=10),
                name=row["order_status"],
            )
        )

    # Update layout for readability
    fig.update_layout(
        title="BTC Close Price with Trade Marks",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
    )

    # Show the plot
    fig.show()
