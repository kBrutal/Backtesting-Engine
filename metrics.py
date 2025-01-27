import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import handle_date_time, load_high_low, get_cfg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def plot_equity_and_drawdown_filled(df):
    """
    Function to plot the equity curve and drawdown plot with filled regions using Plotly for interactivity.
    :param df: DataFrame containing the 'datetime', 'capital', and 'drawdown_percentage' columns.
    """
    # Create a subplot with 2 rows for equity curve and drawdown plot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Equity Curve", "Drawdown Plot with Filled Regions"),
        vertical_spacing=0.15,
    )

    # Adding Equity Curve trace to the first row
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["capital"],
            mode="lines",
            name="Equity Curve",
            line=dict(color="blue"),
            hovertemplate="Date: %{x}<br>Capital: %{y:.2f}",
        ),
        row=1,
        col=1,
    )

    # Adding Drawdown trace with filled area to the second row
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["drawdown_percentage"],
            fill="tozeroy",
            mode="lines",
            name="Drawdown (%)",
            line=dict(color="red"),
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%",
        ),
        row=2,
        col=1,
    )

    # Update layout for better readability
    fig.update_layout(
        title_text="Interactive Equity Curve and Drawdown Plot",
        height=600,
        hovermode="x unified",  # Show hover labels across subplots at the same x position
    )

    # Axis titles
    fig.update_yaxes(title_text="Capital", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown Percentage (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date Time", row=2, col=1)

    # Show the interactive plot
    fig.show()


high_csv, low_csv = load_high_low(get_cfg())


def calculate_max_drawdown(df):
    """
    Function to calculate maximum drawdown percentage.
    :param df: DataFrame containing the 'capital' column.
    :return: Maximum drawdown percentage.
    """
    # Step 1: Calculate the cumulative maximum capital
    df["cumulative_max"] = df["capital"].cummax()

    # Step 2: Calculate the drawdown percentage
    df["drawdown_percentage"] = (
        (df["capital"] - df["cumulative_max"]) / df["cumulative_max"] * 100
    )

    # Step 3: Find the maximum drawdown percentage
    max_drawdown_percentage = df["drawdown_percentage"].min()
    avg_drawdown_percentage = df["drawdown_percentage"].mean()

    return max_drawdown_percentage, avg_drawdown_percentage


# def calculate_average_holding_duration(
#     df: pd.DataFrame, date_column: str) -> pd.Timedelta:

#     # Convert the specified date column to datetime
#     # df[date_column] = pd.to_datetime(df[date_column])
#     df[date_column] = df[date_column].map(handle_date_time)

#     holding_duration = []
#     for i in range(0, len(df) - 1, 2):
#         holding_duration.append(df.loc[i + 1, "date_time"] - df.loc[i, "date_time"])

#     print(f"maximium time period idx {np.argmax(holding_duration)*2}")
#     # Calculate the average holding duration
#     average_holding_duration = np.mean(holding_duration)
#     max_holding_duration = np.max(holding_duration)

#     return average_holding_duration, max_holding_duration



def calculate_holding_periods(tradesheet):
    
    # Ensure date_time column is in datetime format
    tradesheet['date_time'] = pd.to_datetime(tradesheet['date_time'])

    # Initialize lists to store holding periods for each trade
    holding_periods = []

    en_d = tradesheet['date_time'].iloc[0]
    ex_d = None

    for i in range(1, len(tradesheet)-1):
        if((tradesheet['order_status'].iloc[i] == "LONG" or tradesheet['order_status'].iloc[i] == "SHORT") and (tradesheet['signal'].iloc[i] == -2 or tradesheet['signal'].iloc[i] == 2)):
            ex_d = tradesheet['date_time'].iloc[i]
            holding_periods.append(ex_d - en_d)
            en_d = tradesheet['date_time'].iloc[i]
        elif(tradesheet['order_status'].iloc[i] == "Squared_Off"):
            ex_d = tradesheet['date_time'].iloc[i]
            holding_periods.append(ex_d - en_d)
            en_d = tradesheet['date_time'].iloc[i+1]
    if(len(holding_periods) == 0):
        return 0, 0
    return np.mean(holding_periods), np.max(holding_periods)



def compute_metrics(
    signals: pd.DataFrame,
    trade_sheet: pd.DataFrame,
    plot: bool = False,
    leverage: int = 1,
    slippage: float = 0.0015,
    capital: float = 1000,
):
    signals["returns"] = 0.0
    signals["pnl"] = np.nan
    total_fee = 0
    signals["capital"] = 1000.0
    initial_capital = capital
    gross_profit = 0
    gross_loss = 0
    status = 0
    entry_price = 0
    trades = 0
    total_long_trades = 0
    total_short_trades = 0

    for i in range(len(signals)):
        # capital=1000
        signal = signals.loc[i, "signals"]
        if status != 0:
            if status == 1:
                if signal == 0:
                    continue
                # square off the current position
                exit_price = signals.loc[i, "close"]
                signals.loc[i, "returns"] = (
                    ((exit_price - entry_price) / entry_price) * status * leverage
                )
                signals.loc[i, "pnl"] = (
                    capital
                    * ((exit_price - entry_price) / entry_price)
                    * status
                    * leverage
                )
                total_fee += capital * slippage
                capital -= capital * slippage
                capital += signals.loc[i, "pnl"]
                signals.loc[i, "capital"] = capital
                if signals.loc[i, "pnl"] > 0:
                    gross_profit += capital - (
                        signals.loc[i - 1, "capital"] if i > 0 else 0
                    )
                if signals.loc[i, "pnl"] < 0:
                    gross_loss += capital - (
                        signals.loc[i - 1, "capital"] if i > 0 else 0
                    )

                if signal == -1:  # status= 0, do nothing after squaring off
                    status = 0
                elif signal == -2:  # status = -1, go short after squaring off
                    status = -1
                    entry_price = signals.loc[i, "close"]
                    total_short_trades += 1
                elif signal == 1:  # status = 1, go long after squaring off
                    status = 1
                    entry_price = signals.loc[i, "close"]
                    total_long_trades += 1
                trades += 1

            elif status == -1:
                if signal == 0:
                    continue

                # square off the short position
                exit_price = signals.loc[i, "close"]
                signals.loc[i, "returns"] = (
                    ((exit_price - entry_price) / entry_price) * status * leverage
                )
                signals.loc[i, "pnl"] = (
                    capital
                    * ((exit_price - entry_price) / entry_price)
                    * status
                    * leverage
                )
                total_fee += capital * slippage
                capital -= capital * slippage
                capital += signals.loc[i, "pnl"]
                signals.loc[i, "capital"] = capital
                if signals.loc[i, "pnl"] > 0:
                    gross_profit += capital - (
                        signals.loc[i - 1, "capital"] if i > 0 else 0
                    )
                if signals.loc[i, "pnl"] < 0:
                    gross_loss += capital - (
                        signals.loc[i - 1, "capital"] if i > 0 else 0
                    )

                if signal == 1:  # status= 0, do nothing after squaring off
                    status = 0
                elif signal == 2:  # status = 1, go long after squaring off
                    status = 1
                    entry_price = signals.loc[i, "close"]
                    total_long_trades += 1
                elif signal == -1:  # status = -1, go short after squaring off
                    status = -1
                    entry_price = signals.loc[i, "close"]
                    total_short_trades += 1

                trades += 1

        elif status == 0:
            if signal == 0:
                continue

            entry_price = signals.loc[i, "close"]
            if signal == 1:
                status = 1
                total_long_trades += 1
            elif signal == -1:
                status = -1
                total_short_trades += 1
            signals.loc[i, "capital"] = capital
    profit_percent = (
        signals[signals["pnl"] > 0]["pnl"].count()
        / signals[signals["pnl"] != np.nan]["pnl"].count()
    )
    # max_drawdown = calculate_max_drawdown(signals['capital'].values)
    max_drawdown, avg_drawdown = calculate_max_drawdown(signals)
    win_rate = (
        signals[(signals["pnl"] > 0) & (pd.notna(signals["pnl"]))]["pnl"].count()
        / len(signals[pd.notna(signals["pnl"])])
    ) * 100
    loss_rate = (
        signals[(signals["pnl"] < 0) & (pd.notna(signals["pnl"]))]["pnl"].count()
        / len(signals[pd.notna(signals["pnl"])])
    ) * 100
    net_profit = signals.loc[len(signals) - 1, "capital"] - initial_capital
    avg_winning_trade = (
        0
        if (signals[signals["pnl"] > 0]["pnl"].count() <= 0)
        else signals[signals["pnl"] > 0]["pnl"].sum()
        / signals[signals["pnl"] > 0]["pnl"].count()
    )
    avg_losing_trade = (
        0
        if (signals[signals["pnl"] < 0]["pnl"].count() <= 0)
        else signals[signals["pnl"] < 0]["pnl"].sum()
        / signals[signals["pnl"] < 0]["pnl"].count()
    )
    buy_and_hold_return = (
        (signals.loc[len(signals) - 1, "close"] - signals.loc[0, "close"])
        / signals.loc[0, "close"]
    ) * initial_capital - initial_capital * slippage
    largest_losing_trade = signals[signals["pnl"] < 0]["pnl"].min()
    largest_winning_trade = signals[signals["pnl"] > 0]["pnl"].max()
    exit_date1 = get_cfg().backtester.exit_date.split(" ")[0]
    entry_date1 = get_cfg().backtester.entry_date.split(" ")[0]
    high_csv["datetime_"] = high_csv["datetime"].apply(lambda x: x.split(" ")[0])
    entry_price = high_csv[high_csv["datetime_"] == entry_date1]["close"].iloc[0] 
    exit_price = high_csv[high_csv["datetime_"] == exit_date1]["close"].iloc[-1]
    benchmark_returns = (
        (exit_price - entry_price) / entry_price
    ) * initial_capital - initial_capital * slippage

    avg_returns = signals[signals["returns"] != 0]["returns"].mean()
    returns_dev = signals[signals["returns"] != 0]["returns"].std()
    sharpe_ratio = (avg_returns / returns_dev) * np.sqrt(365)

    neg_returns = signals[signals["returns"] < 0]["returns"].std()
    sortino_ratio = (avg_returns / neg_returns) * np.sqrt(365)

    max_pnl = signals["pnl"].max()
    min_pnl = signals["pnl"].min()

    avg_holding_duration, maximum_holding_duration = calculate_holding_periods(
        trade_sheet
    )
    min_portfolio_balance = signals["capital"].min()
    max_portfolio_balance = signals["capital"].max()
    final_balance = signals.loc[len(signals) - 1, "capital"]

    # Define the two dates
    date1 = datetime.strptime(signals['datetime'].iloc[0], "%Y-%m-%d %H:%M:%S")
    date2 = datetime.strptime(signals['datetime'].iloc[-1], "%Y-%m-%d %H:%M:%S")

    # Calculate the difference in days and divide by 365
    years_difference = (date2 - date1).days / 365

    cagr = np.power((final_balance/initial_capital), (1/years_difference)) - 1

    metrics_dict = {
        "final_balance": final_balance,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "total_long_trades": total_long_trades,
        "total_short_trades": total_short_trades,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "avg_winning_trade": avg_winning_trade,
        "avg_losing_trade": avg_losing_trade,
        "buy_and_hold_return": buy_and_hold_return,
        "largest_losing_trade": largest_losing_trade,
        "largest_winning_trade": largest_winning_trade,
        "maximum_drawdown": max_drawdown,
        "average_drawdown":avg_drawdown,
        "CAGR": cagr,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "average_holding_duration": avg_holding_duration,
        "maximum_holding_duration": maximum_holding_duration,
        "maximum_pnl": max_pnl,
        "minimum_pnl": min_pnl,
        "min_portfolio_balance": min_portfolio_balance,
        "max_portfolio_balance": max_portfolio_balance,
        "num_of_trades": trades,
        "total_fee": total_fee,
        "profit_percent": profit_percent,
        "benchmark_returns": benchmark_returns,
    }

    metrics_df = pd.Series(metrics_dict)
    if plot:
        plot_equity_and_drawdown_filled(signals)
    return metrics_df
