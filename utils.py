import yaml
from easydict import EasyDict
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CACHE = {}


def get_cfg(file_path: str = "config.yaml") -> EasyDict:
    """Get the config yaml file as an EasyDict object.

    Parameters:
        file_path (str): The path to the config yaml file.
    Returns:
        EasyDict: The yaml file as an EasyDict object.
    """
    if file_path not in CACHE:
        CACHE[file_path] = _load_cfg(file_path)
    if CACHE[file_path] is None:
        raise FileNotFoundError(f"Config file not found at {file_path}")
    return CACHE[file_path]


def _load_cfg(file_path: str = "config.yaml") -> EasyDict | None:
    """Load the config yaml file as an EasyDict object.

    Parameters:
        file_path (str): The path to the config yaml file.
    Returns:
        EasyDict: The yaml file as an EasyDict object.
    """
    with open(file_path, "r") as stream:
        try:
            return EasyDict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            return None


def load_high_low(config: EasyDict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the high and low timeframe dataframes

    Args:
        config (EasyDict): Config object containing the data file paths and strategy parameters

    Returns:
        Tuple (pd.DataFrame, pd.DataFrame): DataFrames containing the high and low timeframe data
    """
    DATA_DIR = Path(config.data.path)

    load_time = perf_counter()
    high_csv = pd.read_csv(DATA_DIR / config.data.files[config.backtester.high_time])
    # print(f"High CSV loaded in {perf_counter() - load_time:.4f} seconds")
    load_time = perf_counter()
    low_csv = pd.read_csv(DATA_DIR / config.data.files[config.backtester.low_time])
    low_csv["tp"] = config.backtester.tp
    low_csv["sl"] = config.backtester.sl
    # print(f"Low CSV loaded in {perf_counter() - load_time:.4f} seconds")
    return high_csv, low_csv


def handle_date_time(date_time: str) -> datetime:
    try:
        # Try parsing with microseconds
        date_object = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        date_object = datetime.strptime(date_time + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    return date_object


def trade_log(
    date_time: str,
    executed_price: float,
    capital: float,
    signal: int,
    status: int,
    order_type: str,
    p: float,
    stop_loss: float,
    trade_sheet: pd.DataFrame,
    margin_hit: bool = False,
):
    """Log the trade details

    Args:
        date_time (str): Timestamp of the trade
        executed_price (float): Price at which the trade was executed
        capital (float): Capital at the time of trade
        signal (int): Signal for the trade
        status (int): Status of the trade
        order_type (str): Type of the order
        p (float): Profit/Loss percentage
        stop_loss (float): Stop loss price
        trade_sheet (pd.DataFrame): DataFrame to log the trade details
    """
    if status == 1:
        order_status = "LONG"
    elif status == -1:
        order_status = "SHORT"
    elif status == 0:
        order_status = "Squared_Off"

    log = {
        "date_time": date_time,
        "executed_price": executed_price,
        "capital": capital,
        "signal": signal,
        "order_status": order_status,
        "order_type": order_type,
        "profit_loss%": p,
        "stop_loss": stop_loss,
        "margin_hit": margin_hit,
    }
    trade_sheet.loc[len(trade_sheet)] = log


def adjust(low_pointer: int, high_pointer: int, low_csv: int, high_csv: int) -> int:
    """Adjust the low pointer to match the high pointer

    Args:
        low_pointer (int): pointer to the low timeframe data
        high_pointer (int): pointer to the high timeframe data
        low_csv (int): low timeframe data
        high_csv (int): high timeframe data

    Returns:
        int: Adjusted low pointer
    """
    while handle_date_time(high_csv["datetime"].iloc[high_pointer]) > handle_date_time(
        low_csv["datetime"].iloc[low_pointer]
    ):
        low_pointer += 1
    return low_pointer


def generate_csv(
    ptr: int,
    signal: int,
    low_csv: pd.DataFrame,
    signal_csv: pd.DataFrame,
    signal_type: str,
):
    """Generate the Signal CSV

    Args:
        ptr (int): Pointer to the data, either high or low
        signal (int): Signal for the trade
        low_csv (pd.DataFrame): low timeframe data
        signal_csv (pd.DataFrame): DataFrame to log the signal details
    """
    data = low_csv.iloc[ptr]
    log = {
        "datetime": data["datetime"],
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
        "signals": signal,
        "signal_type": signal_type,
    }
    signal_csv.loc[len(signal_csv)] = log


def tpsl(
    low_pointer: int,
    high_pointer: int,
    low_csv: pd.DataFrame,
    high_csv: pd.DataFrame,
    glob: EasyDict,
    trailing=False,
) -> Tuple[int, int]:
    """Function to check the target price and stop loss conditions

    Args:
        low_pointer (int): Pointer to the low timeframe data
        low_csv (pd.DataFrame): DataFrame containing the low timeframe data
        margin (float): Margin for the trade
        leverage (int): Leverage for the trade
        glob (EasyDict): Global variables containing the trade details
        trailing (bool, optional): Is the Stop loss trailing. Defaults to False.

    Returns:
        Tuple: (int, int)
        - 1 if the condition is satisfied, 0 otherwise
        - Index at which the condition is satisfied
    """
    if high_pointer + 1 == len(high_csv):
        tpsl_check_end_time = handle_date_time("3030-01-01")
    else:
        tpsl_check_end_time = handle_date_time(
            high_csv["datetime"].iloc[high_pointer + 1]
        )
    index = int(low_pointer)
    # glob.tp = low_csv["tp"].iloc[index]
    # glob.sl = low_csv["sl"].iloc[index]

    while index < len(low_csv) and tpsl_check_end_time > handle_date_time(
        low_csv["datetime"].iloc[index]
    ):
        Close = low_csv["close"].iloc[index]
        if glob.status == 1:
            glob.trailing_price = max(glob.trailing_price, Close)
        if glob.status == -1:
            glob.trailing_price = min(glob.trailing_price, Close)
        target_price = glob.entry_price + glob.status * glob.entry_price * glob.tp
        stop_loss = glob.entry_price - glob.status * glob.entry_price * glob.sl
        trailing_stop_loss = (
            glob.trailing_price - glob.status * glob.trailing_price * glob.sl
        )
        if glob.status == 1 and (
            target_price <= Close
            or stop_loss >= Close
            or (trailing and trailing_stop_loss >= Close)
        ):
            date_time = low_csv["datetime"].iloc[index]
            # print("TPSL hit at ", date_time)
            return 1, index
        if glob.status == -1 and (
            target_price >= Close
            or stop_loss <= Close
            or (trailing and trailing_stop_loss <= Close)
        ):
            date_time = low_csv["datetime"].iloc[index]
            # print("TPSL hit at ", date_time)
            return 1, index
        index += 1
    # print("Ending TPSL check at ", low_csv["datetime"].iloc[index-1])
    return 0, 0


def to_minutes(time: str) -> int:
    """Convert the time to minutes

    Args:
        time (str): Time in 1d, 1h, 1m format

    Returns:
        int: Time in minutes
    """
    unit = time[-1]
    value = int(time[:-1])
    match unit:
        case "d":
            return value * 24 * 60
        case "h":
            return value * 60
        case "m":
            return value
        case "w":
            return value * 7 * 24 * 60
        case _:
            raise ValueError(f"Invalid unit: {unit}")


def check_for_presence_in_low_csv(
    low_pointer: int,
    low_csv: pd.DataFrame,
    current_date: str,
    future_time_diff: int = 15,
):
    low_time = (
        handle_date_time(low_csv.loc[1, "datetime"])
        - handle_date_time(low_csv.loc[0, "datetime"])
    ).seconds / 60
    current_date = handle_date_time(current_date)
    while (
        low_pointer < len(low_csv)
        and handle_date_time(low_csv.loc[low_pointer, "datetime"]) < current_date
    ):
        low_pointer += 1
    prev_pointer = low_pointer - 1
    # print(f"{current_date}-->{low_csv.loc[prev_pointer,'datetime']}")
    if (
        prev_pointer >= 0
        and (
            current_date - handle_date_time(low_csv.loc[prev_pointer, "datetime"])
        ).seconds
        / 60
        <= low_time
    ):
        return True, prev_pointer
    elif (
        low_pointer < len(low_csv)
        and (
            handle_date_time(low_csv.loc[low_pointer, "datetime"]) - current_date
        ).seconds
        / 60
        <= future_time_diff
    ):
        return True, low_pointer  # Checking if exists in next 15 minutes
    return False, None


def plot_trade_sheet(csv: pd.DataFrame, low_csv: pd.DataFrame):
    low_csv["datetime"] = pd.to_datetime(low_csv["datetime"])
    csv["date_time"] = pd.to_datetime(csv["date_time"])
    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor("lightgrey")
    plt.gcf().set_facecolor("lightgrey")
    (line,) = plt.plot(
        low_csv["datetime"],
        low_csv["close"],
        label="Closing Price",
        color="white",
        linewidth=2,
    )

    for i in range(len(csv)):
        signal = csv.loc[i, "signal"]
        signal_time = csv.loc[i, "date_time"]

        matching_row = low_csv[low_csv["datetime"] == signal_time]
        if not matching_row.empty:
            close_price = matching_row["close"].values[0]
            if signal == 1:
                plt.annotate(
                    "↑",
                    xy=(signal_time, close_price),
                    color="lime",
                    fontsize=20,
                    ha="center",
                )
            elif signal == -1:
                plt.annotate(
                    "↓",
                    xy=(signal_time, close_price),
                    color="orangered",
                    fontsize=20,
                    ha="center",
                )
            elif signal == 2:
                plt.annotate(
                    "↑",
                    xy=(signal_time, close_price),
                    color="chartreuse",
                    fontsize=20,
                    ha="center",
                )
            elif signal == -2:
                plt.annotate(
                    "↓",
                    xy=(signal_time, close_price),
                    color="tomato",
                    fontsize=20,
                    ha="center",
                )
        else:
            print("did not match. Something wrong with signal csv")
            print(f"{signal_time}")
    green_up = mpatches.Patch(color="lime", label="Long Entry (+1)")
    red_down = mpatches.Patch(color="orangered", label="Short Entry (-1)")
    dark_green_up = mpatches.Patch(color="chartreuse", label="Sq Off + Long Entry")
    dark_red_down = mpatches.Patch(color="tomato", label="Sq Off + Short Entry")
    plt.legend(handles=[green_up, red_down, dark_green_up, dark_red_down])
    plt.title("Closing Price with Trade Signals")
    plt.xlabel("Date and Time")
    plt.ylabel("Closing Price")
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()


def calculate_benchmark_percentage(high_csv: pd.DataFrame):
    return (
        high_csv.loc[len(high_csv) - 1, "close"] - high_csv.loc[0, "open"]
    ) / high_csv.loc[0, "open"]


def check_margin(capital, margin_price, margin, leverage):
    if capital < margin_price:
        # print(
        #     f"Capital: {capital}, Hit Margin Price: {margin_price}, New Margin Price: {capital * leverage * margin}"
        # )
        margin_price = capital * leverage * margin
        return margin_price, 1
    else:
        margin_price = max(margin_price, capital * leverage * margin)
        return margin_price, 0
