import pandas as pd
from easydict import EasyDict
from datetime import datetime
from utils import (
    get_cfg,
    load_high_low,
    adjust,
    generate_csv,
    trade_log,
    tpsl,
    check_for_presence_in_low_csv,
    check_margin,
)
from strategy import BaseStrategy


def generate_signals(
    strategy: BaseStrategy,
    glob: EasyDict,
    high_csv: pd.DataFrame,
    low_csv: pd.DataFrame,
    trade_sheet: pd.DataFrame,
    signal_csv: pd.DataFrame,
    margin: float = 0.02,
    leverage: int = 1,
    trailing: bool = False,
    slippage: float = 0.0015,
    capital: float = 1000,
    entry_date: datetime = None,
    exit_date: datetime = None,
):
    """Signal Generation for the backtesting or live trading

    Args:
        strategy (BaseStrategy): Strategy to be used for generating the signals
        glob (EasyDict): Global variables to store the status of the trade
        high_csv (pd.DataFrame): High timeframe data
        low_csv (pd.DataFrame): Low timeframe data
        trade_sheet (pd.DataFrame): Trade Book to log the trades
        signal_csv (pd.DataFrame): Signal file to log the generated signals
        margin (float, optional): Margin for the trade. Defaults to 0.02.
        leverage (int, optional): Leverage for the trade. Defaults to 1.
        trailing (bool, optional): Trailing stop loss. Defaults to False.
        slippage (float, optional): Slippage for the trade. Defaults to 0.0015.
        capital (float, optional): Initial Capital for the trade. Defaults to 1000.
        exit_date (datetime, optional): Exit date for the trade. Defaults to None.
        entry_date (datetime, optional): Entry date for the trade. Defaults to None.
    """
    entry_index = 0
    margin_price = capital * leverage * margin
    exit_index = len(high_csv) - 1
    if entry_date:
        entry_index = high_csv[
            pd.to_datetime(high_csv["datetime"]) >= entry_date
        ].index[0]
    if exit_date:
        if pd.to_datetime(high_csv["datetime"].iloc[-1]) <= exit_date:
            exit_index = len(high_csv) - 1
        else:
            exit_index = (
                high_csv[pd.to_datetime(high_csv["datetime"]) > exit_date].index[0] - 1
            )
    print("Backtesting Started")
    low_pointer = 1
    pnl = 0
    margin_hit = 0
    for i in range(entry_index, exit_index):
        low_pointer = adjust(low_pointer, i, low_csv, high_csv)
        if glob.status != 0:
            hit, ind = tpsl(low_pointer, i, low_csv, high_csv, glob, trailing)
            date_time = low_csv["datetime"].iloc[ind]
            exit_price = low_csv["close"].iloc[ind]
            if hit > 0:
                pnl = (
                    capital
                    * ((exit_price - glob.entry_price) / glob.entry_price)
                    * glob.status
                    * leverage
                )
                p = (
                    ((exit_price - glob.entry_price) / glob.entry_price)
                    * glob.status
                    * leverage
                )
                glob.total_fee += capital * slippage
                capital -= capital * slippage
                capital += pnl
                margin_price, margin_hit = check_margin(
                    capital, margin_price, margin, leverage
                )
                signal = -1 * glob.status
                glob.status = 0
                if pnl > 0:
                    signal_type = "TP"
                else:
                    signal_type = "SL"
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    signal_type,
                    p,
                    0,
                    trade_sheet,
                    margin_hit,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "tpsl")
                glob.trades += 1
                continue
        if i == exit_index - 1:
            continue
        present, ind = check_for_presence_in_low_csv(
            low_pointer, low_csv, high_csv.loc[i + 1, "datetime"]
        )
        if present == False:
            continue
        exit_price = low_csv["close"].iloc[ind]
        date_time = low_csv["datetime"].iloc[ind]
        pnl = (
            capital
            * ((exit_price - glob.entry_price) / glob.entry_price)
            * glob.status
            * leverage
        )
        p = (
            ((exit_price - glob.entry_price) / glob.entry_price)
            * glob.status
            * leverage
        )
        if glob.status == 1:
            if strategy.check_short_entry(i):
                glob.total_fee += capital * slippage
                capital -= capital * slippage
                capital += pnl
                glob.status = -1
                signal = -2
                stop_loss = glob.entry_price - glob.entry_price * glob.status * glob.sl
                margin_price, margin_hit = check_margin(
                    capital, margin_price, margin, leverage
                )
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    p,
                    stop_loss,
                    trade_sheet,
                    margin_hit,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "market")
                glob.trades += 1
            if strategy.check_long_exit(i):
                glob.total_fee += capital * slippage
                capital -= capital * slippage
                capital += pnl
                glob.status = 0
                signal = -1
                margin_price, margin_hit = check_margin(
                    capital, margin_price, margin, leverage
                )
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    p,
                    0,
                    trade_sheet,
                    margin_hit,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "market")
                glob.trades += 1
            margin_price = max(margin_price, capital * leverage * margin)

        elif glob.status == -1:
            if strategy.check_long_entry(i):
                glob.total_fee += capital * slippage
                capital -= capital * slippage
                capital += pnl
                glob.status = 1
                signal = 2
                stop_loss = glob.entry_price - glob.entry_price * glob.status * glob.sl
                margin_price, margin_hit = check_margin(
                    capital, margin_price, margin, leverage
                )
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    p,
                    stop_loss,
                    trade_sheet,
                    margin_hit,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "market")
                glob.trades += 1
            if strategy.check_short_exit(i):
                glob.total_fee += capital * slippage
                capital -= capital * slippage
                capital += pnl
                glob.status = 0
                signal = 1
                margin_price, margin_hit = check_margin(
                    capital, margin_price, margin, leverage
                )
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    p,
                    0,
                    trade_sheet,
                    margin_hit,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "market")
                glob.trades += 1
            margin_price = max(margin_price, capital * leverage * margin)

        elif glob.status == 0:
            if strategy.check_long_entry(i):
                glob.status = 1
                signal = 1
                stop_loss = glob.entry_price - glob.entry_price * glob.status * glob.sl
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    0,
                    stop_loss,
                    trade_sheet,
                )
                # print("=>Long at ",date_time)
                generate_csv(ind, signal, low_csv, signal_csv, "market")

            elif strategy.check_short_entry(i):
                glob.status = -1
                signal = -1
                stop_loss = glob.entry_price - glob.entry_price * glob.status * glob.sl
                trade_log(
                    date_time,
                    exit_price,
                    capital,
                    signal,
                    glob.status,
                    "Market",
                    0,
                    stop_loss,
                    trade_sheet,
                )
                generate_csv(ind, signal, low_csv, signal_csv, "market")

    if glob.status != 0:
        present, ind = check_for_presence_in_low_csv(
            low_pointer, low_csv, high_csv.loc[i + 1, "datetime"]
        )
        if present == False:
            ind = len(low_csv) - 1
        exit_price = low_csv.loc[ind, "close"]
        date_time = low_csv.loc[ind, "datetime"]
        pnl = (
            capital
            * ((exit_price - glob.entry_price) / glob.entry_price)
            * glob.status
            * leverage
        )
        glob.total_fee += capital * slippage
        capital -= capital * slippage
        capital += pnl
        p = (
            ((exit_price - glob.entry_price) / glob.entry_price)
            * glob.status
            * leverage
        )
        if capital < margin_price:
            margin_hit = 1
            print(margin_price, capital)
            margin_price = capital * leverage * margin
        else:
            margin_hit = 0
            margin_price = max(margin_price, capital * leverage * margin)
        signal = -1 * glob.status
        # if not get_cfg().backtester.quarter_wise_result:
        #     glob.status = 0
        glob.status = 0
        trade_log(
            date_time,
            exit_price,
            capital,
            signal,
            glob.status,
            "Market",
            p,
            0,
            trade_sheet,
            margin_hit,
        )
        margin_price = max(margin_price, capital * leverage * margin)
        generate_csv(ind, signal, low_csv, signal_csv, "market")
        glob.trades += 1
        print("Backtesting ended successfully")


def check_signal_file(signal_csv: pd.DataFrame, config: EasyDict):
    """Check the signal file for the backtesting

    Args:
        signal_csv (pd.DataFrame): Signal file generated for the backtesting
        config (EasyDict): Configuration for the backtesting
    """
    capital = config.backtester.capital
    pnl = 0
    transaction_cost = 0
    slippage = config.backtester.slippage
    status = 0
    entry_price = 0
    total_fee = 0
    leverage = config.backtester.leverage
    for i in range(len(signal_csv)):
        signal = signal_csv["signals"].iloc[i]
        exit_price = signal_csv["close"].iloc[i]
        if status == 0:
            if signal == 1:
                status = 1
                transaction_cost = capital * slippage
                entry_price = exit_price
            elif signal == -1:
                status = -1
                transaction_cost = capital * slippage
                entry_price = exit_price
        elif status == 1:
            if signal == -2:
                pnl = (
                    capital
                    * (exit_price - entry_price)
                    / entry_price
                    * status
                    * leverage
                )
                capital += pnl
                capital -= transaction_cost
                total_fee += transaction_cost
                transaction_cost = capital * slippage
                entry_price = exit_price
                status = -1
            elif signal == -1:
                pnl = (
                    capital
                    * (exit_price - entry_price)
                    / entry_price
                    * status
                    * leverage
                )
                capital += pnl
                capital -= transaction_cost
                total_fee += transaction_cost
                transaction_cost = 0
                status = 0
        elif status == -1:
            if signal == 2:
                pnl = (
                    capital
                    * (exit_price - entry_price)
                    / entry_price
                    * status
                    * leverage
                )
                capital += pnl
                capital -= transaction_cost
                total_fee += transaction_cost
                transaction_cost = capital * slippage
                entry_price = exit_price
                status = 1
            elif signal == 1:
                pnl = (
                    capital
                    * (exit_price - entry_price)
                    / entry_price
                    * status
                    * leverage
                )
                capital += pnl
                capital -= transaction_cost
                total_fee += transaction_cost
                transaction_cost = 0
                status = 0
    # print(f'[Check Signal File] Final Capital: {capital}')
    # print(f'[Check Signal File] Total Fee: {total_fee}')
