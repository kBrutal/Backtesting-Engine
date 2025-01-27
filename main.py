import pandas as pd
from utils import (
    load_high_low,
    get_cfg,
    plot_trade_sheet,
    handle_date_time,
    calculate_benchmark_percentage,
)
from easydict import EasyDict
from backtesting_ps_code import generate_signals, check_signal_file
from strategy import (
    EMAStrategy,
    ButterChebyStrategy,
    Neelabh_Strategy,
    gaussian_lsma1,
    Gaussian_Kalman,
    VidyaCrossoverStrategy,
    Gaussian_LSMA_Hawkes_ADX_RSI,
    My_Strategy_8,
    My_Strategy_9

)
from metrics import compute_metrics
from datetime import datetime, timedelta
from pprint import pprint
from chart import plot_btc_with_trades
from dateutil.relativedelta import relativedelta


def quarter_wise_result(high_csv, low_csv, entry_date, exit_date):
    quarter_wise_metrics = pd.DataFrame(
        columns=[
            "Initial_balance",
            "Final_balance",
            "Profit(%)",
            "Benchmark(%)",
            "Benchmark_Beaten",
            "gross_profit",
            "From",
            "To",
            "Total_trades",
            "Long_trades",
            "Short_trades",
            "Win_Rate",
        ]
    )

    start_quarter_date = entry_date
    end_quarter_date = exit_date

    # print("Entry Date: ", entry_date)
    # print("Exit Date: ", exit_date)

    cnt = 0
    old_status = 0
    while start_quarter_date < exit_date:
        cnt += 1
        # end_quarter_date = start_quarter_date + timedelta(days=90)
        end_quarter_date = (start_quarter_date + relativedelta(months=3)).replace(day=1) - relativedelta(days=1)
        end_quarter_date = end_quarter_date.replace(hour=23, minute=59, second=59)
        if end_quarter_date > exit_date:
            end_quarter_date = exit_date

        # start_quarter_date_t = start_quarter_date - timedelta(seconds=1)
        # end_quarter_date_t = end_quarter_date + timedelta(seconds=1)

        # print("start quarter date temp ",start_quarter_date," end quarter date temp: ",end_quarter_date)

        quarter_low_csv = low_csv[
            ((low_csv["datetime"].apply(handle_date_time)) >= start_quarter_date)
            & ((low_csv["datetime"].apply(handle_date_time)) <= (end_quarter_date))
        ]
        quarter_high_csv = high_csv[
            ((high_csv["datetime"].apply(handle_date_time)) >= start_quarter_date)
            & ((high_csv["datetime"].apply(handle_date_time)) <= end_quarter_date)
        ]
        quarter_low_csv = quarter_low_csv.reset_index(drop=True)
        quarter_high_csv = quarter_high_csv.reset_index(drop=True)
        # quarter_high_csv = high_csv
        
        quarter_signal_csv = pd.DataFrame(
            columns=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signals",
                "signal_type",
            ],
        )

        quarter_trade_sheet = pd.DataFrame(
            columns=[
                "date_time",
                "executed_price",
                "capital",
                "signal",
                "order_status",
                "order_type",
                "profit_loss%",
                "stop_loss",
            ]
        )
        old_status=0
        Quarter_GLOB = EasyDict(
            tp=get_cfg().backtester.tp,  # Target Price Percentage
            sl=get_cfg().backtester.sl,  # Stop Loss Percentage
            entry_price=1,
            trailing_price=0,  # Trailing Price of the current position of the trade (used for the calculation of trailing stop loss)
            date_time=high_csv.loc[
                0, "datetime"
            ],  # Intialzing the start date of the backtesting
            status=old_status,  # Initialing the status as 0 (no position currently)
            total_fee=0,
            trades=0,
        )

        strat = My_Strategy_9(
            quarter_high_csv, quarter_low_csv, get_cfg(), Quarter_GLOB
        )

        generate_signals(
            strat,
            Quarter_GLOB,
            quarter_high_csv,
            quarter_low_csv,
            quarter_trade_sheet,
            quarter_signal_csv,
            margin=get_cfg().backtester.margin,
            leverage=get_cfg().backtester.leverage,
            trailing=get_cfg().backtester.trailing,
            slippage=get_cfg().backtester.slippage,
            capital=get_cfg().backtester.capital,
            entry_date=start_quarter_date,  # Add start data
            exit_date=end_quarter_date,  # Add end data
        )
        # print(
        #     start_quarter_date,
        #     end_quarter_date,
        #     len(quarter_signal_csv),
        #     len(quarter_high_csv),
        #     len(quarter_low_csv),
        #     len(quarter_trade_sheet),
        # )

        if len(quarter_signal_csv) == 0:
            # start_quarter_date = end_quarter_date
            start_quarter_date = end_quarter_date + timedelta(seconds=1)
            old_status = Quarter_GLOB.status
            continue

        quarter_signal_csv = quarter_signal_csv.drop(columns=["signal_type"])

        print("Quarter Metrics: ")
        quarter_metrics = metrics(quarter_signal_csv, quarter_trade_sheet)

        benchmark_return_percentage = calculate_benchmark_percentage(quarter_high_csv)

        # print("high csv start ",high_csv.loc[0,"datetime"]," end ",high_csv.loc[len(high_csv)-1,"datetime"])
        print("Start Date: ",quarter_high_csv.loc[0,"datetime"],"Expected Date: ",start_quarter_date)
        # print("Opening Price: ",quarter_high_csv.loc[0,"open"])
        print("End Date: ",quarter_high_csv.loc[len(quarter_high_csv)-1,"datetime"],"Expected End Date: ",end_quarter_date)
        # print("Closing Price: ",quarter_high_csv.loc[len(quarter_high_csv)-1,"close"])

        quarter_result_row = {
            "Initial_balance": get_cfg().backtester.capital,
            "Final_balance": quarter_metrics["final_balance"],
            "Profit(%)": quarter_metrics["net_profit"]
            / get_cfg().backtester.capital
            * 100,
            "Benchmark(%)": benchmark_return_percentage * 100,
            "Benchmark_Beaten": (
                "Yes"
                if (
                    quarter_metrics["net_profit"]
                    > benchmark_return_percentage * get_cfg().backtester.capital
                )
                else "No"
            ),
            "gross_profit": quarter_metrics["gross_profit"],
            "From": start_quarter_date,
            "To": end_quarter_date,
            "Total_trades": quarter_metrics["num_of_trades"],
            "Long_trades": quarter_metrics["total_long_trades"],
            "Short_trades": quarter_metrics["total_short_trades"],
            "Win_Rate": quarter_metrics["win_rate"],
        }

        print(quarter_result_row)

        quarter_wise_metrics.loc[len(quarter_wise_metrics)] = quarter_result_row

        start_quarter_date = end_quarter_date + timedelta(seconds=1)

        old_status = Quarter_GLOB.status

        print("!!!!!!!!!!! old status: ", old_status)

    quarter_wise_metrics.to_csv("quarter_wise_metrics.csv", index=False)

    print("Total Quarters: ", cnt)

    return quarter_wise_metrics


def main():
    trade_sheet = pd.DataFrame(
        columns=[
            "date_time",
            "executed_price",
            "capital",
            "signal",
            "order_status",
            "order_type",
            "profit_loss%",
            "stop_loss",
            "margin_hit",
        ]
    )
    signal_csv = pd.DataFrame(
        columns=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signals",
            "signal_type",
        ],
    )
    entry_date = get_cfg().backtester.entry_date
    exit_date = get_cfg().backtester.exit_date
    high_csv, low_csv = load_high_low(get_cfg())
    low_csv = low_csv[
        (low_csv["datetime"] >= entry_date) & (low_csv["datetime"] <= exit_date)
    ]
    low_csv = low_csv.reset_index(drop=True)
    entry_date = datetime.strptime(entry_date, "%Y-%m-%d %H:%M:%S")
    exit_date = datetime.strptime(exit_date, "%Y-%m-%d %H:%M:%S")

    GLOB = EasyDict(
        tp=get_cfg().backtester.tp,  # Target Price Percentage
        sl=get_cfg().backtester.sl,  # Stop Loss Percentage
        entry_price=1,
        trailing_price=0,  # Trailing Price of the current position of the trade (used for the calculation of trailing stop loss)
        date_time=high_csv.loc[
            0, "datetime"
        ],  # Intialzing the start date of the backtesting
        status=0,  # Initialing the status as 0 (no position currently)
        total_fee=0,
        trades=0,
    )

    strat = gaussian_lsma1(high_csv, low_csv, get_cfg(), GLOB)

    generate_signals(
        strat,
        GLOB,
        high_csv,
        low_csv,
        trade_sheet,
        signal_csv,
        margin=get_cfg().backtester.margin,
        leverage=get_cfg().backtester.leverage,
        trailing=get_cfg().backtester.trailing,
        slippage=get_cfg().backtester.slippage,
        capital=get_cfg().backtester.capital,
        entry_date=datetime(
            entry_date.year, entry_date.month, entry_date.day
        ),  # Add start data
        exit_date=datetime(
            exit_date.year, exit_date.month, exit_date.day
        ),  # Add end data
    )
    signal_csv = signal_csv.drop(columns=["signal_type"])
    trade_sheet.to_csv("trade_sheet.csv", index=False)
    signal_csv.to_csv("signal_csv.csv", index=False)

    if get_cfg().backtester.quarter_wise_result:
        quarter_wise_result(high_csv, low_csv, entry_date, exit_date)

    if get_cfg().backtester.plot_trade_sheet:
        plot_trade_sheet(trade_sheet, low_csv)

    return signal_csv, trade_sheet


def metrics(signal_csv: pd.DataFrame, trade_sheet: pd.DataFrame):
    metrics_result = compute_metrics(
        signal_csv,
        trade_sheet,
        get_cfg().backtester.plots.show,
        get_cfg().backtester.leverage,
        get_cfg().backtester.slippage,
        get_cfg().backtester.capital,
    )
    pprint(metrics_result)
    output_string = f"Strategy from {get_cfg().backtester.entry_date} to {get_cfg().backtester.exit_date} with tp={get_cfg().backtester.tp} and sl={get_cfg().backtester.sl}"
    output_string += "Metrics Results:\n"
    output_string += str(metrics_result)
    with open("metrics_output.txt", "a") as file:
        file.write(output_string + "\n\n\n")
    check_signal_file(signal_csv, get_cfg())

    return metrics_result


if __name__ == "__main__":
    signal_csv,trade_sheet = main()
    if get_cfg().backtester.print_metrics:
        metrics(signal_csv, trade_sheet)
        plot_btc_with_trades()
