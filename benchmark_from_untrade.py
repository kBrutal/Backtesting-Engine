import pandas as pd
result = [
    {
      "index": 1,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-13.155600",
      "Final Balance": "1719.976047",
      "From": "2020-01-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "18",
      "Maximum Adverse Excursion": "4.204533",
      "Profit(%)": "71.997605",
      "Short Trades": "15",
      "To": "2020-03-31 23:59:59",
      "Total Trades": "33",
      "Win Rate": "54.545455"
    },
    {
      "index": 2,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "37.602292",
      "Final Balance": "950.834129",
      "From": "2020-04-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "24",
      "Maximum Adverse Excursion": "14.734867",
      "Profit(%)": "-4.916587",
      "Short Trades": "20",
      "To": "2020-06-30 23:59:59",
      "Total Trades": "44",
      "Win Rate": "38.636364"
    },
    {
      "index": 3,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "18.288523",
      "Final Balance": "1137.070898",
      "From": "2020-07-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "20",
      "Maximum Adverse Excursion": "5.718800",
      "Profit(%)": "13.707090",
      "Short Trades": "20",
      "To": "2020-09-30 23:59:59",
      "Total Trades": "40",
      "Win Rate": "40.000000"
    },
    {
      "index": 4,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "170.314976",
      "Final Balance": "1754.678322",
      "From": "2020-10-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "20",
      "Maximum Adverse Excursion": "5.287150",
      "Profit(%)": "75.467832",
      "Short Trades": "16",
      "To": "2020-12-31 23:59:59",
      "Total Trades": "36",
      "Win Rate": "33.333333"
    },
    {
      "index": 5,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "69.951121",
      "Final Balance": "2423.404660",
      "From": "2021-01-01 00:00:00",
    
  "Initial Balance": "1000.000000",
      "Long Trades": "22",
      "Maximum Adverse Excursion": "6.653134",
      "Profit(%)": "142.340466",
      "Short Trades": "10",
      "To": "2021-03-31 23:59:59",
      "Total Trades": "32",
      "Win Rate": "43.750000"
    },
    {
      "index": 6,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-41.189006",
      "Final Balance": "1032.314574",
      "From": "2021-04-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "20",
      "Maximum Adverse Excursion": "6.687764",
      "Profit(%)": "3.231457",
      "Short Trades": "30",
      "To": "2021-06-30 23:59:59",
      "Total Trades": "50",
      "Win Rate": "50.000000"
    },
    {
      "index": 7,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "27.687261",
      "Final Balance": "1066.923953",
      "From": "2021-07-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "19",
      "Maximum Adverse Excursion": "6.151425",
      "Profit(%)": "6.692395",

      "Short Trades": "23",
      "To": "2021-09-30 23:59:59",
      "Total Trades": "42",
      "Win Rate": "42.857143"
    },
    {
      "index": 8,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-5.700817",
      "Final Balance": "1113.036001",
      "From": "2021-10-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "25",
      "Maximum Adverse Excursion": "6.242896",
      "Profit(%)": "11.303600",
      "Short Trades": "30",
      "To": "2021-12-31 23:59:59",
      "Total Trades": "55",
      "Win Rate": "41.818182"
    },
    {
      "index": 9,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-2.381431",
      "Final Balance": "1284.949592",
      "From": "2022-01-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "19",
      "Maximum Adverse Excursion": "6.597981",
      "Profit(%)": "28.494959",
      "Short Trades": "24",
      "To": "2022-03-31 23:59:59",
      "Total Trades": "43",
      "Win Rate": "60.465116"
    },
    {
     
 "index": 10,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-58.250132",
      "Final Balance": "1124.705254",
      "From": "2022-04-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "13",
      "Maximum Adverse Excursion": "6.072776",
      "Profit(%)": "12.470525",
      "Short Trades": "25",
      "To": "2022-06-30 23:59:59",
      "Total Trades": "38",
      "Win Rate": "52.631579"
    },
    {
      "index": 11,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "0.902503",
      "Final Balance": "948.815542",
      "From": "2022-07-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "26",
      "Maximum Adverse Excursion": "6.973894",
      "Profit(%)": "-5.118446",
      "Short Trades": "26",
      "To": "2022-09-30 23:59:59",
      "Total Trades": "52",
      "Win Rate": "44.230769"
    },
    {
      "index": 12,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-14.121618",
      "Final Balance": "1016.781953",
      "From": "2022-10-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "12",
      "Maximum Adverse Excursion": "3.720335",
      "Profit(%)": "1.678195",
      "Short Trades": "22",
      "To": "2022-12-31 23:59:59",
      "Total Trades": "34",
      "Win Rate": "44.117647"
    },
    {
      "index": 13,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "69.012037",
      "Final Balance": "1185.086440",
      "From": "2023-01-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "21",
      "Maximum Adverse Excursion": "5.017109",
      "Profit(%)": "18.508644",
      "Short Trades": "20",
      "To": "2023-03-31 23:59:59",
      "Total Trades": "41",
      "Win Rate": "36.585366"
    },
    {
      "index": 14,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "8.788994",
      "Final Balance": "1109.531525",
      "From": "2023-04-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "24",
      "Maximum Adverse Excursion": "6.491005",
      "Profit(%)": "10.953153",
      "Short Trades": "23",
      "To": "2023-06-30 23:59:59",
      "Total Trades": "47",
      "Win Rate": "44.680851"
    },
    {
      "index": 15,
      "Benchmark Beaten?": "Yes",
      "Benchmark(%)": "-14.055013",
      "Final Balance": "1006.553597",
      "From": "2023-07-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "21",
      "Maximum Adverse Excursion": "1.592828",
      "Profit(%)": "0.655360",
      "Short Trades": "29",
      "To": "2023-09-30 23:59:59",
      "Total Trades": "50",
      "Win Rate": "46.000000"
    },
    {
      "index": 16,
      "Benchmark Beaten?": "No",
      "Benchmark(%)": "51.138501",
      "Final Balance": "1153.093794",
      "From": "2023-10-01 00:00:00",
      "Initial Balance": "1000.000000",
      "Long Trades": "23",
      "Maximum Adverse Excursion": "9.881019",
      "Profit(%)": "15.309379",
      "Short Trades": "13",
      "To": "2023-12-31 19:57:00",
      "Total Trades": "36",
      "Win Rate": "41.666667"
    }
  ]




def main():
    df = pd.DataFrame(result)
    df.drop(columns=['index'], inplace=True)

    return df


if __name__ == '__main__':
    df = main()
    df1 = pd.read_csv("quarter_wise_metrics.csv")

    leng = min(len(df), len(df1))
    df_compare = pd.DataFrame(columns=["From_us","From_untrade","To_From","To_untrade","Benchmark(%)_us","Benchmark(%)_untrade","Benchmark Beaten?_us","Benchmark Beaten?_untrade","Final Balance_us","Final Balance_untrade"])
    for i in range(leng):
        temp_dict = {}
        temp_dict["From_us"] = df1.loc[i, "From"]
        temp_dict["From_untrade"] = df.loc[i, "From"]
        temp_dict["To_From"] = df1.loc[i, "To"]
        temp_dict["To_untrade"] = df.loc[i, "To"]
        temp_dict["Benchmark(%)_us"] = df1.loc[i, "Benchmark(%)"]
        temp_dict["Benchmark(%)_untrade"] = df.loc[i, "Benchmark(%)"]
        temp_dict["Benchmark Beaten?_us"] = df1.loc[i, "Benchmark_Beaten"]
        temp_dict["Benchmark Beaten?_untrade"] = df.loc[i, "Benchmark Beaten?"]
        temp_dict["Final Balance_us"] = df1.loc[i, "Final_balance"]
        temp_dict["Final Balance_untrade"] = df.loc[i, "Final Balance"]
        
        df_compare.loc[len(df_compare)] = temp_dict


    df_compare.to_csv("benchmark_compare.csv", index=False)
