from scipy.signal import butter, cheby1, filtfilt
import pandas as pd
import numpy as np
from easydict import EasyDict
from scipy.signal import get_window, fftconvolve
from scipy.ndimage import gaussian_filter1d
from pykalman import KalmanFilter


class BaseStrategy:
    def __init__(
        self,
        high_csv: pd.DataFrame,
        low_csv: pd.DataFrame,
        config: EasyDict,
        glob: EasyDict,
    ):
        self.high_csv = high_csv
        self.shifted_high_csv = high_csv.shift(1)
        self.low_csv = low_csv
        self.config = config
        self.glob = glob
        self.preprocessing()

    def preprocessing(self):
        pass

    def check_long_entry(self, high_pointer: int):
        pass

    def check_short_entry(self, high_pointer: int):
        pass

    def check_long_exit(self, high_pointer: int):
        pass

    def check_short_exit(self, high_pointer: int):
        pass


class EMAStrategy(BaseStrategy):
    def preprocessing(self):
        self.high_csv["long_EMA"] = (
            self.high_csv["close"].ewm(span=12, adjust=False).mean()
        )
        self.high_csv["short_EMA"] = (
            self.high_csv["close"].ewm(span=9, adjust=False).mean()
        )

    def check_long_entry(self, high_pointer: int):
        long_ema = self.high_csv["long_EMA"].iloc[high_pointer]
        short_ema = self.high_csv["short_EMA"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if short_ema > long_ema:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        long_ema = self.high_csv["long_EMA"].iloc[high_pointer]
        short_ema = self.high_csv["short_EMA"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if short_ema < long_ema:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        return 0

    def check_short_exit(self, high_pointer: int):
        return 0


class ButterChebyStrategy(BaseStrategy):
    def butterworth(self, data: pd.DataFrame):
        order = self.config.strategies.strat_cheby.butterworth.order
        cutoff_freq = self.config.strategies.strat_cheby.butterworth.cutoff_frequency
        b, a = butter(N=order, Wn=cutoff_freq, btype="low", analog=False, output="ba")
        smooth_data = filtfilt(b, a, data["close"], padlen=0)
        return smooth_data

    def chebyshev(self, data: pd.DataFrame):
        cutoff_freq = self.config.strategies.strat_cheby.chebyshev.cutoff_frequency
        rp = self.config.strategies.strat_cheby.chebyshev.ripple_factor
        order = self.config.strategies.strat_cheby.chebyshev.order
        b, a = cheby1(
            N=order, rp=rp, Wn=cutoff_freq, btype="low", analog=False, output="ba"
        )
        smooth_data = filtfilt(b, a, data["close"], padlen=0)
        return smooth_data

    def preprocessing(self):
        self.high_csv["butter"] = 0.0
        self.high_csv["cheby"] = 0.0
        df_temp = pd.DataFrame(columns=self.high_csv.columns)
        for i in range(1, len(self.high_csv) + 1):
            df_temp.loc[len(df_temp)] = self.high_csv.loc[i - 1]
            self.high_csv.loc[i - 1, "butter"] = self.butterworth(df_temp)[-1]
            self.high_csv.loc[i - 1, "cheby"] = self.chebyshev(df_temp)[-1]
        self.low_csv["tp"] = self.glob.tp
        self.low_csv["sl"] = self.glob.sl

    def check_long_entry(self, high_pointer: int):
        i = high_pointer
        if not i:
            return False
        c1 = self.high_csv.loc[i, "cheby"] > self.high_csv.loc[i, "butter"]
        c2 = self.high_csv.loc[i - 1, "cheby"] < self.high_csv.loc[i - 1, "butter"]
        Close = self.high_csv.loc[i, "close"]
        if c1 and c2:
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return True
        return False

    def check_short_entry(self, high_pointer: int):
        i = high_pointer
        if not i:
            return False
        c1 = self.high_csv.loc[i, "cheby"] < self.high_csv.loc[i, "butter"]
        c2 = self.high_csv.loc[i - 1, "cheby"] > self.high_csv.loc[i - 1, "butter"]
        Close = self.high_csv.loc[i, "close"]
        if c1 and c2:
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return True
        return False

    def check_long_exit(self, high_pointer: int):
        return 0

    def check_short_exit(self, high_pointer: int):
        return 0


class Neelabh_Strategy(BaseStrategy):
    def ehlers_bandpass_filter(
        self, data, period=20, adaptive=True, delta=0.25, fraction=0.25
    ):
        PI = np.pi
        imult = 0.635
        qmult = 0.338
        inphase = np.zeros(len(data))
        quadrature = np.zeros(len(data))
        re = np.zeros(len(data))
        im = np.zeros(len(data))
        delta_IQ = np.zeros(len(data))  # Initialize delta_IQ to avoid UnboundLocalError
        inst = np.zeros(len(data))
        V = np.zeros(len(data))

        if adaptive:
            for i in range(7, len(data)):
                P = data[i] - data[i - 7]
                inphase[i] = (
                    1.25 * (data[i - 3] - imult * data[i - 5]) + imult * inphase[i - 3]
                )
                quadrature[i] = (
                    data[i - 5] - qmult * data[i] + qmult * quadrature[i - 2]
                )
                re[i] = (
                    0.2
                    * (inphase[i] * inphase[i - 1] + quadrature[i] * quadrature[i - 1])
                    + 0.8 * re[i - 1]
                )
                im[i] = (
                    0.2
                    * (inphase[i] * quadrature[i - 1] - inphase[i - 1] * quadrature[i])
                    + 0.8 * im[i - 1]
                )

                if re[i] != 0:
                    delta_IQ[i] = np.arctan(im[i] / re[i])

                inst_values = np.where(np.cumsum(delta_IQ[: i + 1]) > 2 * PI)[0]
                inst[i] = inst_values[0] if len(inst_values) > 0 else inst[i - 1]

                period = max(8, 0.25 * inst[i] + 0.75 * period)

        # Bandpass Filter Parameters
        beta = np.cos((2 * PI) / period)
        gamma = 1 / np.cos((4 * PI * delta) / period)
        alpha = gamma - np.sqrt(gamma**2 - 1)
        BP = np.zeros(len(data))

        # Calculate Bandpass (BP) and Trend
        for i in range(2, len(data)):
            BP[i] = (
                0.5 * (1 - alpha) * (data[i] - data[i - 2])
                + beta * (1 + alpha) * BP[i - 1]
                - alpha * BP[i - 2]
            )

        trend = pd.Series(BP).rolling(window=int(2 * period)).mean().values

        # Peak and Valley calculations
        Peak = np.zeros(len(data))
        Valley = np.zeros(len(data))
        for i in range(2, len(data)):
            Peak[i] = (
                BP[i - 1]
                if BP[i - 1] > BP[i] and BP[i - 1] > BP[i - 2]
                else Peak[i - 1]
            )
            Valley[i] = (
                BP[i - 1]
                if BP[i - 1] < BP[i] and BP[i - 1] < BP[i - 2]
                else Valley[i - 1]
            )

        AvgPeak = pd.Series(Peak).rolling(window=int(2 * period)).mean() * fraction
        AvgValley = pd.Series(Valley).rolling(window=int(2 * period)).mean() * fraction
        Middle = (AvgPeak + AvgValley) / 2

        result = pd.DataFrame(
            {"trend": trend, "upper": AvgPeak, "lower": AvgValley, "middle": Middle}
        )
        return result

    def ehlers_hilbert_oscillator(slef, data):
        """
        Calculate the Ehlers Hilbert Oscillator (I1, Q1, I3, Q3, Value1) on price data.
        :param data: DataFrame with 'price' column.
        :return: DataFrame with I1, Q1, I3, Q3, and Value1 columns.
        """
        pi = 2 * np.arcsin(1)
        period = np.zeros(len(data))
        I1, Q1 = np.zeros(len(data)), np.zeros(len(data))
        jI, jQ = np.zeros(len(data)), np.zeros(len(data))
        I2, Q2 = np.zeros(len(data)), np.zeros(len(data))
        Re, Im = np.zeros(len(data)), np.zeros(len(data))
        smooth_period = np.zeros(len(data))
        I3, Q3, Value1 = np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))
        smoothed = np.zeros(len(data))
        detrended = np.zeros(len(data))

        for i in range(6, len(data)):
            # Smooth the input data
            smoothed[i] = (
                4 * data[i] + 3 * data[i - 1] + 2 * data[i - 2] + data[i - 3]
            ) / 10

            # Calculate the Detrender
            detrended[i] = (
                0.0962 * smoothed[i]
                + 0.5769 * smoothed[i - 2]
                - 0.5769 * smoothed[i - 4]
                - 0.0962 * smoothed[i - 6]
            )

            # Compute I1 and Q1
            Q1[i] = (
                0.0962 * detrended[i]
                + 0.5769 * detrended[i - 2]
                - 0.5769 * detrended[i - 4]
                - 0.0962 * detrended[i - 6]
            )
            I1[i] = detrended[i - 3]

            # Advance the phase of I1 and Q1 by 90 degrees
            jI[i] = (
                0.0962 * I1[i]
                + 0.5769 * I1[i - 2]
                - 0.5769 * I1[i - 4]
                - 0.0962 * I1[i - 6]
            )
            jQ[i] = (
                0.0962 * Q1[i]
                + 0.5769 * Q1[i - 2]
                - 0.5769 * Q1[i - 4]
                - 0.0962 * Q1[i - 6]
            )

            # Phasor addition for 3-bar averaging
            I2[i] = I1[i] - jQ[i]
            Q2[i] = Q1[i] + jI[i]

            # Smooth I2 and Q2
            I2[i] = 0.2 * I2[i] + 0.8 * I2[i - 1]
            Q2[i] = 0.2 * Q2[i] + 0.8 * Q2[i - 1]

            # Homodyne Discriminator
            Re[i] = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]
            Im[i] = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]
            Re[i] = 0.2 * Re[i] + 0.8 * Re[i - 1]
            Im[i] = 0.2 * Im[i] + 0.8 * Im[i - 1]

            # Calculate Period
            if Im[i] != 0 and Re[i] != 0:
                period[i] = 2 * pi / np.arctan(Im[i] / Re[i])
            period[i] = np.clip(period[i], 6, 50)
            period[i] = 0.2 * period[i] + 0.8 * period[i - 1]

            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

            # Calculate Q3 and I3
            Q3[i] = (
                0.5
                * (smoothed[i] - smoothed[i - 2])
                * (0.1759 * smooth_period[i] + 0.4607)
            )
            for count in range(int(np.ceil(smooth_period[i] / 2))):
                I3[i] += Q3[i - count]
            I3[i] = 1.57 * I3[i] / np.ceil(smooth_period[i] / 2)

            # Calculate Value1
            for count in range(int(np.ceil(smooth_period[i] / 4))):
                Value1[i] += Q3[i - count]
            Value1[i] = 1.25 * Value1[i] / np.ceil(smooth_period[i] / 4)

        # Store results in DataFrame
        result = pd.DataFrame(
            {"I1": I1, "Q1": Q1, "I3": I3, "Q3": Q3, "Value1": Value1}
        )
        return result

    def preprocessing(self):
        data = self.high_csv["close"]
        bandpass_result = self.ehlers_bandpass_filter(data)
        bandpass_result.fillna(0, inplace=True)
        bandpass_result["upper"] = bandpass_result["upper"].fillna(0)
        data1 = bandpass_result["upper"]
        oscillator_data1 = self.ehlers_hilbert_oscillator(data1)
        self.high_csv["I1"] = oscillator_data1["I1"]
        self.high_csv["Value1"] = oscillator_data1["Value1"]
        # print(self.high_csv)

    def check_long_entry(self, high_pointer: int):
        close = self.high_csv["close"].iloc[high_pointer]
        if (
            self.high_csv["Value1"].iloc[high_pointer]
            > self.high_csv["I1"].iloc[high_pointer]
            and self.high_csv["Value1"].iloc[high_pointer - 1]
            < self.high_csv["I1"].iloc[high_pointer - 1]
        ):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return True
        return False

    def check_short_entry(self, high_pointer: int):
        close = self.high_csv["close"].iloc[high_pointer]
        if (
            self.high_csv["Value1"].iloc[high_pointer]
            < self.high_csv["I1"].iloc[high_pointer]
            and self.high_csv["Value1"].iloc[high_pointer - 1]
            > self.high_csv["I1"].iloc[high_pointer - 1]
        ):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        return 0

    def check_short_exit(self, high_pointer: int):
        return 0


class gaussian_lsma1(BaseStrategy):
    def get_gaussian_filter(self, cyclePeriod=11, poles=3):
        PI = np.pi
        beta = (1 - np.cos(2 * PI / cyclePeriod)) / (pow(2, 1 / poles) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)
        src = self.high_csv["close"]
        filter_arr = np.zeros_like(src)

        # Loop through each bar and calculate the filter value
        for i in range(len(src)):
            if poles == 1:
                filter_arr[i] = (
                    alpha * src[i] + (1 - alpha) * filter_arr[i - 1]
                    if i >= 1
                    else src[i]
                )
            elif poles == 2:
                filter_arr[i] = (
                    (
                        pow(alpha, 2) * src[i]
                        + 2 * (1 - alpha) * filter_arr[i - 1]
                        - pow(1 - alpha, 2) * filter_arr[i - 2]
                    )
                    if i >= 2
                    else src[i]
                )
            elif poles == 3:
                filter_arr[i] = (
                    (
                        pow(alpha, 3) * src[i]
                        + 3 * (1 - alpha) * filter_arr[i - 1]
                        - 3 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + pow(1 - alpha, 3) * filter_arr[i - 3]
                    )
                    if i >= 3
                    else src[i]
                )
            elif poles == 4:
                filter_arr[i] = (
                    (
                        pow(alpha, 4) * src[i]
                        + 4 * (1 - alpha) * filter_arr[i - 1]
                        - 6 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + 4 * pow(1 - alpha, 3) * filter_arr[i - 3]
                        - pow(1 - alpha, 4) * filter_arr[i - 4]
                    )
                    if i >= 4
                    else src[i]
                )

        return filter_arr

    def lsma(self, length=5):
        # Calculate the x-values (time indices) for the regression
        x = np.arange(length)

        # Initialize an array to store the LSMA values
        lsma_values = np.full(len(self.high_csv), np.nan)

        # Loop through each point where we can calculate LSMA
        for i in range(length - 1, len(self.high_csv)):
            y = (
                self.high_csv["close"].iloc[i - length + 1 : i + 1].values
            )  # Get y-values (close prices)

            # Calculate the slope (m) and intercept (b) for the line of best fit
            A = np.vstack([x, np.ones(length)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calculate the LSMA as the value at the end of the regression line
            lsma_values[i] = m * (length - 1) + b

        return lsma_values

    def hawkes_process(self):
        """Implement Hawkes process for volatility estimation."""
        kappa = 3
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv["close"].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv["close"].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def atr(self, period=14):
        high = self.high_csv["high"]
        low = self.high_csv["low"]
        close = self.high_csv["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    def rsi(self):
        delta = self.high_csv["close"].diff()
        rsi_length = 10

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(window=rsi_length, min_periods=rsi_length).mean()
        avg_loss = losses.rolling(window=rsi_length, min_periods=rsi_length).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero or infinite values
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(100)

        return rsi

    def preprocessing(self):
        self.high_csv["LSMA"] = self.lsma()
        self.high_csv["Gaussian Filter"] = self.get_gaussian_filter()
        hawkes_window = 5
        entry_percentile = 0.95
        exit_percentile = 0.1
        self.high_csv["vol_hawkes"] = self.hawkes_process()
        self.high_csv["vol_entry_threshold"] = (
            self.high_csv["vol_hawkes"]
            .rolling(window=hawkes_window, min_periods=1)
            .quantile(entry_percentile)
        )
        self.high_csv["vol_exit_threshold"] = (
            self.high_csv["vol_hawkes"]
            .rolling(window=hawkes_window, min_periods=1)
            .quantile(exit_percentile)
        )
        self.high_csv["rsi"] = self.rsi()
        self.high_csv["atr"] = self.atr()

    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1:
            return 0
        close = self.high_csv["close"].iloc[high_pointer]
        if (
            self.high_csv["LSMA"].iloc[high_pointer]
            > self.high_csv["Gaussian Filter"].iloc[high_pointer]
            and self.high_csv["LSMA"].iloc[high_pointer - 1]
            <= self.high_csv["Gaussian Filter"].iloc[high_pointer - 1]
            and self.high_csv["rsi"].iloc[high_pointer] > 35
        ):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1:
            return 0
        close = self.high_csv["close"].iloc[high_pointer]
        if (
            self.high_csv["LSMA"].iloc[high_pointer]
            < self.high_csv["Gaussian Filter"].iloc[high_pointer]
            and self.high_csv["LSMA"].iloc[high_pointer - 1]
            >= self.high_csv["Gaussian Filter"].iloc[high_pointer - 1]
            and self.high_csv["rsi"].iloc[high_pointer] < 70
        ):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0
        if (
            self.high_csv["vol_hawkes"].iloc[high_pointer]
            < self.high_csv["vol_exit_threshold"].iloc[high_pointer]
        ):
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0
        if (
            self.high_csv["vol_hawkes"].iloc[high_pointer]
            < self.high_csv["vol_exit_threshold"].iloc[high_pointer]
        ):
            return 1
        return 0


class Gaussian_Kalman(BaseStrategy):
    def gaussian_filter_func(self, window_size=5, sigma=1):
        return gaussian_filter1d(self.high_csv["close"].values, sigma=sigma)

    def kalman_filter_func(self):
        kf = KalmanFilter(
            initial_state_mean=self.high_csv["close"].iloc[0], n_dim_obs=1
        )
        print("values passed to kalman")
        print(self.high_csv["close"].values)
        state_means, _ = kf.filter(self.high_csv["close"].values)
        return state_means.flatten()

    def butterworth_filter(self, order=2, cutoff=0.1):
        nyq = 0.5
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, self.high_csv["close"].values)

    def causal_rloess_filter_func(self, frac=0.3):
        n = len(self.high_csv["close"].values)
        smoothed = np.full(n, np.nan)
        for i in range(n):
            window_size = int(frac * n)
            start = max(0, i - window_size)
            end = i + 1
            y = self.high_csv["close"].values[start:end]
            x = np.arange(start, end)
            coeff = np.polyfit(x, y, 1)
            smoothed[i] = np.polyval(coeff, i)
        return smoothed

    def median_filter_func(self, window_size=3):
        return (
            self.high_csv["close"]
            .values.rolling(window=window_size, min_periods=1)
            .median()
            .values
        )

    def lsma_func(self, length=14):
        x = np.arange(length)
        lsma = np.full(len(self.high_csv["close"].values), np.nan)
        for i in range(length - 1, len(self.high_csv["close"].values)):
            y = self.high_csv[i - length + 1 : i + 1]
            A = np.vstack([x, np.ones(length)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            lsma[i] = m * (length - 1) + c
        return lsma

    def hamming_window_filter_func(self, window_size=5):
        window = get_window("hamming", window_size)
        window = window / window.sum()
        filtered = fftconvolve(self.high_csv["close"].values, window, mode="same")
        return filtered

    def rolling_quantile_func(self, window=10, quantile=0.5):
        return (
            self.high_csv["close"]
            .values.rolling(window=window, min_periods=1)
            .quantile(quantile)
            .values
        )

    def preprocessing(self):
        self.high_csv["butterworth"] = self.gaussian_filter_func()
        self.high_csv["rloess"] = self.kalman_filter_func()
        self.shifted_high_csv["close"] = self.high_csv["close"].shift(1)

    def check_long_entry(self, high_pointer: int):
        if high_pointer == 0:
            return 0
        butterworth = self.high_csv["butterworth"].iloc[high_pointer]
        butterworth_1 = self.high_csv.shift(1)["butterworth"].iloc[high_pointer]
        rloess = self.high_csv["rloess"].iloc[high_pointer]
        rloess_1 = self.high_csv.shift(1)["rloess"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if butterworth > rloess and rloess_1 <= butterworth_1:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if high_pointer == 0:
            return 0
        butterworth = self.high_csv["butterworth"].iloc[high_pointer]
        butterworth_1 = self.high_csv.shift(1)["butterworth"].iloc[high_pointer]
        rloess = self.high_csv["rloess"].iloc[high_pointer]
        rloess_1 = self.high_csv.shift(1)["rloess"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if butterworth < rloess and rloess_1 >= butterworth_1:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        return 0

    def check_short_exit(self, high_pointer: int):
        return 0


class VidyaCrossoverStrategy(BaseStrategy):
    def preprocessing(self):
        # Calculate VIDYA with lengths 5 and 9
        close_prices = self.high_csv["close"]
        self.vidya_5 = self.vidya(close_prices, pds=8)
        self.vidya_9 = self.vidya(close_prices, pds=14)
        # Add VIDYA columns to high_csv DataFrame for easy access
        self.high_csv["VIDYA_5"] = self.vidya_5
        self.high_csv["VIDYA_9"] = self.vidya_9

    def vidya(self, price_series, pds=9, fix_cmo=True, select_cmo=True):
        """
        Compute the Variable Index Dynamic Average (VIDYA) of a price series.
        """
        price_series = pd.Series(price_series).reset_index(drop=True)
        alpha = 2 / (pds + 1)
        momm = price_series.diff().fillna(0)
        m1 = momm.apply(lambda x: x if x >= 0 else 0)
        m2 = momm.apply(lambda x: -x if x < 0 else 0)
        cmo_period = 9 if fix_cmo else pds
        sm1 = m1.rolling(window=cmo_period).sum()
        sm2 = m2.rolling(window=cmo_period).sum()
        cmo_denominator = sm1 + sm2
        cmo_denominator = cmo_denominator.replace(0, np.nan)
        chande_mo = 100 * (sm1 - sm2) / cmo_denominator
        chande_mo = chande_mo.fillna(0)
        if select_cmo:
            k = chande_mo.abs() / 100
        else:
            k = price_series.rolling(window=pds).std().fillna(0)
        vidya = np.zeros(len(price_series))
        vidya[0] = price_series.iloc[0]
        for t in range(1, len(price_series)):
            alpha_k = alpha * k.iloc[t]
            alpha_k = min(max(alpha_k, 0), 1)
            vidya[t] = alpha_k * price_series.iloc[t] + (1 - alpha_k) * vidya[t - 1]
        vidya_series = pd.Series(vidya, index=price_series.index)
        return vidya_series

    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1:
            return 0  # Already in a long position
        close = self.high_csv["close"].iloc[high_pointer]
        # Condition 1: VIDYA(5) crosses above VIDYA(9)
        cond1 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer]
            > self.high_csv["VIDYA_9"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer - 1]
            <= self.high_csv["VIDYA_9"].iloc[high_pointer - 1]
        )
        if cond1 and cond2:
            self.glob.entry_price = close
            # Update the global status to long position
            self.glob.status = 1
            return 1  # Signal to buy
        return 0  # Hold

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1:
            return 0  # Already in a short position
        close = self.high_csv["close"].iloc[high_pointer]
        # Condition 1: VIDYA(5) crosses below VIDYA(9)
        cond1 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer]
            < self.high_csv["VIDYA_9"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer - 1]
            >= self.high_csv["VIDYA_9"].iloc[high_pointer - 1]
        )
        if cond1 and cond2:
            self.glob.entry_price = close
            # Update the global status to short position
            self.glob.status = -1
            return 1  # Signal to sell
        return 0  # Hold

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0  # Not in a long position
        # Condition: VIDYA(5) crosses below VIDYA(9)
        cond1 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer]
            < self.high_csv["VIDYA_9"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer - 1]
            >= self.high_csv["VIDYA_9"].iloc[high_pointer - 1]
        )
        if cond1 and cond2:
            # Exit long position
            self.glob.status = 0  # Update status to flat
            return 1  # Signal to exit long
        return 0  # Hold

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0  # Not in a short position
        # Condition: VIDYA(5) crosses above VIDYA(9)
        cond1 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer]
            > self.high_csv["VIDYA_9"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["VIDYA_5"].iloc[high_pointer - 1]
            <= self.high_csv["VIDYA_9"].iloc[high_pointer - 1]
        )
        if cond1 and cond2:
            # Exit short position
            self.glob.status = 0  # Update status to flat
            return 1  # Signal to exit short
        return 0  # Hold


class My_Strategy_(BaseStrategy):
    def get_gaussian_filter(self, cyclePeriod=11, poles=3):
        PI = np.pi
        beta = (1 - np.cos(2 * PI / cyclePeriod)) / (pow(2, 1 / poles) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)
        src = self.high_csv["close"]
        filter_arr = np.zeros_like(src)

        # Loop through each bar and calculate the filter value
        for i in range(len(src)):
            if poles == 1:
                filter_arr[i] = (
                    alpha * src[i] + (1 - alpha) * filter_arr[i - 1]
                    if i >= 1
                    else src[i]
                )
            elif poles == 2:
                filter_arr[i] = (
                    (
                        pow(alpha, 2) * src[i]
                        + 2 * (1 - alpha) * filter_arr[i - 1]
                        - pow(1 - alpha, 2) * filter_arr[i - 2]
                    )
                    if i >= 2
                    else src[i]
                )
            elif poles == 3:
                filter_arr[i] = (
                    (
                        pow(alpha, 3) * src[i]
                        + 3 * (1 - alpha) * filter_arr[i - 1]
                        - 3 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + pow(1 - alpha, 3) * filter_arr[i - 3]
                    )
                    if i >= 3
                    else src[i]
                )
            elif poles == 4:
                filter_arr[i] = (
                    (
                        pow(alpha, 4) * src[i]
                        + 4 * (1 - alpha) * filter_arr[i - 1]
                        - 6 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + 4 * pow(1 - alpha, 3) * filter_arr[i - 3]
                        - pow(1 - alpha, 4) * filter_arr[i - 4]
                    )
                    if i >= 4
                    else src[i]
                )

        return filter_arr

    def lsma(self, length=5):
        # Calculate the x-values (time indices) for the regression
        x = np.arange(length)

        # Initialize an array to store the LSMA values
        lsma_values = np.full(len(self.high_csv), np.nan)

        # Loop through each point where we can calculate LSMA
        for i in range(length - 1, len(self.high_csv)):
            y = (
                self.high_csv["close"].iloc[i - length + 1 : i + 1].values
            )  # Get y-values (close prices)

            # Calculate the slope (m) and intercept (b) for the line of best fit
            A = np.vstack([x, np.ones(length)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calculate the LSMA as the value at the end of the regression line
            lsma_values[i] = m * (length - 1) + b

        return lsma_values

    def hawkes_process(self):
        """Implement Hawkes process for volatility estimation."""
        kappa = 3
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv["close"].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv["close"].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def adx(self, period=10):
        high = self.high_csv["high"]
        low = self.high_csv["low"]
        close = self.high_csv["close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx

    def rsi(self):
        delta = self.high_csv["close"].diff()
        rsi_length = 10

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(window=rsi_length, min_periods=rsi_length).mean()
        avg_loss = losses.rolling(window=rsi_length, min_periods=rsi_length).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero or infinite values
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(100)

        return rsi

    def preprocessing(self):
        self.high_csv["LSMA"] = self.lsma()
        self.high_csv["Gaussian Filter"] = self.get_gaussian_filter()
        hawkes_window = 11
        entry_percentile = 0.95
        exit_percentile = 0.05
        self.high_csv["vol_hawkes"] = self.hawkes_process()
        self.high_csv["vol_entry_threshold"] = (
            self.high_csv["vol_hawkes"]
            .rolling(window=hawkes_window, min_periods=1)
            .quantile(entry_percentile)
        )
        self.high_csv["vol_exit_threshold"] = (
            self.high_csv["vol_hawkes"]
            .rolling(window=hawkes_window, min_periods=1)
            .quantile(exit_percentile)
        )
        self.high_csv["rsi"] = self.rsi()
        self.high_csv["adx"] = self.adx()

    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1:
            return 0
        close = self.high_csv["close"].iloc[high_pointer]
        cond1 = (
            self.high_csv["LSMA"].iloc[high_pointer]
            > self.high_csv["Gaussian Filter"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["LSMA"].iloc[high_pointer - 1]
            <= self.high_csv["Gaussian Filter"].iloc[high_pointer - 1]
        )
        cond3 = (
            self.high_csv["adx"].iloc[high_pointer] < 55
            and self.high_csv["rsi"].iloc[high_pointer] > 35
        )
        if cond1 and cond2 and cond3:
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1:
            return 0
        close = self.high_csv["close"].iloc[high_pointer]
        cond1 = (
            self.high_csv["LSMA"].iloc[high_pointer]
            < self.high_csv["Gaussian Filter"].iloc[high_pointer]
        )
        cond2 = (
            self.high_csv["LSMA"].iloc[high_pointer - 1]
            >= self.high_csv["Gaussian Filter"].iloc[high_pointer - 1]
        )
        cond3 = (
            self.high_csv["adx"].iloc[high_pointer] < 55
            and self.high_csv["rsi"].iloc[high_pointer] < 70
        )
        if cond1 and cond2 and cond3:
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0
        if (
            self.high_csv["vol_hawkes"].iloc[high_pointer]
            < self.high_csv["vol_exit_threshold"].iloc[high_pointer]
        ):
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0
        if (
            self.high_csv["vol_hawkes"].iloc[high_pointer]
            < self.high_csv["vol_exit_threshold"].iloc[high_pointer]
        ):
            return 1
        return 0


class Gaussian_LSMA_Hawkes_ADX_RSI(BaseStrategy):

    def get_gaussian_filter(self, cyclePeriod=11, poles=3):
        PI = np.pi
        beta = (1 - np.cos(2 * PI / cyclePeriod)) / (pow(2, 1 / poles) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)
        src = self.high_csv['close']
        filter_arr = np.zeros_like(src)

        # Loop through each bar and calculate the filter value
        for i in range(len(src)):
            if poles == 1:
                filter_arr[i] = alpha * src[i] + (1 - alpha) * filter_arr[i - 1] if i >= 1 else src[i]
            elif poles == 2:
                filter_arr[i] = (
                        pow(alpha, 2) * src[i]
                        + 2 * (1 - alpha) * filter_arr[i - 1]
                        - pow(1 - alpha, 2) * filter_arr[i - 2]
                ) if i >= 2 else src[i]
            elif poles == 3:
                filter_arr[i] = (
                        pow(alpha, 3) * src[i]
                        + 3 * (1 - alpha) * filter_arr[i - 1]
                        - 3 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + pow(1 - alpha, 3) * filter_arr[i - 3]
                ) if i >= 3 else src[i]
            elif poles == 4:
                filter_arr[i] = (
                        pow(alpha, 4) * src[i]
                        + 4 * (1 - alpha) * filter_arr[i - 1]
                        - 6 * pow(1 - alpha, 2) * filter_arr[i - 2]
                        + 4 * pow(1 - alpha, 3) * filter_arr[i - 3]
                        - pow(1 - alpha, 4) * filter_arr[i - 4]
                ) if i >= 4 else src[i]

        return filter_arr

    def lsma(self, length=5):
        # Calculate the x-values (time indices) for the regression
        x = np.arange(length)

        # Initialize an array to store the LSMA values
        lsma_values = np.full(len(self.high_csv), np.nan)

        # Loop through each point where we can calculate LSMA
        for i in range(length - 1, len(self.high_csv)):
            y = self.high_csv['close'].iloc[i - length + 1:i + 1].values  # Get y-values (close prices)

            # Calculate the slope (m) and intercept (b) for the line of best fit
            A = np.vstack([x, np.ones(length)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calculate the LSMA as the value at the end of the regression line
            lsma_values[i] = m * (length - 1) + b

        return lsma_values

    def hawkes_process(self, k=3):
        """Implement Hawkes process for volatility estimation."""
        kappa = k
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv['close'].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv['close'].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def adx(self, period=10):
        high = self.high_csv['high']
        low = self.high_csv['low']
        close = self.high_csv['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx

    def rsi(self, length = 10):
        delta = self.high_csv['close'].diff()
        rsi_length = length

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(window=rsi_length, min_periods=rsi_length).mean()
        avg_loss = losses.rolling(window=rsi_length, min_periods=rsi_length).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero or infinite values
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(100)

        return rsi

    def preprocessing(self):
        self.high_csv['LSMA'] = self.lsma(length=5)
        self.high_csv['Gaussian Filter'] = self.get_gaussian_filter(cyclePeriod=9, poles=3)
        hawkes_window = 15
        entry_percentile = 0.95
        exit_percentile = 0.05
        self.high_csv['vol_hawkes'] = self.hawkes_process(k=3)
        self.high_csv['vol_entry_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window,
                                                                                   min_periods=1).quantile(
            entry_percentile)
        self.high_csv['vol_exit_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window,
                                                                                  min_periods=1).quantile(
            exit_percentile)
        self.high_csv['rsi'] = self.rsi(length=5)
        self.high_csv['adx'] = self.adx(period=10)

    def check_long_entry(self, high_pointer: int):
        if (self.glob.status == 1):
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        cond1 = self.high_csv['LSMA'].iloc[high_pointer] > self.high_csv['Gaussian Filter'].iloc[high_pointer]
        cond2 = self.high_csv['LSMA'].iloc[high_pointer - 1] <= self.high_csv['Gaussian Filter'].iloc[high_pointer - 1]
        cond3 = self.high_csv['adx'].iloc[high_pointer] < 50 and self.high_csv['rsi'].iloc[high_pointer] > 35
        if (cond1 and cond2 and cond3):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if (self.glob.status == -1):
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        cond1 = self.high_csv['LSMA'].iloc[high_pointer] < self.high_csv['Gaussian Filter'].iloc[high_pointer]
        cond2 = self.high_csv['LSMA'].iloc[high_pointer - 1] >= self.high_csv['Gaussian Filter'].iloc[high_pointer - 1]
        cond3 = self.high_csv['adx'].iloc[high_pointer] < 50 and self.high_csv['rsi'].iloc[high_pointer] < 65
        if (cond1 and cond2 and cond3):
            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if (self.glob.status != 1):
            return 0
        if (self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['vol_exit_threshold'].iloc[high_pointer]):
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if (self.glob.status != -1):
            return 0
        if (self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['vol_exit_threshold'].iloc[high_pointer]):
            return 1
        return 0
    


    # Etherium Strategy
class My_Strategy_8(BaseStrategy):

    def tema(self, length=9):
        """Calculate the Triple Exponential Moving Average (TEMA)."""
        close = self.high_csv['close']
        # Calculate the first EMA
        ema1 = close.ewm(span=length, adjust=False).mean()
        # Calculate the second EMA
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        # Calculate the third EMA
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        # Calculate TEMA
        tema = 3 * (ema1 - ema2) + ema3
        return tema

    def get_gaussian_filter(self, cyclePeriod=11, poles=3):
        PI = np.pi
        beta = (1 - np.cos(2 * PI / cyclePeriod)) / (pow(2, 1 / poles) - 1)
        alpha = -beta + np.sqrt(pow(beta, 2) + 2 * beta)
        src = self.high_csv['close']
        filter_arr = np.zeros_like(src)

        for i in range(len(src)):
            if poles == 1:
                filter_arr[i] = alpha * src[i] + (1 - alpha) * filter_arr[i - 1] if i >= 1 else src[i]
            elif poles == 2:
                filter_arr[i] = (
                    pow(alpha, 2) * src[i]
                    + 2 * (1 - alpha) * filter_arr[i - 1]
                    - pow(1 - alpha, 2) * filter_arr[i - 2]
                ) if i >= 2 else src[i]
            elif poles == 3:
                filter_arr[i] = (
                    pow(alpha, 3) * src[i]
                    + 3 * (1 - alpha) * filter_arr[i - 1]
                    - 3 * pow(1 - alpha, 2) * filter_arr[i - 2]
                    + pow(1 - alpha, 3) * filter_arr[i - 3]
                ) if i >= 3 else src[i]
            elif poles == 4:
                filter_arr[i] = (
                    pow(alpha, 4) * src[i]
                    + 4 * (1 - alpha) * filter_arr[i - 1]
                    - 6 * pow(1 - alpha, 2) * filter_arr[i - 2]
                    + 4 * pow(1 - alpha, 3) * filter_arr[i - 3]
                    - pow(1 - alpha, 4) * filter_arr[i - 4]
                ) if i >= 4 else src[i]

        return filter_arr

    def lsma(self, length=5):
        x = np.arange(length)
        lsma_values = np.full(len(self.high_csv), np.nan)

        for i in range(length - 1, len(self.high_csv)):
            y = self.high_csv['close'].iloc[i - length + 1:i + 1].values
            A = np.vstack([x, np.ones(length)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            lsma_values[i] = m * (length - 1) + b

        return lsma_values

    def hawkes_process(self, k=3.0):
        """Implement Hawkes process for volatility estimation."""
        kappa = k
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv['close'].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv['close'].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def adx(self, period=10):
        high = self.high_csv['high']
        low = self.high_csv['low']
        close = self.high_csv['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx

    def preprocessing(self):
        self.high_csv['LSMA'] = self.lsma(length=5)
        self.high_csv['Gaussian Filter'] = self.get_gaussian_filter(cyclePeriod=11, poles=3)
        self.high_csv['TEMA'] = self.tema(length=800)  # Calculate TEMA and add to DataFrame
        entry_window = 8
        exit_window = 16
        entry_percentile = 0.05
        exit_percentile = 0.1
        self.high_csv['vol_hawkes'] = self.hawkes_process(k=3)
        self.high_csv['vol_entry_threshold'] = self.high_csv['vol_hawkes'].rolling(window=entry_window,
                                                                                   min_periods=1).quantile(
            entry_percentile)
        self.high_csv['vol_exit_threshold'] = self.high_csv['vol_hawkes'].rolling(window=exit_window,
                                                                                  min_periods=1).quantile(
            exit_percentile)
        self.high_csv['adx'] = self.adx(period=12)

    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 30:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['LSMA'].iloc[high_pointer] > self.high_csv['Gaussian Filter'].iloc[high_pointer]
        cond2 = self.high_csv['LSMA'].iloc[high_pointer - 1] <= self.high_csv['Gaussian Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55


        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is increasing
            if tema_curr > tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.08
                self.glob.sl = 0.04
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 30:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['LSMA'].iloc[high_pointer] < self.high_csv['Gaussian Filter'].iloc[high_pointer]
        cond2 = self.high_csv['LSMA'].iloc[high_pointer - 1] >= self.high_csv['Gaussian Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55

        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is decreasing
            if tema_curr < tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.08
                self.glob.sl = 0.04
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['vol_exit_threshold'].iloc[high_pointer]:
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['vol_entry_threshold'].iloc[high_pointer]:
            return 1
        return 0

    

class My_Strategy_9(BaseStrategy):

    def hma(self, length=5):
        """Calculate the Hull Moving Average (HMA)."""
        import numpy as np
        data = self.high_csv
        length = int(length)
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))

        # Weighted Moving Average (WMA) function
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

        # Calculate WMA for full length and half length
        wma_full = wma(data['close'], length)
        wma_half = wma(data['close'], half_length)

        # Calculate the difference
        diff = 2 * wma_half - wma_full

        # HMA is WMA of diff over sqrt(length)
        hma = wma(diff, sqrt_length)
        return hma

    def laguerre_filter(self, alpha=0.85):
        """Calculate the Laguerre Filter."""
        import numpy as np
        data = self.high_csv
        gamma = 1 - alpha
        src = (data['high'] + data['low']) / 2  # hl2

        L0 = np.zeros(len(src))
        L1 = np.zeros(len(src))
        L2 = np.zeros(len(src))
        L3 = np.zeros(len(src))
        LagF = np.zeros(len(src))

        for i in range(len(src)):
            if i == 0:
                L0_prev = L1_prev = L2_prev = L3_prev = 0
            else:
                L0_prev = L0[i - 1]
                L1_prev = L1[i - 1]
                L2_prev = L2[i - 1]
                L3_prev = L3[i - 1]

            L0[i] = (1 - gamma) * src.iloc[i] + gamma * L0_prev
            L1[i] = -gamma * L0[i] + L0_prev + gamma * L1_prev
            L2[i] = -gamma * L1[i] + L1_prev + gamma * L2_prev
            L3[i] = -gamma * L2[i] + L2_prev + gamma * L3_prev

            LagF[i] = (L0[i] + 2 * L1[i] + 2 * L2[i] + L3[i]) / 6

        return pd.Series(LagF, index=data.index)

    def tema(self, length=800):
        """Calculate the Triple Exponential Moving Average (TEMA)."""
        close = self.high_csv['close']
        # Calculate the first EMA
        ema1 = close.ewm(span=length, adjust=False).mean()
        # Calculate the second EMA
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        # Calculate the third EMA
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        # Calculate TEMA
        tema = 3 * (ema1 - ema2) + ema3
        return tema

    def hawkes_process(self, k=3):
        """Implement Hawkes process for volatility estimation."""
        kappa = k
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv['close'].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv['close'].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def adx(self, period=13):
        high = self.high_csv['high']
        low = self.high_csv['low']
        close = self.high_csv['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx

    def preprocessing(self):
        self.high_csv['HMA'] = self.hma(length=5)
        self.high_csv['Laguerre Filter'] = self.laguerre_filter(alpha=0.85)
        self.high_csv['TEMA'] = self.tema(length=500)  # Calculate TEMA and add to DataFrame
        hawkes_window_long = 20
        hawkes_window_short = 20
        long_percentile = 0.1
        short_percentile = 0.1
        self.high_csv['vol_hawkes'] = self.hawkes_process(k=3)
        self.high_csv['long_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window_long,
                                                                               min_periods=1).quantile(
            long_percentile)
        self.high_csv['short_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window_short,
                                                                                min_periods=1).quantile(
            short_percentile)
        self.high_csv['adx'] = self.adx(period=12)


    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1 or high_pointer < 1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        prev_close = self.high_csv['close'].iloc[high_pointer - 3:high_pointer].mean()
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 25:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['HMA'].iloc[high_pointer] > self.high_csv['Laguerre Filter'].iloc[high_pointer]
        cond2 = self.high_csv['HMA'].iloc[high_pointer - 1] <= self.high_csv['Laguerre Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55
        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is increasing
            if tema_curr > tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.12
                self.glob.sl = 0.06
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1 or high_pointer < 1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        prev_close = self.high_csv['close'].iloc[high_pointer - 3:high_pointer].mean()
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 25:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['HMA'].iloc[high_pointer] < self.high_csv['Laguerre Filter'].iloc[high_pointer]
        cond2 = self.high_csv['HMA'].iloc[high_pointer - 1] >= self.high_csv['Laguerre Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55
        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is decreasing
            if tema_curr < tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.12
                self.glob.sl = 0.06
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['long_threshold'].iloc[high_pointer]:
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['short_threshold'].iloc[high_pointer]:
            return 1
        return 0