import pandas as pd
import numpy as np
import talib as ta
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from datetime import datetime
import json
import os


def create_run_directory():
    """Create a new directory for this optimization run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"reports/run_{timestamp}"
    subdirs = ["plots", "individual_reports", "parameters"]

    for d in [run_dir] + [f"{run_dir}/{sub}" for sub in subdirs]:
        os.makedirs(d, exist_ok=True)

    return run_dir


def save_optimization_report(report, indicator_name, run_dir):
    """Save optimization report to file"""
    filename = f"{run_dir}/individual_reports/{indicator_name}_report.txt"

    with open(filename, "w") as f:
        f.write(report)

    return filename


def calculate_indicators(df, params=None):
    """Calculate technical indicators with optional parameter optimization"""
    if params is None:
        params = {
            # Original indicators
            "ROC": {"timeperiod": 10},
            "PSAR": {"acceleration": 0.02, "maximum": 0.2},
            "BBANDS": {"timeperiod": 20, "nbdev": 2},
            "RSI": {"timeperiod": 14},
            "ATR": {"timeperiod": 14},
            "NATR": {"timeperiod": 14},
            "MFI": {"timeperiod": 14},
            "CMF": {"fastperiod": 3, "slowperiod": 10},
            # New indicators
            "DMI": {"timeperiod": 14, "adx_period": 14},
            "PPO": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
            "CCI": {"timeperiod": 14},
            "KELTNER": {"timeperiod": 20, "atr_period": 10, "multiplier": 2.0},
            "VWAP": {"timeperiod": 14},
            "RVI": {"timeperiod": 10},
            "TRIX": {"timeperiod": 15},
            "DPO": {"timeperiod": 20},
        }

    # Original Indicators
    df["roc"] = ta.ROC(df["close"], timeperiod=params["ROC"]["timeperiod"])
    df["psar"] = ta.SAR(
        df["high"],
        df["low"],
        acceleration=params["PSAR"]["acceleration"],
        maximum=params["PSAR"]["maximum"],
    )

    upper, middle, lower = ta.BBANDS(
        df["close"],
        timeperiod=params["BBANDS"]["timeperiod"],
        nbdevup=params["BBANDS"]["nbdev"],
        nbdevdn=params["BBANDS"]["nbdev"],
    )
    df["bb_b"] = (df["close"] - lower) / (upper - lower)
    df["rsi"] = ta.RSI(df["close"], timeperiod=params["RSI"]["timeperiod"])

    df["atr"] = ta.ATR(
        df["high"], df["low"], df["close"], timeperiod=params["ATR"]["timeperiod"]
    )
    df["natr"] = ta.NATR(
        df["high"], df["low"], df["close"], timeperiod=params["NATR"]["timeperiod"]
    )

    df["mfi"] = ta.MFI(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        timeperiod=params["MFI"]["timeperiod"],
    )
    df["cmf"] = ta.ADOSC(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        fastperiod=params["CMF"]["fastperiod"],
        slowperiod=params["CMF"]["slowperiod"],
    )

    # New Trend/Momentum Indicators
    plus_di = ta.PLUS_DI(
        df["high"], df["low"], df["close"], timeperiod=params["DMI"]["timeperiod"]
    )
    minus_di = ta.MINUS_DI(
        df["high"], df["low"], df["close"], timeperiod=params["DMI"]["timeperiod"]
    )
    df["dmi"] = plus_di - minus_di
    df["adx"] = ta.ADX(
        df["high"], df["low"], df["close"], timeperiod=params["DMI"]["adx_period"]
    )

    df["ppo"] = ta.PPO(
        df["close"],
        fastperiod=params["PPO"]["fastperiod"],
        slowperiod=params["PPO"]["slowperiod"],
        matype=params["PPO"]["matype"],
    )

    df["trix"] = ta.TRIX(df["close"], timeperiod=params["TRIX"]["timeperiod"])

    # Mean Reversion Indicators
    df["cci"] = ta.CCI(
        df["high"], df["low"], df["close"], timeperiod=params["CCI"]["timeperiod"]
    )

    def calculate_dpo(series, period):
        shifted_ma = ta.SMA(series, timeperiod=period).shift(period // 2 + 1)
        return series - shifted_ma

    df["dpo"] = calculate_dpo(df["close"], params["DPO"]["timeperiod"])

    # Enhanced Volatility Indicators
    def calculate_keltner(df, period, atr_period, multiplier):
        middle = ta.EMA(df["close"], timeperiod=period)
        atr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=atr_period)
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        return (df["close"] - lower) / (upper - lower)

    df["keltner"] = calculate_keltner(
        df,
        params["KELTNER"]["timeperiod"],
        params["KELTNER"]["atr_period"],
        params["KELTNER"]["multiplier"],
    )

    def calculate_rvi(series, period):
        delta = series.diff()
        up = ta.SMA(delta.where(delta > 0, 0), timeperiod=period)
        down = ta.SMA((-delta).where(delta < 0, 0), timeperiod=period)
        return (up / (up + down) * 100) if (up + down).any() else 0

    df["rvi"] = calculate_rvi(df["close"], params["RVI"]["timeperiod"])

    def calculate_vwap(df, period):
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).rolling(period).sum() / df[
            "volume"
        ].rolling(period).sum()
        return (df["close"] - vwap) / df["close"]

    df["vwap"] = calculate_vwap(df, params["VWAP"]["timeperiod"])

    return df


def calculate_performance_metrics(df, indicator, close="close", returns="returns"):
    """Calculate performance metrics for an indicator"""
    metrics = {}

    # Original metrics
    metrics["close_corr"] = df[indicator].corr(df[close])
    metrics["returns_corr"] = df[indicator].corr(df[returns])
    metrics["IC"] = df[indicator].corr(df[returns].shift(-1), method="spearman")
    metrics["signal_noise_ratio"] = abs(df[indicator].mean() / df[indicator].std())

    ac_lags = [1, 5, 10, 20]
    for lag in ac_lags:
        metrics[f"ac_lag_{lag}"] = df[indicator].autocorr(lag=lag)

    hurst = calculate_hurst_exponent(df[indicator])
    metrics["hurst_exponent"] = hurst if hurst is not None else 0.5

    metrics["volatility"] = df[indicator].std()
    metrics["skewness"] = df[indicator].skew()
    metrics["kurtosis"] = df[indicator].kurtosis()

    # Additional metrics for specific indicator types
    if indicator in ["dmi", "ppo", "trix"]:
        up_trend = df[returns] > 0
        metrics["trend_capture"] = ((df[indicator].shift(1) > 0) & up_trend).mean()
        metrics["trend_persistence"] = (
            np.sign(df[indicator]) == np.sign(df[indicator].shift(1))
        ).mean()

    if indicator in ["cci", "dpo"]:
        metrics["mean_rev_efficiency"] = -df[indicator].corr(df[indicator].shift(1))
        extreme_threshold = 2
        extreme_signals = abs(df[indicator]) > (df[indicator].std() * extreme_threshold)
        metrics["extreme_accuracy"] = (
            extreme_signals & (np.sign(df[returns].shift(-1)) != np.sign(df[indicator]))
        ).mean()

    if indicator in ["keltner", "rvi"]:
        realized_vol = df[returns].rolling(5).std()
        metrics["vol_forecast_corr"] = df[indicator].corr(realized_vol)
        high_vol = realized_vol > realized_vol.median()
        metrics["vol_regime_accuracy"] = (
            (df[indicator] > df[indicator].median()) == high_vol
        ).mean()

    return pd.Series(metrics)


def calculate_hurst_exponent(series, lags=None):
    """
    Calculate Hurst exponent for time series

    Args:
    - series: time series data
    - lags: list of lags to use (default: 2 to min(100, len(series)/4))

    Returns:
    - Hurst exponent or None if calculation fails
    """
    try:
        # Remove NaN values
        series = series.dropna()

        # Determine appropriate lags
        if lags is None:
            max_lag = min(100, len(series) // 4)
            lags = range(2, max_lag)

        # Calculate tau for each lag
        tau = []
        for lag in lags:
            # Price differences
            price_diff = np.subtract(series[lag:].values, series[:-lag].values)
            # Avoid zero variance
            if np.var(price_diff) > 0:
                tau.append(np.std(price_diff))
            else:
                return None

        # Avoid empty tau list
        if not tau or len(tau) < 2:
            return None

        # Convert to numpy arrays and calculate logs
        lags_array = np.array(lags)
        tau_array = np.array(tau)

        # Avoid zero or negative values in log
        valid_mask = (tau_array > 0) & (lags_array > 0)
        if not np.any(valid_mask):
            return None

        # Calculate Hurst exponent
        reg = np.polyfit(
            np.log(lags_array[valid_mask]), np.log(tau_array[valid_mask]), 1
        )

        return reg[0] / 2  # Hurst = slope/2

    except Exception as e:
        print(f"Error calculating Hurst exponent: {str(e)}")
        return None


def optimize_indicator_params(df, indicator_name, param_grid, metric="close_corr"):
    """Optimize indicator parameters based on specified metric"""
    results = []
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    if not param_combinations:
        raise ValueError(f"No valid parameter combinations for {indicator_name}")

    for params in param_combinations:
        try:
            df_temp = df.copy()
            # Create a full parameters dictionary with default values
            full_params = {
                # Original indicators
                "ROC": {"timeperiod": 10},
                "PSAR": {"acceleration": 0.02, "maximum": 0.2},
                "BBANDS": {"timeperiod": 20, "nbdev": 2},
                "RSI": {"timeperiod": 14},
                "ATR": {"timeperiod": 14},
                "NATR": {"timeperiod": 14},
                "MFI": {"timeperiod": 14},
                "CMF": {"fastperiod": 3, "slowperiod": 10},
                # New indicators
                "DMI": {"timeperiod": 14, "adx_period": 14},
                "PPO": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
                "CCI": {"timeperiod": 14},
                "KELTNER": {"timeperiod": 20, "atr_period": 10, "multiplier": 2.0},
                "VWAP": {"timeperiod": 14},
                "RVI": {"timeperiod": 10},
                "TRIX": {"timeperiod": 15},
                "DPO": {"timeperiod": 20},
            }

            # Update with current parameters
            full_params[indicator_name].update(params)

            # Calculate indicators with current parameters
            df_temp = calculate_indicators(df_temp, full_params)

            # Map indicator names to column names
            indicator_col = {
                # Original indicators
                "ROC": "roc",
                "RSI": "rsi",
                "BBANDS": "bb_b",
                "PSAR": "psar",
                "ATR": "atr",
                "NATR": "natr",
                "MFI": "mfi",
                "CMF": "cmf",
                # New indicators
                "DMI": "dmi",
                "PPO": "ppo",
                "CCI": "cci",
                "KELTNER": "keltner",
                "VWAP": "vwap",
                "RVI": "rvi",
                "TRIX": "trix",
                "DPO": "dpo",
            }[indicator_name]

            # Calculate metrics
            metrics = calculate_performance_metrics(df_temp, indicator_col)

            # Convert params to string for DataFrame indexing
            param_str = "_".join(f"{k}:{v}" for k, v in params.items())

            # Store results
            results.append(
                {
                    "param_str": param_str,
                    "params": params,
                    "metrics": metrics,
                    "score": float(metrics[metric]),
                }
            )

        except Exception as e:
            print(f"Error with {indicator_name} params {params}: {str(e)}")
            continue

    if not results:
        raise ValueError(f"No valid results found for {indicator_name}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).set_index("param_str")

    # Find best parameters
    best_idx = results_df["score"].abs().idxmax()
    best_params = results_df.loc[best_idx, "params"]
    best_metrics = results_df.loc[best_idx, "metrics"]

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_results": results_df.reset_index(),
    }


def plot_optimization_results(results, indicator_name, run_dir):
    """Plot optimization results for an indicator and save to file"""
    plt.figure(figsize=(15, 10))

    # Get the number of unique parameters
    params = results["all_results"]["params"].iloc[0].keys()
    n_params = len(params)

    if n_params > 1:
        param_data = pd.DataFrame([p for p in results["all_results"]["params"]])
        scores = results["all_results"]["score"]

        if n_params == 2:
            param_names = list(params)
            pivot_table = pd.pivot_table(
                data=pd.concat([param_data, scores], axis=1),
                values="score",
                index=param_names[0],
                columns=param_names[1],
            )

            plt.subplot(221)
            sns.heatmap(pivot_table, annot=True, cmap="RdBu", center=0)
            plt.title(f"{indicator_name} Parameter Performance")
        else:
            plt.subplot(221)
            param_data_normalized = (param_data - param_data.min()) / (
                param_data.max() - param_data.min()
            )
            for i in range(len(param_data)):
                plt.plot(
                    list(params), param_data_normalized.iloc[i], alpha=0.5, color="blue"
                )
            plt.xticks(rotation=45)
            plt.title("Parameter Combinations")
    else:
        plt.subplot(221)
        param_name = list(params)[0]
        param_values = [p[param_name] for p in results["all_results"]["params"]]
        plt.plot(param_values, results["all_results"]["score"])
        plt.xlabel(param_name)
        plt.ylabel("Performance Score")
        plt.title(f"{indicator_name} Parameter Performance")

    # Performance metrics distribution
    plt.subplot(222)
    metrics_df = pd.DataFrame([r for r in results["all_results"]["metrics"]])
    metrics_df.boxplot()
    plt.xticks(rotation=45)
    plt.title("Performance Metrics Distribution")

    # Autocorrelation decay
    plt.subplot(223)
    ac_cols = [col for col in metrics_df.columns if "ac_lag" in col]
    lags = [int(col.split("_")[-1]) for col in ac_cols]
    ac_values = metrics_df[ac_cols].mean()
    plt.plot(lags, ac_values)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Average Autocorrelation Decay")

    plt.tight_layout()

    # Save plot
    filename = f"{run_dir}/plots/{indicator_name}_optimization.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    return filename


def generate_optimization_report(results, indicator_name):
    """Generate detailed report of optimization results"""
    report = [f"\n=== Optimization Report for {indicator_name} ===\n"]

    report.append("Best Parameters:")
    for param, value in results["best_params"].items():
        report.append(f"{param}: {value}")

    report.append("\nPerformance Metrics:")
    metrics = results["best_metrics"]

    metric_descriptions = {
        "close_corr": "Correlation with closing price",
        "returns_corr": "Correlation with returns",
        "IC": "Information Coefficient (predictive power)",
        "signal_noise_ratio": "Signal-to-Noise Ratio",
        "hurst_exponent": "Hurst Exponent (>0.5: trending, <0.5: mean-reverting)",
        "volatility": "Indicator Volatility",
        "skewness": "Distribution Skewness",
        "kurtosis": "Distribution Kurtosis",
    }

    for metric, value in metrics.items():
        if metric in metric_descriptions:
            report.append(f"{metric} ({metric_descriptions[metric]}): {value:.4f}")
        else:
            report.append(f"{metric}: {value:.4f}")

    report.append("\nParameter Sensitivity Analysis:")
    sensitivity = (
        results["all_results"]
        .groupby("param_str")["score"]
        .agg(["mean", "std"])
        .round(4)
    )
    report.append(str(sensitivity))

    return "\n".join(report)


if __name__ == "__main__":

    run_dir = create_run_directory()

    # Load data
    df = pd.read_parquet(
        "~/work/solvendo/optjax/ohlcvf_plus_indicators/btc_futures_data.parquet"
    )
    df["returns"] = df["close"].pct_change()

    # Define parameter grids for optimization
    param_grids = {
        # Original indicators
        "ROC": {"timeperiod": list(range(5, 30, 5))},
        "RSI": {"timeperiod": list(range(10, 30, 5))},
        "BBANDS": {"timeperiod": list(range(10, 30, 5)), "nbdev": [1.5, 2.0, 2.5]},
        "PSAR": {"acceleration": [0.02, 0.025, 0.03], "maximum": [0.2, 0.25, 0.3]},
        "ATR": {"timeperiod": list(range(10, 30, 5))},
        "NATR": {"timeperiod": list(range(10, 30, 5))},
        "MFI": {"timeperiod": list(range(10, 30, 5))},
        "CMF": {"fastperiod": [2, 3, 4], "slowperiod": [8, 10, 12]},
        # New indicators
        "DMI": {
            "timeperiod": list(range(10, 30, 5)),
            "adx_period": list(range(10, 30, 5)),
        },
        "PPO": {
            "fastperiod": list(range(5, 15, 3)),
            "slowperiod": list(range(20, 40, 5)),
            "matype": [0, 1],
        },
        "CCI": {"timeperiod": list(range(10, 30, 5))},
        "KELTNER": {
            "timeperiod": list(range(10, 30, 5)),
            "atr_period": list(range(10, 30, 5)),
            "multiplier": [1.5, 2.0, 2.5],
        },
        "RVI": {"timeperiod": list(range(10, 30, 5))},
        "TRIX": {"timeperiod": list(range(10, 30, 5))},
        "DPO": {"timeperiod": list(range(10, 30, 5))},
        "VWAP": {"timeperiod": list(range(10, 30, 5))},
    }

    # Create summary report
    summary_report = ["=== Optimization Summary ===\n"]

    # Optimize each indicator
    optimized_results = {}
    for indicator, param_grid in param_grids.items():
        print(f"\nOptimizing {indicator}...")
        try:
            results = optimize_indicator_params(df, indicator, param_grid)
            optimized_results[indicator] = results

            # Generate and save plot
            plot_filename = plot_optimization_results(results, indicator, run_dir)

            # Generate and save report
            report = generate_optimization_report(results, indicator)
            report_filename = save_optimization_report(report, indicator, run_dir)

            # Add to summary
            summary_report.append(f"\n{indicator}:")
            summary_report.append(f"Plot: {plot_filename}")
            summary_report.append(f"Report: {report_filename}")
            summary_report.append(f"Best parameters: {results['best_params']}")
            summary_report.append(
                f"Best score: {results['all_results']['score'].max():.4f}\n"
            )

        except Exception as e:
            print(f"Failed to optimize {indicator}: {str(e)}")
            summary_report.append(f"\n{indicator}: Optimization failed - {str(e)}")
            continue

    # Save summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"reports/optimization_summary_{timestamp}.txt"
    with open(summary_filename, "w") as f:
        f.write("\n".join(summary_report))

    print(f"\nOptimization complete. Summary saved to {summary_filename}")

    # Calculate indicators with optimized parameters
    if optimized_results:
        best_params = {
            ind: res["best_params"] for ind, res in optimized_results.items()
        }
        df_optimized = calculate_indicators(df, best_params)

        # Save optimized parameters
        params_filename = f"reports/optimized_parameters_{timestamp}.json"
        with open(params_filename, "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"Optimized parameters saved to {params_filename}")
    else:
        print("\nNo indicators were successfully optimized.")
