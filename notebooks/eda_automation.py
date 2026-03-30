import pandas as pd
import matplotlib.pyplot as plt


def prepare_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    stock_df = df.copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df["Close"] = pd.to_numeric(stock_df["Close"], errors="coerce")
    stock_df = stock_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    stock_df["Daily Return"] = stock_df["Close"].pct_change()
    return stock_df


def plot_normalized_price(df: pd.DataFrame, stock_name: str = "Stock", save_path=None) -> None:
    stock_df = prepare_stock_data(df)
    stock_df["Normalized Price"] = stock_df["Close"] / stock_df["Close"].iloc[0] * 100

    plt.figure(figsize=(12, 6))
    plt.plot(stock_df["Date"], stock_df["Normalized Price"])
    plt.title(f"{stock_name} Normalized Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base = 100)")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_daily_returns(df: pd.DataFrame, stock_name: str = "Stock", save_path=None) -> None:
    stock_df = prepare_stock_data(df)

    plt.figure(figsize=(12, 6))
    plt.plot(stock_df["Date"], stock_df["Daily Return"])
    plt.title(f"{stock_name} Daily Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rolling_volatility(df: pd.DataFrame, stock_name: str = "Stock", window: int = 21, save_path=None) -> None:
    stock_df = prepare_stock_data(df)
    stock_df["Rolling Volatility"] = stock_df["Daily Return"].rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_df["Date"], stock_df["Rolling Volatility"])
    plt.title(f"{stock_name} {window}-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_return_distribution(df: pd.DataFrame, stock_name: str = "Stock", bins: int = 50, save_path=None) -> None:
    stock_df = prepare_stock_data(df)

    plt.figure(figsize=(10, 6))
    plt.hist(stock_df["Daily Return"].dropna(), bins=bins)
    plt.title(f"{stock_name} Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()