import os

import click
import pandas as pd
import pandas_datareader as pdr


FEATURES_PATH = "data.csv"
STOCK_INDEX = "^GSPC"


def get_raw_data(index_name, retry_attempts=3):
    if index_name:
        while retry_attempts > 0:
            try:
                df = pdr.get_data_yahoo(index_name)
                retry_attempts = 0
                return df
            except:
                print(
                    "Data pull failed. {} retry attempts remaining".format(
                        retry_attempts
                    )
                )
                retry_attempts = retry_attempts - 1
    else:
        print("Invalid usage. Parameter index_name is required")
    return None


@click.command()
@click.option("--out_dir")
def main(out_dir: str) -> None:
    # get historical data
    hist_data = pd.read_csv("/data/GSPC_2000_2019.csv")
    hist_data["Date"] = pd.to_datetime(hist_data["Date"])

    # download new data
    sp_df = get_raw_data(STOCK_INDEX)

    # reset index to get date_time as a column
    sp_df = sp_df.reset_index()

    # prepare the required dataframe
    sp_df.rename(columns={"index": "Date"}, inplace=True)
    sp_df = sp_df[["Date", "Close"]]
    data = hist_data.append(
        sp_df.loc[sp_df["Date"] > hist_data["Date"].values[-1]]
    )

    os.makedirs(out_dir, exist_ok=True)
    data.to_csv(os.path.join(out_dir, FEATURES_PATH), index=False)


if __name__ == "__main__":
    main()
