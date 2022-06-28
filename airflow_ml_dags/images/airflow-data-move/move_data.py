import os

import click
import shutil


FEATURES_PATH = "data.csv"


@click.command()
@click.option("--in_dir")
@click.option("--out_dir")
def main(in_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    shutil.move(
        os.path.join(in_dir, FEATURES_PATH),
        os.path.join(out_dir, FEATURES_PATH),
    )


if __name__ == "__main__":
    main()
