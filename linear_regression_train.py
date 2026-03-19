import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_default_inference_input(columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [244.108615, 23.2, 1, 0, 645.6666, 0, 1, 1, 0, 0, 0, 0],
            [425.427505, 18.94, 1, 0, 443.3333, 0, 1, 0, 1, 0, 0, 0],
            [174.27562999999998, 20.83, 1, 0, 433.3333, 0, 1, 0, 0, 1, 0, 0],
            [488.055904, 21.14, 1, 0, 562.3333, 0, 1, 0, 0, 0, 1, 0],
            [224.4384, 19.53, 1, 0, 251.6666, 0, 1, 0, 0, 0, 0, 1],
        ],
        columns=columns,
    )


def run_training(data_csv: Path, output_dir: Path, random_state: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv, index_col=0)
    df.dropna(inplace=True)

    x_data = df.iloc[:, 1:]
    y_data = df.iloc[:, 0]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_data, test_size=0.2, shuffle=True, random_state=random_state
    )

    model = LinearRegression()
    model_linear = model.fit(x_train, y_train)
    y_predict = model_linear.predict(x_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "random_state": random_state,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    test_pred_df = pd.DataFrame({"y_test": y_test.values, "y_predict": y_predict})
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    df_new = build_default_inference_input(df.columns[1:])
    x_new = scaler.transform(df_new)
    new_pred = model_linear.predict(x_new)
    new_pred_df = pd.DataFrame(
        {"region": ["선릉", "신논현", "신사", "압구정", "청담"], "predicted_ECLO": new_pred}
    )
    new_pred_df.to_csv(output_dir / "inference_predictions.csv", index=False)
    df_new.to_csv(output_dir / "inference_input.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_predict, alpha=0.6)
    mn = min(float(y_test.min()), float(y_predict.min()))
    mx = max(float(y_test.max()), float(y_predict.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.xlabel("Actual ECLO")
    plt.ylabel("Predicted ECLO")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Linear regression training completed.\n")
        f.write(f"Input data: {data_csv}\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write("Saved files:\n")
        f.write("- metrics.json\n")
        f.write("- test_predictions.csv\n")
        f.write("- inference_input.csv\n")
        f.write("- inference_predictions.csv\n")
        f.write("- actual_vs_predicted.png\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear regression and save outputs to a folder."
    )
    parser.add_argument(
        "--data-csv",
        default="accident/afterpreprocessing.csv",
        help="학습 데이터 CSV 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="linear_regression_outputs",
        help="출력 파일 저장 폴더",
    )
    parser.add_argument("--random-state", type=int, default=18591, help="train_test_split seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_csv=Path(args.data_csv),
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
    )
