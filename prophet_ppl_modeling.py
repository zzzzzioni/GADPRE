import argparse
from pathlib import Path

import pandas as pd
from prophet import Prophet


REGION_MAP = {
    "압구정": ["Apgujeong-dong"],
    "신사": ["Sinsa-dong"],
    "선릉": ["Daechi4-dong"],
    "신논현": ["Yeoksam1-dong", "Nonhyeon1-dong"],
}


def weeken(ds) -> int:
    date = pd.to_datetime(ds)
    return 1 if date.weekday() in (5, 6) else 0


def make_test_df(region: pd.DataFrame) -> pd.DataFrame:
    temp = region.groupby("datetime", as_index=False)["방문자수"].sum()
    temp.columns = ["ds", "y"]
    temp["weeken"] = temp["ds"].apply(weeken)
    return temp


def how_many_ppl_in(region: pd.DataFrame, periods: int, freq: str) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    region = region.copy()
    region["cap"] = 5000
    region["floor"] = 100

    m = Prophet(
        growth="logistic",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=1,
    )
    m.add_country_holidays(country_name="KR")
    m.add_seasonality(name="day", period=24, fourier_order=1)
    m.add_regressor("weeken")

    m.fit(region)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    future["weeken"] = future["ds"].apply(weeken)
    future["cap"] = 5000
    future["floor"] = 10

    raw_pred = m.predict(future)
    pred = raw_pred[["ds", "yhat"]]
    return m, pred, raw_pred


def pick_clusters(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if {"region", "time", "방문자수"}.issubset(df.columns):
        data = df.copy()
        data["datetime"] = pd.to_datetime(data["time"])
        return {
            "압구정": data[data["region"] == "압구정"],
            "신사": data[data["region"] == "신사"],
            "선릉": data[data["region"] == "선릉"],
            "신논현": data[data["region"] == "신논현"],
        }

    required = {"행정동", "datetime", "방문자수"}
    if not required.issubset(df.columns):
        raise ValueError(
            "입력 CSV에 필요한 컬럼이 없습니다. "
            "지원 스키마: [region,time,방문자수] 또는 [행정동,datetime,방문자수]"
        )

    data = df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    return {
        "압구정": data[data["행정동"].isin(REGION_MAP["압구정"])],
        "신사": data[data["행정동"].isin(REGION_MAP["신사"])],
        "선릉": data[data["행정동"].isin(REGION_MAP["선릉"])],
        "신논현": data[data["행정동"].isin(REGION_MAP["신논현"])],
    }


def run_prophet_modeling(input_csv: Path, output_dir: Path, periods: int, freq: str, query_date: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    clusters = pick_clusters(df)

    results: dict[str, pd.DataFrame] = {}
    for cluster_name, region_df in clusters.items():
        if region_df.empty:
            raise ValueError(f"{cluster_name} 데이터가 비어 있습니다. 입력 데이터를 확인해주세요.")
        region_ts = make_test_df(region_df)
        _, pred, _ = how_many_ppl_in(region_ts, periods=periods, freq=freq)
        results[cluster_name] = pred

    save_names = {
        "압구정": "압구정ppl.csv",
        "신사": "신사ppl.csv",
        "선릉": "선릉ppl.csv",
        "신논현": "신논현ppl.csv",
    }
    for cluster_name, pred_df in results.items():
        pred_df.to_csv(output_dir / save_names[cluster_name], encoding="utf-8")

    print(f"완료: {output_dir} 에 4개 예측 파일 저장")

    if query_date:
        print(f"\n[조회 시각: {query_date}]")
        for cluster_name in ["선릉", "신논현", "신사", "압구정"]:
            val = results[cluster_name][results[cluster_name]["ds"] == query_date]["yhat"]
            print(cluster_name, ":", val if not val.empty else "해당 시각 없음")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prophet 기반 유동인구 모델링 분리 스크립트")
    parser.add_argument(
        "--input-csv",
        default="New Folder With Items/pplfinal.csv",
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="GADPRE/prophet_outputs",
        help="예측 결과 저장 폴더",
    )
    parser.add_argument("--periods", type=int, default=2000, help="미래 예측 스텝 수")
    parser.add_argument("--freq", default="H", help="예측 주기 (예: H)")
    parser.add_argument(
        "--query-date",
        default=None,
        help="특정 시각 yhat 조회 (예: 2024-05-10 00:00:00)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prophet_modeling(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        periods=args.periods,
        freq=args.freq,
        query_date=args.query_date,
    )
