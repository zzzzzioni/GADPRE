#########################################################
#                  total_pipeline notebook script        #
#########################################################

import argparse
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def model_predict(data: pd.DataFrame) -> None:
    df = pd.read_csv("./accident/afterpreprocessing.csv", index_col=0)
    df.dropna(inplace=True)

    x_data = df.iloc[:, 1:]
    y_data = df.iloc[:, 0]

    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, shuffle=True, random_state=30
    )

    model = LinearRegression()
    model_linear = model.fit(x_train, y_train)
    y_predict = model_linear.predict(x_test)

    _ = mean_squared_error(y_test, y_predict)
    _ = mean_absolute_error(y_test, y_predict)

    x = scaler.fit_transform(data)
    final = model_linear.predict(x)

    print("------------------예측-ECLO-지수------------------")
    for eclo, region in zip(final, ["선릉", "신논현", "신사", "압구정", "청담"]):
        print(region, ":", eclo)
    print("------------------------------------------------")


def web_scraping_job(url: str, location_name: str) -> str | None:
    chrome_service = ChromeService()
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    driver.get(url)
    driver.implicitly_wait(2)
    try:
        driver.maximize_window()
    except Exception:
        pass

    time.sleep(2)
    link = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "side_traffic"))
    )
    driver.execute_script("arguments[0].click();", link)

    time.sleep(2)
    average_speed_value_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//b[@class='rpt_roadSpd rpt_data']/span[contains(@class, 'color_')]")
        )
    )
    average_speed_html = average_speed_value_element.get_attribute("innerHTML")

    match = re.search(r"\d+\.\d+", average_speed_html)
    average_speed_value = match.group() if match else None

    driver.quit()
    return average_speed_value


def get_speed(region_list: list[str]) -> list[tuple[str, str | None]]:
    locations = [
        ("https://data.seoul.go.kr/SeoulRtd/?hotspotNm=%EC%84%A0%EB%A6%89%EC%97%AD", "선릉"),
        (
            "https://data.seoul.go.kr/SeoulRtd/?hotspotNm=%EC%8B%A0%EB%85%BC%ED%98%84%EC%97%AD%C2%B7%EB%85%BC%ED%98%84%EC%97%AD&x=37.5080801205745&y=127.02340583366124",
            "신논현",
        ),
        ("https://data.seoul.go.kr/SeoulRtd/?hotspotNm=%EA%B0%80%EB%A1%9C%EC%88%98%EA%B8%B8", "신사"),
        (
            "https://data.seoul.go.kr/SeoulRtd/?hotspotNm=%EC%95%95%EA%B5%AC%EC%A0%95%EB%A1%9C%EB%8D%B0%EC%98%A4%EA%B1%B0%EB%A6%AC&y=127.0386028&x=%EC%95%95%EA%B5%AC%EC%A0%95%EB%A1%9C%EB%8D%B0%EC%98%A4%EA%B1%B0%EB%A6%AC",
            "압구정",
        ),
        (
            "https://data.seoul.go.kr/SeoulRtd/?hotspotNm=%EC%B2%AD%EB%8B%B4%EB%8F%99%20%EB%AA%85%ED%92%88%EA%B1%B0%EB%A6%AC&y=127.04376509735859&x=%EC%B2%AD%EB%8B%B4%EB%8F%99%20%EB%AA%85%ED%92%88%EA%B1%B0%EB%A6%AC",
            "청담",
        ),
    ]

    data: list[tuple[str, str | None]] = []
    for url, name in locations:
        speed = web_scraping_job(url, name)
        data.append((name, speed))
    return data


def get_ppl_of(date: str, region: pd.DataFrame) -> pd.Series:
    """
    date 형식 : '2024-04-30 01:00:00'
    """
    return region[region["ds"] == date]["yhat"]


def _to_scalar(value) -> float:
    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError("인구 예측값이 비어 있습니다. date 값을 확인해주세요.")
        return float(value.iloc[0])
    return float(value)


def predict_ppl(region_list: list[str], date: str) -> list[tuple[str, float]]:
    ap_fin = pd.read_csv("압구정ppl.csv", index_col=0)
    ss_fin = pd.read_csv("신사ppl.csv", index_col=0)
    dc_fin = pd.read_csv("선릉ppl.csv", index_col=0)
    snh_fin = pd.read_csv("신논현ppl.csv", index_col=0)

    data: list[tuple[str, float]] = []
    # [선릉, 신논현, 신사, 압구정, 청담]
    for region, df in zip(region_list, [dc_fin, snh_fin, ss_fin, ap_fin, ap_fin]):
        ppl = _to_scalar(get_ppl_of(date, df))
        data.append((region, ppl))

    return data


def inference(date: str, week: str, tm: str) -> None:
    region_list = ["선릉", "신논현", "신사", "압구정", "청담"]

    ppl = predict_ppl(region_list, date)
    spd = get_speed(region_list)

    if week == "주중":
        weekday = 1
        weekend = 0
    else:
        weekday = 0
        weekend = 1

    if tm == "밤":
        night = 1
        dawn = 0
    else:
        night = 0
        dawn = 1

    df = pd.read_csv("./accident/afterpreprocessing.csv", index_col=0)

    df_new = pd.DataFrame(
        [
            [ppl[0][1] * 2, spd[0][1], weekday, weekend, 645.6666, dawn, night, 1, 0, 0, 0, 0],
            [ppl[1][1], spd[1][1], weekday, weekend, 443.3333, dawn, night, 0, 1, 0, 0, 0],
            [ppl[2][1] * 5, spd[2][1], weekday, weekend, 433.3333, dawn, night, 0, 0, 1, 0, 0],
            [ppl[3][1] * 8, spd[3][1], weekday, weekend, 562.3333, dawn, night, 0, 0, 0, 1, 0],
            [ppl[4][1] * 5, spd[4][1], weekday, weekend, 251.6666, dawn, night, 0, 0, 0, 0, 1],
        ],
        columns=df.columns[1:],
    )

    print(df_new)
    model_predict(df_new)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run total pipeline inference.")
    parser.add_argument("--date", default="2024-05-10 00:00:00", help="예: 2024-05-10 00:00:00")
    parser.add_argument("--week", default="주중", choices=["주중", "주말"], help="주중/주말")
    parser.add_argument("--time", default="밤", choices=["밤", "새벽"], help="밤/새벽")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(args.date, args.week, args.time)
