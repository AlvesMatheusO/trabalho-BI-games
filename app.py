from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "vgchartz-2024.csv"
HTML_PATH = BASE_DIR / "index.html"

NUMERIC_COLS = [
    "total_sales",
    "critic_score",
    "na_sales",
    "pal_sales",
    "jp_sales",
    "other_sales",
]

REQUIRED_COLS = [
    "title",
    "console",
    "genre",
    "publisher",
    "developer",
    "critic_score",
    "total_sales",
    "na_sales",
    "jp_sales",
    "pal_sales",
    "other_sales",
]


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return 0
        return float(value)
    if pd.isna(value):
        return None
    return value


def load_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["title", "console", "genre", "publisher", "developer"]:
        df[col] = df[col].fillna("Não informado").astype(str)

    genre_mean = df.groupby("genre")["critic_score"].transform("mean")
    global_mean = df["critic_score"].mean()
    df["critic_score"] = df["critic_score"].fillna(genre_mean).fillna(global_mean)

    for col in ["total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]:
        df[col] = df[col].fillna(0)

    return df


df = load_dataset()


@app.route("/")
def home():
    return send_file(HTML_PATH)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "rows": int(len(df)), "csv": CSV_PATH.name})


@app.route("/api/genres")
def genres():
    values = df["genre"].dropna().sort_values().unique().tolist()
    return jsonify(values)


def filter_by_genre(genre):
    if genre and genre != "all":
        return df[df["genre"].str.lower() == genre.lower()].copy()
    return df.copy()


def top_sum(dataframe, group_col, value_col, limit=10):
    data = dataframe.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(limit).round(2)
    return {str(k): float(v) for k, v in data.items()}


def top_mean(dataframe, group_col, value_col, limit=10):
    data = dataframe.groupby(group_col)[value_col].mean().sort_values(ascending=False).head(limit).round(2)
    return {str(k): float(v) for k, v in data.items()}


def regional_distribution(dataframe):
    regions = {
        "América do Norte": float(dataframe["na_sales"].sum()),
        "Europa/PAL": float(dataframe["pal_sales"].sum()),
        "Japão": float(dataframe["jp_sales"].sum()),
        "Outros": float(dataframe["other_sales"].sum()),
    }
    total = sum(regions.values())
    if total <= 0:
        return {key: 0 for key in regions}
    return {key: round((value / total) * 100, 2) for key, value in regions.items()}


def critic_histogram(dataframe):
    bins = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    values = [
        int(((dataframe["critic_score"] >= 0) & (dataframe["critic_score"] < 2)).sum()),
        int(((dataframe["critic_score"] >= 2) & (dataframe["critic_score"] < 4)).sum()),
        int(((dataframe["critic_score"] >= 4) & (dataframe["critic_score"] < 6)).sum()),
        int(((dataframe["critic_score"] >= 6) & (dataframe["critic_score"] < 8)).sum()),
        int(((dataframe["critic_score"] >= 8) & (dataframe["critic_score"] <= 10)).sum()),
    ]
    return {"labels": bins, "values": values}


def strategic_quadrant(dataframe):
    grouped = (
        dataframe.groupby("genre")
        .agg(total_sales=("total_sales", "sum"), avg_critic=("critic_score", "mean"), games=("title", "count"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(12)
    )
    result = []
    for _, row in grouped.iterrows():
        result.append({
            "x": round(float(row["avg_critic"]), 2),
            "y": round(float(row["total_sales"]), 2),
            "z": int(row["games"]),
            "name": row["genre"],
        })
    return result


def radar_strategy(dataframe):
    grouped = (
        dataframe.groupby("genre")
        .agg(
            total_sales=("total_sales", "sum"),
            avg_critic=("critic_score", "mean"),
            games=("title", "count"),
            na=("na_sales", "sum"),
            pal=("pal_sales", "sum"),
            jp=("jp_sales", "sum"),
            other=("other_sales", "sum"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(5)
    )
    if grouped.empty:
        return {"categories": [], "series": []}
    max_sales = grouped["total_sales"].max()
    max_games = grouped["games"].max()
    series = []
    for _, row in grouped.iterrows():
        global_presence = (int(row["na"] > 0) + int(row["pal"] > 0) + int(row["jp"] > 0) + int(row["other"] > 0)) / 4 * 100
        sales_score = (row["total_sales"] / max_sales * 100) if max_sales > 0 else 0
        critic_score = (row["avg_critic"] / 10 * 100) if row["avg_critic"] > 0 else 0
        volume_score = (row["games"] / max_games * 100) if max_games > 0 else 0
        opportunity = critic_score * 0.35 + sales_score * 0.35 + global_presence * 0.2 + (100 - volume_score) * 0.1
        series.append({
            "name": row["genre"],
            "data": [round(float(sales_score), 2), round(float(critic_score), 2), round(float(global_presence), 2), round(float(volume_score), 2), round(float(opportunity), 2)],
        })
    return {"categories": ["Vendas", "Crítica", "Presença Global", "Volume de Jogos", "Oportunidade"], "series": series}


def publisher_ranking(dataframe):
    data = (
        dataframe.groupby("publisher")
        .agg(total_sales=("total_sales", "sum"), avg_critic=("critic_score", "mean"), games=("title", "count"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(10)
    )
    return {
        "labels": data["publisher"].tolist(),
        "sales": data["total_sales"].round(2).tolist(),
        "critic": data["avg_critic"].round(2).tolist(),
        "games": data["games"].astype(int).tolist(),
    }


def console_market_share(dataframe):
    top = dataframe.groupby("console")["total_sales"].sum().sort_values(ascending=False).head(8)
    return {"labels": top.index.astype(str).tolist(), "values": top.round(2).tolist()}


def scatter_points(dataframe):
    valid = (
        dataframe[["critic_score", "total_sales", "title", "genre"]]
        .dropna()
        .query("critic_score >= 0 and total_sales >= 0")
        .sort_values("total_sales", ascending=False)
        .head(1200)
    )
    return valid.to_dict(orient="records")


def generate_insights(dataframe, genre):
    if dataframe.empty:
        return ["Nenhum dado encontrado para o filtro selecionado."]
    top_genre = dataframe.groupby("genre")["total_sales"].sum().sort_values(ascending=False).head(1)
    top_console = dataframe.groupby("console")["total_sales"].sum().sort_values(ascending=False).head(1)
    top_publisher = dataframe.groupby("publisher")["total_sales"].sum().sort_values(ascending=False).head(1)
    avg_critic = dataframe["critic_score"].mean()
    total_sales = dataframe["total_sales"].sum()
    total_games = len(dataframe)
    insights = []
    if genre == "all" and not top_genre.empty:
        insights.append(f"O gênero com maior volume de vendas é {top_genre.index[0]}, com {top_genre.iloc[0]:.1f}M em vendas.")
    if not top_console.empty:
        insights.append(f"O console mais forte no filtro atual é {top_console.index[0]}, somando {top_console.iloc[0]:.1f}M em vendas.")
    if not top_publisher.empty:
        insights.append(f"A publisher líder é {top_publisher.index[0]}, indicando uma possível referência estratégica para posicionamento.")
    if avg_critic >= 7.5:
        insights.append("A nota crítica média está em um nível saudável. Isso sugere boa aceitação qualitativa do catálogo analisado.")
    else:
        insights.append("A nota crítica média está abaixo de 7.5. Estratégias de melhoria de qualidade e polimento podem ser importantes.")
    if total_games > 0:
        sales_per_game = total_sales / total_games
        if sales_per_game >= 1:
            insights.append(f"A média de vendas por jogo é de {sales_per_game:.2f}M, indicando alto potencial comercial por título.")
        else:
            insights.append(f"A média de vendas por jogo é de {sales_per_game:.2f}M, sugerindo mercado mais pulverizado ou nichado.")
    return insights


@app.route("/api/summary")
def summary():
    genre = request.args.get("genre", "all")
    filtered = filter_by_genre(genre)
    if filtered.empty:
        return jsonify({
            "globalMetrics": {"totalGames": 0, "totalSales": 0, "avgCritic": 0, "uniqueConsoles": 0},
            "topConsoles": {}, "topGenres": {}, "topPublishers": {"labels": [], "sales": [], "critic": [], "games": []},
            "regional": {}, "marketShare": {"labels": [], "values": []}, "criticHistogram": {"labels": [], "values": []},
            "quadrant": [], "radar": {"categories": [], "series": []}, "scatter": [], "insights": [],
        })
    total_sales = filtered["total_sales"].sum()
    avg_critic = filtered["critic_score"].mean()
    payload = {
        "globalMetrics": {
            "totalGames": int(len(filtered)),
            "totalSales": round(float(total_sales), 2),
            "avgCritic": round(float(avg_critic), 2),
            "uniqueConsoles": int(filtered["console"].nunique()),
        },
        "topConsoles": top_sum(filtered, "console", "total_sales", 10),
        "topGenres": top_sum(filtered, "genre", "total_sales", 10),
        "topPublishers": publisher_ranking(filtered),
        "regional": regional_distribution(filtered),
        "marketShare": console_market_share(filtered),
        "criticHistogram": critic_histogram(filtered),
        "quadrant": strategic_quadrant(filtered),
        "radar": radar_strategy(filtered),
        "scatter": scatter_points(filtered),
        "insights": generate_insights(filtered, genre),
    }
    return jsonify(clean_json(payload))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)