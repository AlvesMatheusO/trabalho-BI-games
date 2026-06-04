from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ── CAMINHOS — sempre absolutos baseados no diretório do app.py ───────────────
BASE_DIR  = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"

# Tenta o CSV consolidado (tratado) primeiro; cai no original se não existir
CONSOLIDATED_CSV = BASE_DIR / "vgchartz_consolidated.csv"
ORIGINAL_CSV     = BASE_DIR / "vgchartz-2024.csv"

if CONSOLIDATED_CSV.exists():
    CSV_PATH = CONSOLIDATED_CSV
    print(f"[INFO] Usando CSV tratado: {CSV_PATH.name}")
elif ORIGINAL_CSV.exists():
    CSV_PATH = ORIGINAL_CSV
    print(f"[WARN] CSV consolidado não encontrado. Usando original: {CSV_PATH.name}")
else:
    raise FileNotFoundError(
        "Nenhum CSV encontrado. Coloque 'vgchartz_consolidated.csv' "
        "ou 'vgchartz-2024.csv' na mesma pasta que app.py."
    )

# ── COLUNAS ESPERADAS ─────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "total_sales", "critic_score",
    "na_sales", "pal_sales", "jp_sales", "other_sales",
]

# Mapeamento: colunas do CSV consolidado → nomes usados pelo app
# O CSV consolidado tem os mesmos nomes, então nenhum rename é necessário.
REQUIRED_COLS = [
    "title", "genre", "publisher", "developer",
    "critic_score", "total_sales",
    "na_sales", "jp_sales", "pal_sales", "other_sales",
]

# O CSV consolidado não tem "console" como coluna única (é lista separada por |)
# Criamos um console sintético para manter compatibilidade com as funções
CONSOLE_COL_EXISTS = True  # será verificado após leitura


# ── LIMPEZA JSON ──────────────────────────────────────────────────────────────
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


# ── CARGA DO DATASET ──────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(f"[INFO] {len(df):,} linhas carregadas de {CSV_PATH.name}")
    print(f"[INFO] Colunas: {df.columns.tolist()}")

    # Verificar colunas obrigatórias
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    # Converter numéricos
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preencher textos
    for col in ["title", "genre", "publisher", "developer"]:
        if col in df.columns:
            df[col] = df[col].fillna("Não informado").astype(str)

    # Criar coluna "console" compatível se não existir
    if "console" not in df.columns:
        # No CSV consolidado a coluna se chama "console" (lista separada por |)
        # Se ainda assim não existir, criamos uma genérica
        df["console"] = "Multi"
    else:
        df["console"] = df["console"].fillna("Multi").astype(str)

    # Imputação de critic_score ausente → média do gênero → média global
    if df["critic_score"].isna().any():
        genre_mean  = df.groupby("genre")["critic_score"].transform("mean")
        global_mean = df["critic_score"].mean()
        df["critic_score"] = (
            df["critic_score"]
            .fillna(genre_mean)
            .fillna(global_mean)
        )

    # Imputação de vendas ausentes → 0 (conservador para totais consolidados)
    for col in ["total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]:
        df[col] = df[col].fillna(0)

    print(f"[INFO] Dataset pronto: {len(df):,} jogos")
    return df


df = load_dataset()


# ── ROTAS ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_file(HTML_PATH)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "rows":   int(len(df)),
        "csv":    CSV_PATH.name,
    })


@app.route("/api/genres")
def genres():
    values = df["genre"].dropna().sort_values().unique().tolist()
    return jsonify(values)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def filter_by_genre(genre: str) -> pd.DataFrame:
    if genre and genre.lower() != "all":
        return df[df["genre"].str.lower() == genre.lower()].copy()
    return df.copy()


def top_sum(dataframe, group_col, value_col, limit=10):
    data = (
        dataframe.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
        .round(2)
    )
    return {str(k): float(v) for k, v in data.items()}


def regional_distribution(dataframe):
    regions = {
        "América do Norte": float(dataframe["na_sales"].sum()),
        "Europa/PAL":       float(dataframe["pal_sales"].sum()),
        "Japão":            float(dataframe["jp_sales"].sum()),
        "Outros":           float(dataframe["other_sales"].sum()),
    }
    total = sum(regions.values())
    if total <= 0:
        return {k: 0 for k in regions}
    return {k: round(v / total * 100, 2) for k, v in regions.items()}


def critic_histogram(dataframe):
    bins   = ["0-2", "2-4", "4-6", "6-8", "8-10"]
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
        .agg(
            total_sales=("total_sales", "sum"),
            avg_critic= ("critic_score", "mean"),
            games=      ("title", "count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(12)
    )
    return [
        {
            "x":    round(float(r["avg_critic"]), 2),
            "y":    round(float(r["total_sales"]), 2),
            "z":    int(r["games"]),
            "name": r["genre"],
        }
        for _, r in grouped.iterrows()
    ]


def radar_strategy(dataframe):
    grouped = (
        dataframe.groupby("genre")
        .agg(
            total_sales=("total_sales", "sum"),
            avg_critic= ("critic_score", "mean"),
            games=      ("title", "count"),
            na=         ("na_sales", "sum"),
            pal=        ("pal_sales", "sum"),
            jp=         ("jp_sales", "sum"),
            other=      ("other_sales", "sum"),
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
        global_pres  = (int(row["na"]>0)+int(row["pal"]>0)+int(row["jp"]>0)+int(row["other"]>0)) / 4 * 100
        sales_score  = (row["total_sales"] / max_sales * 100) if max_sales > 0 else 0
        critic_score = (row["avg_critic"]  / 10        * 100) if row["avg_critic"] > 0 else 0
        volume_score = (row["games"]       / max_games  * 100) if max_games > 0 else 0
        opportunity  = critic_score*0.35 + sales_score*0.35 + global_pres*0.2 + (100-volume_score)*0.1
        series.append({
            "name": row["genre"],
            "data": [
                round(float(sales_score), 2),
                round(float(critic_score), 2),
                round(float(global_pres), 2),
                round(float(volume_score), 2),
                round(float(opportunity), 2),
            ],
        })
    return {
        "categories": ["Vendas", "Crítica", "Presença Global", "Volume de Jogos", "Oportunidade"],
        "series": series,
    }


def publisher_ranking(dataframe):
    data = (
        dataframe.groupby("publisher")
        .agg(
            total_sales=("total_sales", "sum"),
            avg_critic= ("critic_score", "mean"),
            games=      ("title", "count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(10)
    )
    return {
        "labels": data["publisher"].tolist(),
        "sales":  data["total_sales"].round(2).tolist(),
        "critic": data["avg_critic"].round(2).tolist(),
        "games":  data["games"].astype(int).tolist(),
    }


def console_market_share(dataframe):
    """
    No CSV consolidado, 'console' é uma lista separada por ' | '.
    Explodimos para contar por plataforma individual.
    """
    exploded = (
        dataframe[["title", "console", "total_sales"]]
        .assign(console=dataframe["console"].str.split(r"\s*\|\s*"))
        .explode("console")
    )
    top = (
        exploded.groupby("console")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )
    return {
        "labels": top.index.astype(str).tolist(),
        "values": top.round(2).tolist(),
    }


def scatter_points(dataframe):
    valid = (
        dataframe[["critic_score", "total_sales", "title", "genre"]]
        .dropna()
        .query("critic_score >= 0 and total_sales >= 0")
        .sort_values("total_sales", ascending=False)
        .head(1200)
    )
    return valid.to_dict(orient="records")


def generate_insights(dataframe, genre: str):
    if dataframe.empty:
        return ["Nenhum dado encontrado para o filtro selecionado."]

    insights = []
    top_genre     = dataframe.groupby("genre")["total_sales"].sum().sort_values(ascending=False)
    top_publisher = dataframe.groupby("publisher")["total_sales"].sum().sort_values(ascending=False)
    avg_critic    = dataframe["critic_score"].mean()
    total_sales   = dataframe["total_sales"].sum()
    total_games   = len(dataframe)

    if genre.lower() == "all" and not top_genre.empty:
        insights.append(
            f"O gênero com maior volume de vendas é {top_genre.index[0]}, "
            f"com {top_genre.iloc[0]:.1f}M em vendas."
        )

    if not top_publisher.empty:
        insights.append(
            f"A publisher líder é {top_publisher.index[0]}, com "
            f"{top_publisher.iloc[0]:.1f}M em vendas totais."
        )

    quality_msg = (
        "A nota crítica média está em um nível saudável (≥ 7.5), "
        "sugerindo boa aceitação qualitativa do catálogo."
        if avg_critic >= 7.5 else
        "A nota crítica média está abaixo de 7.5. Estratégias de melhoria "
        "de qualidade e polimento podem ser importantes."
    )
    insights.append(quality_msg)

    if total_games > 0:
        spg = total_sales / total_games
        insights.append(
            f"Média de {spg:.2f}M vendas por jogo — "
            + ("alto potencial comercial por título." if spg >= 1 else
               "mercado mais pulverizado ou nichado.")
        )

    return insights


# ── ENDPOINT PRINCIPAL ────────────────────────────────────────────────────────
@app.route("/api/summary")
def summary():
    genre    = request.args.get("genre", "all")
    filtered = filter_by_genre(genre)

    if filtered.empty:
        return jsonify({
            "globalMetrics":   {"totalGames": 0, "totalSales": 0, "avgCritic": 0, "uniqueConsoles": 0},
            "topConsoles": {}, "topGenres": {},
            "topPublishers":   {"labels": [], "sales": [], "critic": [], "games": []},
            "regional": {},    "marketShare": {"labels": [], "values": []},
            "criticHistogram": {"labels": [], "values": []},
            "quadrant": [],    "radar": {"categories": [], "series": []},
            "scatter": [],     "insights": [],
        })

    # Contar consoles únicos (explodindo a coluna lista)
    consoles_unique = (
        filtered["console"]
        .str.split(r"\s*\|\s*")
        .explode()
        .nunique()
    )

    payload = {
        "globalMetrics": {
            "totalGames":     int(len(filtered)),
            "totalSales":     round(float(filtered["total_sales"].sum()), 2),
            "avgCritic":      round(float(filtered["critic_score"].mean()), 2),
            "uniqueConsoles": int(consoles_unique),
        },
        "topConsoles":    top_sum(filtered, "genre", "total_sales", 10),  # genre como proxy
        "topGenres":      top_sum(filtered, "genre", "total_sales", 10),
        "topPublishers":  publisher_ranking(filtered),
        "regional":       regional_distribution(filtered),
        "marketShare":    console_market_share(filtered),
        "criticHistogram":critic_histogram(filtered),
        "quadrant":       strategic_quadrant(filtered),
        "radar":          radar_strategy(filtered),
        "scatter":        scatter_points(filtered),
        "insights":       generate_insights(filtered, genre),
    }
    return jsonify(clean_json(payload))


# ── START ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)