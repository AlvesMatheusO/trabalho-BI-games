"""
tratamento_dados.py
═══════════════════════════════════════════════════════════════════════════════
Script de pré-processamento do dataset VGChartz 2024.

Problemas identificados e resolvidos:
  1. DLCs / Expansões / Map Packs / Track Packs registrados como jogos
  2. Bundles multi-jogo (2 ou 3 jogos vendidos juntos)
  3. Compilações de múltiplos jogos antigos
  4. Registros agregados (console = "All", "Series", "WW") sem dados úteis
  5. Mesmo jogo em múltiplos consoles — consolidado quando necessário
  6. 70% de total_sales ausente — imputação por média de gênero
  7. 90% de critic_score ausente — imputação hierárquica
  8. Datas com granularidade variável (ano, mês, dia)
  9. Dados de vendas regionais parcialmente ausentes

Saídas:
  vgchartz_clean.csv        → dataset limpo, 1 linha por jogo por console
  vgchartz_consolidated.csv → 1 linha por jogo (vendas somadas de todos consoles)
  relatorio_limpeza.txt     → log detalhado de cada etapa
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from datetime import datetime

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────
INPUT_CSV  = Path("vgchartz-2024.csv")
OUT_CLEAN  = Path("vgchartz_clean.csv")
OUT_CONSOL = Path("vgchartz_consolidated.csv")
OUT_LOG    = Path("relatorio_limpeza.txt")

REGIONAL_COLS = ["na_sales", "jp_sales", "pal_sales", "other_sales"]
SALES_COLS    = ["total_sales"] + REGIONAL_COLS

# Consoles de loja digital — tratados como plataforma legítima, não removidos
DIGITAL_CONSOLES = {
    "XBL",   # Xbox Live Arcade
    "PSN",   # PlayStation Network
    "DSiW",  # DSiWare
    "VC",    # Virtual Console
    "And",   # Android
    "iOS",   # Apple iOS
    "OSX",   # macOS App Store
    "WW",    # WiiWare
    "PC",    # já incluso como plataforma
}

# Consoles agregados — registros inúteis sem dados de venda
AGGREGATE_CONSOLES = {"All", "Series"}

# ═══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 1 — PADRÕES DE CLASSIFICAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

# 1-A: DLC / Expansão — conteúdo adicional que NÃO é um jogo completo
RE_DLC = re.compile(
    r"""
    \b(
        dlc                         |
        expansion\s+pack            |
        add-on\s+pack               |
        addon\s+pack                |
        season\s+pass               |
        map\s+pack                  |
        track\s+pack                |
        mission\s+pack              |
        character\s+pack            |
        costume\s+pack              |
        skin\s+pack                 |
        weapon\s+pack               |
        content\s+pack              |
        downloadable\s+content
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 1-B: Bundles de múltiplos jogos empacotados juntos
RE_BUNDLE = re.compile(
    r"""
    \b(
        double\s+pack               |
        twin\s+pack                 |
        dual\s+pack                 |
        two\s+pack                  |
        triple\s+pack               |
        2-in-1                      |
        3-in-1
    )\b
    |
    \b\d+\s+games?\s+in\s+\d+\b    # "2 Games in 1", "3 Games in 1"
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 1-C: Compilações / coleções de múltiplos jogos distintos
#      (diferente de remaster de 1 jogo, que é mantido)
RE_COMPILATION = re.compile(
    r"""
    \b(
        anthology                   |
        compilation                 |
        the\s+complete\s+saga       |
        arcade\s+classics           |
        classic\s+collection        |
        heritage\s+collection       |
        greatest\s+hits\s+collection|
        museum                      |
        archives                    |
        mega\s+collection           |
        retro\s+collection          |
        megamix                     |
        all.stars
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 1-D: Edições que SÃO o jogo — mantidas como jogos legítimos
#      Remaster, Director's Cut, GOTY etc. são releases do jogo base
RE_EDITION_KEEP = re.compile(
    r"""
    \b(
        remastered                  |
        remake                      |
        definitive\s+edition        |
        enhanced\s+edition          |
        game\s+of\s+the\s+year      |
        goty\s+edition              |
        director.?s\s+cut           |
        special\s+edition           |
        anniversary\s+edition       |
        deluxe\s+edition            |
        ultimate\s+edition          |
        legendary\s+edition         |
        complete\s+edition          |
        redux                       |
        extended\s+cut              |
        warmastered\s+edition       |
        royal\s+edition
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_title(title: str, console: str) -> str:
    """
    Retorna o tipo do registro:
      'base_game'         → jogo ou edição especial de um único jogo
      'dlc_expansion'     → DLC, expansão, pack de conteúdo
      'multi_game_bundle' → bundle com 2+ jogos distintos
      'compilation'       → coletânea de vários jogos antigos
      'aggregate_record'  → linha de total de série / plataforma All
    """
    if console in AGGREGATE_CONSOLES:
        return "aggregate_record"

    # Edições especiais de jogo único → manter antes de checar compilações
    if RE_EDITION_KEEP.search(title):
        return "base_game"

    if RE_DLC.search(title):
        return "dlc_expansion"

    if RE_BUNDLE.search(title):
        return "multi_game_bundle"

    if RE_COMPILATION.search(title):
        return "compilation"

    return "base_game"


# ═══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 2 — FUNÇÕES DE LIMPEZA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_release_date(series: pd.Series) -> pd.Series:
    """
    Normaliza datas com granularidade variada:
      '2013-09-17' → ok
      '2013-09-01' → mantém (mês sem dia exato)
      '2013-01-01' → mantém (só ano)
      Strings inválidas → NaT
    """
    return pd.to_datetime(series, errors="coerce", format="mixed")


def impute_sales(df: pd.DataFrame, col: str, by: str = "genre") -> pd.Series:
    """
    Imputa valores ausentes em coluna de vendas:
      1. Média do gênero + console
      2. Média do gênero
      3. Mediana global
    """
    result = df[col].copy()
    mask   = result.isna()

    # Nível 1: média por gênero + console
    means_gc = df.groupby(["genre", "console"])[col].transform("mean")
    result = result.where(~mask, means_gc)
    mask   = result.isna()

    # Nível 2: média por gênero
    means_g = df.groupby("genre")[col].transform("mean")
    result  = result.where(~mask, means_g)
    mask    = result.isna()

    # Nível 3: mediana global (conservador para outliers)
    result  = result.where(~mask, df[col].median())

    return result.round(3)


def impute_critic_score(df: pd.DataFrame) -> pd.Series:
    """
    Imputa critic_score ausente (89,6% dos dados):
      1. Média por gênero + console
      2. Média por gênero
      3. Média global
    Marca qual método foi usado em coluna auxiliar.
    """
    result = df["critic_score"].copy()
    mask   = result.isna()

    means_gc = df.groupby(["genre", "console"])["critic_score"].transform("mean")
    result   = result.where(~mask, means_gc)
    mask     = result.isna()

    means_g  = df.groupby("genre")["critic_score"].transform("mean")
    result   = result.where(~mask, means_g)
    mask     = result.isna()

    result   = result.where(~mask, df["critic_score"].mean())

    return result.round(2)


def reconstruct_total_sales(df: pd.DataFrame) -> pd.Series:
    """
    Quando total_sales está ausente mas regionais existem,
    reconstrói total_sales como soma das regionais.
    """
    regional_sum = df[REGIONAL_COLS].sum(axis=1, min_count=1)
    total = df["total_sales"].copy()

    # Só usa soma regional quando total está ausente E soma regional > 0
    missing_total = total.isna()
    has_regional  = regional_sum.notna() & (regional_sum > 0)
    total         = total.where(~(missing_total & has_regional), regional_sum)

    return total


def normalize_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas de texto: strip, capitalização uniforme.
    Substitui 'N/A', 'Unknown', valores vazios por NaN.
    """
    text_cols = ["title", "console", "genre", "publisher", "developer"]
    for col in text_cols:
        df[col] = (
            df[col]
            .str.strip()
            .replace({"N/A": np.nan, "Unknown": np.nan, "": np.nan, "Não informado": np.nan})
        )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 3 — PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run(log):

    def write(msg=""):
        print(msg)
        log.write(msg + "\n")

    write("=" * 72)
    write("  VGChartz 2024 — Script de Tratamento de Dados")
    write(f"  Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write("=" * 72)

    # ── CARGA ─────────────────────────────────────────────────────────────────
    write("\n[1] CARGA DO ARQUIVO")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    write(f"    Linhas carregadas : {len(df):>7,}")
    write(f"    Colunas           : {df.shape[1]}")
    write(f"    Colunas presentes : {df.columns.tolist()}")

    # Remove coluna de imagem — sem utilidade analítica
    if "img" in df.columns:
        df.drop(columns=["img"], inplace=True)
        write("    ✓ Coluna 'img' removida")

    # ── PASSO 1: Normalização de texto ────────────────────────────────────────
    write("\n[2] NORMALIZAÇÃO DE TEXTO")
    df = normalize_text_cols(df)
    write("    ✓ Títulos e campos de texto: strip + substituição de placeholders")

    # ── PASSO 2: Remoção de duplicatas exatas ─────────────────────────────────
    write("\n[3] DUPLICATAS EXATAS")
    n_before = len(df)
    df.drop_duplicates(subset=["title", "console"], keep="first", inplace=True)
    n_dup = n_before - len(df)
    write(f"    Duplicatas removidas (title+console): {n_dup:,}")
    write(f"    Linhas restantes                    : {len(df):,}")

    # ── PASSO 3: Classificação de tipos ───────────────────────────────────────
    write("\n[4] CLASSIFICAÇÃO DE REGISTROS")
    df["record_type"] = df.apply(
        lambda r: classify_title(str(r["title"]), str(r["console"])), axis=1
    )
    type_counts = df["record_type"].value_counts()
    for t, n in type_counts.items():
        write(f"    {t:<25}: {n:>6,}")

    # ── PASSO 4: Filtragem — manter apenas jogos base ─────────────────────────
    write("\n[5] FILTRAGEM — REMOVER NÃO-JOGOS")
    removed = df[df["record_type"] != "base_game"].copy()
    df      = df[df["record_type"] == "base_game"].copy()

    write(f"    Registros removidos : {len(removed):,}")
    write(f"    Jogos mantidos      : {len(df):,}")

    # Breakdown do que foi removido
    removed_breakdown = removed["record_type"].value_counts()
    for t, n in removed_breakdown.items():
        write(f"      → {t}: {n:,}")

    # Salvar removidos para auditoria
    removed_path = Path("vgchartz_removidos.csv")
    removed.to_csv(removed_path, index=False)
    write(f"    ℹ Registros removidos salvos em: {removed_path}")

    # ── PASSO 5: Datas ────────────────────────────────────────────────────────
    write("\n[6] TRATAMENTO DE DATAS")
    df["release_date"] = parse_release_date(df["release_date"])
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["release_month"]= df["release_date"].dt.month.astype("Int64")

    n_no_date = df["release_date"].isna().sum()
    write(f"    Datas válidas   : {df['release_date'].notna().sum():,}")
    write(f"    Datas ausentes  : {n_no_date:,}")
    write(f"    Intervalo       : {int(df['release_year'].min())} – {int(df['release_year'].max())}")

    # ── PASSO 6: Reconstrução de total_sales via regionais ────────────────────
    write("\n[7] RECONSTRUÇÃO DE VENDAS")
    n_missing_total_before = df["total_sales"].isna().sum()
    df["total_sales"] = reconstruct_total_sales(df)
    n_missing_total_after  = df["total_sales"].isna().sum()
    recovered = n_missing_total_before - n_missing_total_after
    write(f"    total_sales ausentes antes    : {n_missing_total_before:,}")
    write(f"    Recuperados via soma regional : {recovered:,}")
    write(f"    total_sales ausentes após     : {n_missing_total_after:,}")

    # ── PASSO 7: Imputação de vendas ausentes ─────────────────────────────────
    write("\n[8] IMPUTAÇÃO DE VENDAS AUSENTES")
    for col in SALES_COLS:
        n_miss = df[col].isna().sum()
        df[col] = impute_sales(df, col)
        write(f"    {col:<15}: {n_miss:>5,} ausentes imputados via média gênero+console → gênero → mediana global")

    # Remover linhas onde total_sales ficou zerado (dado realmente inexistente)
    n_zero = (df["total_sales"] == 0).sum()
    write(f"\n    Linhas com total_sales = 0 após imputação: {n_zero:,}")
    # Não removemos — zeros podem representar lançamentos recentes sem dados

    # ── PASSO 8: Imputação de critic_score ────────────────────────────────────
    write("\n[9] IMPUTAÇÃO DE CRITIC_SCORE")
    n_miss_cs = df["critic_score"].isna().sum()
    write(f"    Ausentes originais: {n_miss_cs:,} ({n_miss_cs/len(df)*100:.1f}%)")
    df["critic_score"]       = impute_critic_score(df)
    df["critic_score_imputed"] = (df["critic_score"].notna()) & (n_miss_cs > 0)
    write("    ✓ Imputação: média gênero+console → média gênero → média global")
    write(f"    critic_score ausentes restantes: {df['critic_score'].isna().sum():,}")

    # ── PASSO 9: Colunas derivadas ────────────────────────────────────────────
    write("\n[10] COLUNAS DERIVADAS")

    # Aprovação implícita pela nota (0–10 → 0–100%)
    df["score_normalized"] = (df["critic_score"] / 10 * 100).round(1)

    # Receita estimada (total_sales em M × $40 preço médio histórico físico)
    df["est_revenue_usd_M"] = (df["total_sales"] * 40).round(2)

    # Plataforma categorizada
    platform_map = {
        "PS5": "Current Gen", "XS": "Current Gen", "NS": "Current Gen",
        "PS4": "Previous Gen", "XOne": "Previous Gen", "WiiU": "Previous Gen",
        "PS3": "Last Gen", "X360": "Last Gen", "Wii": "Last Gen",
        "PS2": "Retro", "XB": "Retro", "GC": "Retro", "DC": "Retro",
        "PS": "Retro", "N64": "Retro", "SNES": "Retro", "NES": "Retro",
        "DS": "Handheld", "3DS": "Handheld", "PSP": "Handheld", "PSV": "Handheld",
        "GBA": "Handheld", "GB": "Handheld", "GBC": "Handheld",
        "PC": "PC/Digital", "OSX": "PC/Digital", "Linux": "PC/Digital",
        "XBL": "Digital", "PSN": "Digital", "DSiW": "Digital",
        "VC": "Digital", "And": "Mobile", "iOS": "Mobile",
    }
    df["platform_gen"] = df["console"].map(platform_map).fillna("Other")

    # Decade do lançamento
    df["release_decade"] = (df["release_year"] // 10 * 10).astype("Int64").astype(str) + "s"
    df.loc[df["release_year"].isna(), "release_decade"] = np.nan

    write("    ✓ score_normalized   — nota em escala 0–100")
    write("    ✓ est_revenue_usd_M  — receita estimada (sales × $40)")
    write("    ✓ platform_gen       — geração/tipo de plataforma")
    write("    ✓ release_decade     — década de lançamento")
    write("    ✓ release_year       — ano extraído da data")
    write("    ✓ release_month      — mês extraído da data")

    # ── PASSO 10: Tipos de dados finais ───────────────────────────────────────
    write("\n[11] TIPAGEM FINAL")
    df["total_sales"]   = df["total_sales"].astype(float).round(3)
    df["critic_score"]  = df["critic_score"].astype(float).round(2)
    for col in REGIONAL_COLS:
        df[col] = df[col].astype(float).round(3)
    df["developer"]  = df["developer"].fillna("Não informado")
    df["publisher"]  = df["publisher"].fillna("Não informado")
    df["genre"]      = df["genre"].fillna("Não informado")
    write("    ✓ Tipos numéricos aplicados, NaN de texto preenchidos")

    # ── SAÍDA 1: dataset limpo por console ────────────────────────────────────
    write("\n[12] SALVANDO vgchartz_clean.csv")
    col_order = [
        "title", "console", "platform_gen", "genre", "publisher", "developer",
        "release_date", "release_year", "release_month", "release_decade",
        "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales",
        "critic_score", "score_normalized",
        "est_revenue_usd_M", "record_type", "last_update",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(OUT_CLEAN, index=False)
    write(f"    ✓ {len(df):,} linhas × {len(col_order)} colunas salvas em {OUT_CLEAN}")

    # ── SAÍDA 2: dataset consolidado por jogo (soma de consoles) ──────────────
    write("\n[13] CONSOLIDAÇÃO POR JOGO (soma de todos os consoles)")

    # Para consolidar: somar vendas, manter metadados do console com mais vendas
    agg_dict = {
        "total_sales":      "sum",
        "na_sales":         "sum",
        "jp_sales":         "sum",
        "pal_sales":        "sum",
        "other_sales":      "sum",
        "critic_score":     "max",   # nota mais alta entre plataformas
        "score_normalized": "max",
        "est_revenue_usd_M":"sum",
        "release_date":     "min",   # data de lançamento mais antiga = original
        "release_year":     "min",
        "release_decade":   "first",
        "genre":            "first",
        "publisher":        "first",
        "developer":        "first",
        "console":          lambda x: " | ".join(sorted(x.unique())),  # lista de consoles
        "platform_gen":     lambda x: " | ".join(sorted(x.unique())),
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    consolidated = (
        df.groupby("title", as_index=False)
          .agg(agg_dict)
    )
    consolidated["n_consoles"] = (
        df.groupby("title")["console"].nunique().reindex(consolidated["title"]).values
    )
    consolidated = consolidated.sort_values("total_sales", ascending=False)

    col_order_c = [
        "title", "genre", "publisher", "developer",
        "release_date", "release_year", "release_decade",
        "n_consoles", "console",
        "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales",
        "critic_score", "score_normalized", "est_revenue_usd_M",
    ]
    col_order_c = [c for c in col_order_c if c in consolidated.columns]
    consolidated[col_order_c].to_csv(OUT_CONSOL, index=False)

    write(f"    ✓ {len(consolidated):,} jogos únicos consolidados")
    write(f"    ✓ Salvos em {OUT_CONSOL}")

    # ── RESUMO FINAL ──────────────────────────────────────────────────────────
    write("\n" + "=" * 72)
    write("  RESUMO FINAL")
    write("=" * 72)
    write(f"  Registros originais          : {64016:>7,}")
    write(f"  Duplicatas removidas         : {n_dup:>7,}")
    write(f"  DLCs / Expansões removidos   : {removed_breakdown.get('dlc_expansion', 0):>7,}")
    write(f"  Bundles multi-jogo removidos : {removed_breakdown.get('multi_game_bundle', 0):>7,}")
    write(f"  Compilações removidas        : {removed_breakdown.get('compilation', 0):>7,}")
    write(f"  Registros agregados removidos: {removed_breakdown.get('aggregate_record', 0):>7,}")
    write(f"  Jogos únicos (por console)   : {len(df):>7,}")
    write(f"  Jogos únicos (consolidados)  : {len(consolidated):>7,}")
    write(f"\n  Arquivos gerados:")
    write(f"    → {OUT_CLEAN}   (por console)")
    write(f"    → {OUT_CONSOL}  (consolidado)")
    write(f"    → vgchartz_removidos.csv      (auditoria)")
    write(f"    → {OUT_LOG}     (este relatório)")
    write("=" * 72)

    # ── ESTATÍSTICAS DOS DADOS LIMPOS ─────────────────────────────────────────
    write("\n  TOP 10 JOGOS POR VENDAS CONSOLIDADAS:")
    top10 = consolidated.head(10)[["title", "total_sales", "genre", "critic_score", "n_consoles"]]
    for _, r in top10.iterrows():
        write(f"    {r['title'][:45]:<45} {r['total_sales']:>6.2f}M  {r['genre']:<12} nota:{r['critic_score']:.1f}  {r['n_consoles']} console(s)")

    write("\n  DISTRIBUIÇÃO POR GÊNERO (top 10 por vendas totais consolidadas):")
    genre_sales = (
        consolidated.groupby("genre")["total_sales"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .head(10)
    )
    for gn, row in genre_sales.iterrows():
        write(f"    {gn:<18} total:{row['sum']:>8.1f}M  jogos:{row['count']:>5,}  média:{row['mean']:>5.2f}M")

    write("\n  VENDAS POR GERAÇÃO DE PLATAFORMA:")
    if "platform_gen" in df.columns:
        pg = df.groupby("platform_gen")["total_sales"].sum().sort_values(ascending=False)
        for gen, sales in pg.items():
            write(f"    {gen:<18} {sales:>8.1f}M")

    return df, consolidated


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not INPUT_CSV.exists():
        print(f"ERRO: Arquivo '{INPUT_CSV}' não encontrado.")
        print("Coloque o arquivo vgchartz-2024.csv na mesma pasta que este script.")
        sys.exit(1)

    with open(OUT_LOG, "w", encoding="utf-8") as log:
        df_clean, df_consolidated = run(log)

    print(f"\n✅ Processamento concluído! Relatório em: {OUT_LOG}")