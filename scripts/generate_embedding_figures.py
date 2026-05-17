from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "reports/.matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", str(Path("reports/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "04_embeddings"
OUTPUT = ROOT / "thesis" / "latex" / "figuras" / "generated"
SPLIT_XL_TEST = ROOT / "data" / "processed" / "04_splits_xl" / "test.csv"
V3_XL_MODEL = ROOT / "models" / "sentence_transformers" / "v3_iterative_hn_e5_base_bidirectional_xl"


@dataclass(frozen=True)
class MetricSource:
    order: int
    candidate: str
    family: str
    direction: str
    path: Path
    hypothesis: str
    change: str
    result: str
    decision: str


SOURCES = [
    MetricSource(
        1,
        "E5-small baseline",
        "E5-small",
        "principal",
        REPORTS / "baseline" / "baseline_retrieval.json",
        "Medir el punto de partida sin adaptar el modelo.",
        "E5-small preentrenado sin fine-tuning.",
        "R@1 bajo: el espacio multilingue no alinea bien el par por si solo.",
        "Usar fine-tuning contrastivo.",
    ),
    MetricSource(
        2,
        "v1 E5-small",
        "E5-small",
        "principal",
        REPORTS / "v1" / "v1_retrieval.json",
        "Comprobar si los pares positivos del corpus adaptan el espacio semantico.",
        "Fine-tuning con MultipleNegativesRankingLoss.",
        "Mejora fuerte frente al baseline.",
        "Explorar negativos dificiles.",
    ),
    MetricSource(
        3,
        "v2 hard-only",
        "E5-small",
        "principal",
        REPORTS / "v2_hn_controlled_hard" / "v2_hn_controlled_hard_retrieval.json",
        "Ver si solo los negativos duros aportan senal suficiente.",
        "Entrenamiento con hard negatives controlados.",
        "Mejora moderada, menor que hard/medium.",
        "Combinar negativos duros y medios.",
    ),
    MetricSource(
        4,
        "v2 hard/medium",
        "E5-small",
        "principal",
        REPORTS / "v2_hn_controlled" / "v2_hn_controlled_retrieval.json",
        "Probar si los negativos medios estabilizan la discriminacion.",
        "Entrenamiento con hard y medium negatives.",
        "Mejora mayor que hard-only.",
        "Migrar a una base de mayor capacidad.",
    ),
    MetricSource(
        5,
        "E5-base baseline",
        "E5-base",
        "principal",
        REPORTS / "experiments" / "baseline_e5_base" / "baseline_e5_base_retrieval.json",
        "Evaluar si una base mas grande mejora sin ajuste.",
        "E5-base preentrenado sin fine-tuning.",
        "Sigue bajo sin adaptacion, pero ofrece mayor capacidad.",
        "Aplicar fine-tuning a E5-base.",
    ),
    MetricSource(
        6,
        "v1 E5-base",
        "E5-base",
        "principal",
        REPORTS / "experiments" / "v1_e5_base" / "v1_e5_base_retrieval.json",
        "Repetir fine-tuning contrastivo con base mas fuerte.",
        "E5-base + MultipleNegativesRankingLoss.",
        "Supera claramente a la linea E5-small.",
        "Usar E5-base como linea principal.",
    ),
    MetricSource(
        7,
        "v1 E5-base bidir.",
        "E5-base",
        "esp_to_shi",
        REPORTS
        / "experiments"
        / "v1_e5_base_bidirectional"
        / "v1_e5_base_bidirectional_esp_to_shi_retrieval.json",
        "Evitar favorecer una sola direccion.",
        "Fine-tuning bidireccional.",
        "Mejora leve en espanol -> shiwilu.",
        "Evaluar tambien shiwilu -> espanol.",
    ),
    MetricSource(
        7,
        "v1 E5-base bidir.",
        "E5-base",
        "shi_to_esp",
        REPORTS
        / "experiments"
        / "v1_e5_base_bidirectional"
        / "v1_e5_base_bidirectional_shi_to_esp_retrieval.json",
        "Evitar favorecer una sola direccion.",
        "Fine-tuning bidireccional.",
        "Mejora en shiwilu -> espanol frente a la variante principal.",
        "Mantener evaluacion bidireccional.",
    ),
    MetricSource(
        8,
        "v2 E5-base",
        "E5-base",
        "esp_to_shi",
        REPORTS
        / "experiments"
        / "v2_hn_controlled_e5_base"
        / "v2_hn_controlled_e5_base_esp_to_shi_retrieval.json",
        "Combinar E5-base con mineria controlada de negativos.",
        "Hard/medium negatives sobre v1 E5-base.",
        "Mejora relevante frente a v1 E5-base.",
        "Probar configuracion bidireccional.",
    ),
    MetricSource(
        8,
        "v2 E5-base",
        "E5-base",
        "shi_to_esp",
        REPORTS
        / "experiments"
        / "v2_hn_controlled_e5_base"
        / "v2_hn_controlled_e5_base_shi_to_esp_retrieval.json",
        "Verificar desempeno inverso con negativos controlados.",
        "Evaluacion shiwilu -> espanol.",
        "Comportamiento similar en direccion inversa.",
        "Probar entrenamiento bidireccional.",
    ),
    MetricSource(
        9,
        "v2 E5-base bidir.",
        "E5-base",
        "esp_to_shi",
        REPORTS
        / "experiments"
        / "v2_hn_controlled_e5_base_bidirectional"
        / "v2_hn_controlled_e5_base_bidirectional_esp_to_shi_retrieval.json",
        "Unir negativos controlados y entrenamiento bidireccional.",
        "Hard/medium negatives bidireccionales.",
        "Candidato anterior fuerte en espanol -> shiwilu.",
        "Usarlo como base para mineria iterativa.",
    ),
    MetricSource(
        9,
        "v2 E5-base bidir.",
        "E5-base",
        "shi_to_esp",
        REPORTS
        / "experiments"
        / "v2_hn_controlled_e5_base_bidirectional"
        / "v2_hn_controlled_e5_base_bidirectional_shi_to_esp_retrieval.json",
        "Unir negativos controlados y entrenamiento bidireccional.",
        "Hard/medium negatives bidireccionales.",
        "Candidato anterior fuerte en shiwilu -> espanol.",
        "Usarlo como base para mineria iterativa.",
    ),
    MetricSource(
        10,
        "v3 iterativo bidir.",
        "E5-base",
        "esp_to_shi",
        REPORTS
        / "experiments"
        / "v3_iterative_hn_e5_base_bidirectional"
        / "v3_iterative_hn_e5_base_bidirectional_esp_to_shi_retrieval.json",
        "Minar nuevos negativos desde el mejor candidato previo.",
        "Nueva iteracion de hard/medium negatives.",
        "Mejor R@1 y MRR en espanol -> shiwilu.",
        "Seleccionado como candidato final provisional.",
    ),
    MetricSource(
        10,
        "v3 iterativo bidir.",
        "E5-base",
        "shi_to_esp",
        REPORTS
        / "experiments"
        / "v3_iterative_hn_e5_base_bidirectional"
        / "v3_iterative_hn_e5_base_bidirectional_shi_to_esp_retrieval.json",
        "Minar nuevos negativos desde el mejor candidato previo.",
        "Nueva iteracion de hard/medium negatives.",
        "Mejor R@1 y MRR en shiwilu -> espanol.",
        "Seleccionado como candidato final provisional.",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        payload = read_json(source.path)
        metrics = payload["metrics"]
        rows.append(
            {
                "order": source.order,
                "candidate": source.candidate,
                "family": source.family,
                "direction": source.direction,
                "recall_at_1": metrics["recall@1"],
                "recall_at_5": metrics["recall@5"],
                "recall_at_10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "mean_rank": metrics["mean_rank"],
                "rank_1": metrics["rank_distribution"]["rank_1"],
                "rank_2_5": metrics["rank_distribution"]["rank_2_5"],
                "rank_6_10": metrics["rank_distribution"]["rank_6_10"],
                "rank_11_50": metrics["rank_distribution"]["rank_11_50"],
                "rank_51_plus": metrics["rank_distribution"]["rank_51_plus"],
                "hypothesis": source.hypothesis,
                "change": source.change,
                "result": source.result,
                "decision": source.decision,
                "source_report": str(source.path.relative_to(ROOT)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "embedding_metrics_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_current(name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT / f"{name}.pdf"
    png_path = OUTPUT / f"{name}.png"
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=240)
    plt.close()


def plot_evolution(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8.4, 4.2))
    primary = [row for row in rows if row["direction"] in {"principal", "esp_to_shi"}]
    inverse = [row for row in rows if row["direction"] == "shi_to_esp"]
    x_primary = [row["order"] for row in primary]
    x_inverse = [row["order"] for row in inverse]
    labels = [row["candidate"] for row in primary]

    plt.plot(x_primary, [row["recall_at_1"] for row in primary], marker="o", label="R@1 principal / espanol -> shiwilu")
    plt.plot(x_primary, [row["mrr"] for row in primary], marker="s", label="MRR principal / espanol -> shiwilu")
    plt.plot(x_inverse, [row["recall_at_1"] for row in inverse], marker="o", linestyle="--", label="R@1 shiwilu -> espanol")
    plt.plot(x_inverse, [row["mrr"] for row in inverse], marker="s", linestyle="--", label="MRR shiwilu -> espanol")

    plt.xticks(x_primary, labels, rotation=28, ha="right", fontsize=8)
    plt.ylim(0, 1.0)
    plt.ylabel("Valor de metrica")
    plt.title("Evolucion experimental del modelo de embeddings")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="lower right", fontsize=7)
    save_current("embedding_evolution_r1_mrr")


def plot_final_recall(rows: list[dict[str, Any]]) -> None:
    final_rows = [row for row in rows if row["candidate"] == "v3 iterativo bidir."]
    labels = ["R@1", "R@5", "R@10", "MRR"]
    directions = ["esp_to_shi", "shi_to_esp"]
    direction_labels = ["Castellano -> shiwilu", "Shiwilu -> castellano"]
    metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    width = 0.35
    x = list(range(len(labels)))

    plt.figure(figsize=(7.2, 4.4))
    for index, direction in enumerate(directions):
        row = next(item for item in final_rows if item["direction"] == direction)
        offset = (index - 0.5) * width
        plt.bar([value + offset for value in x], [row[metric] for metric in metrics], width=width, label=direction_labels[index])

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title("Metricas finales del modelo v3")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    save_current("embedding_v3_final_metrics")


def plot_rank_distribution(rows: list[dict[str, Any]]) -> None:
    final_rows = [row for row in rows if row["candidate"] == "v3 iterativo bidir."]
    buckets = ["rank_1", "rank_2_5", "rank_6_10", "rank_11_50", "rank_51_plus"]
    labels = ["Rank 1", "Rank 2-5", "Rank 6-10", "Rank 11-50", "Rank 51+"]
    directions = ["esp_to_shi", "shi_to_esp"]
    direction_labels = ["Castellano -> shiwilu", "Shiwilu -> castellano"]
    width = 0.35
    x = list(range(len(labels)))

    plt.figure(figsize=(7.6, 4.4))
    for index, direction in enumerate(directions):
        row = next(item for item in final_rows if item["direction"] == direction)
        offset = (index - 0.5) * width
        plt.bar([value + offset for value in x], [row[bucket] for bucket in buckets], width=width, label=direction_labels[index])

    plt.xticks(x, labels)
    plt.ylabel("Consultas")
    plt.title("Distribucion de rangos del modelo v3")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    save_current("embedding_v3_rank_distribution")


def plot_negative_mix() -> None:
    report = read_json(
        REPORTS
        / "controlled_hn"
        / "v3_iterative_hn_e5_base_bidirectional"
        / "hard_negatives_v3_iterative_hn_e5_base_bidirectional_report.json"
    )
    counts = report["counts"]["difficulty_counts"]
    labels = ["Medium negatives", "Hard negatives"]
    values = [counts["medium"], counts["hard"]]

    plt.figure(figsize=(5.8, 4.2))
    plt.bar(labels, values)
    plt.ylabel("Filas negativas")
    plt.title("Composicion de negativos usados en v3")
    plt.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        plt.text(index, value + 40, str(value), ha="center")
    save_current("embedding_v3_negative_mix")


def test_parallel_rows() -> list[dict[str, str]]:
    with SPLIT_XL_TEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    complete_rows = [
        row
        for row in rows
        if row["ESP_normalizado"].strip() and row["SHIWILU_normalizado"].strip()
    ]
    return sorted(complete_rows, key=lambda item: (item.get("source", ""), item["group_id"], item["pair_id"]))


def plot_v3_embedding_space() -> None:
    from sentence_transformers import SentenceTransformer

    rows = test_parallel_rows()
    spanish_texts = [f"query: {row['ESP_normalizado']}" for row in rows]
    shiwilu_texts = [f"passage: {row['SHIWILU_normalizado']}" for row in rows]

    model = SentenceTransformer(str(V3_XL_MODEL))
    spanish_embeddings = model.encode(
        spanish_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    shiwilu_embeddings = model.encode(
        shiwilu_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    projection = PCA(n_components=2, random_state=42).fit_transform(
        list(spanish_embeddings) + list(shiwilu_embeddings)
    )
    spanish_points = projection[: len(rows)]
    shiwilu_points = projection[len(rows) :]

    plt.figure(figsize=(7.8, 5.0))
    for esp_point, shi_point in zip(spanish_points, shiwilu_points):
        plt.plot(
            [esp_point[0], shi_point[0]],
            [esp_point[1], shi_point[1]],
            color="#b8bcc2",
            linewidth=0.32,
            alpha=0.24,
            zorder=1,
        )

    plt.scatter(
        spanish_points[:, 0],
        spanish_points[:, 1],
        s=9,
        marker="o",
        color="#1f77b4",
        alpha=0.68,
        label="Castellano",
        zorder=2,
    )
    plt.scatter(
        shiwilu_points[:, 0],
        shiwilu_points[:, 1],
        s=11,
        marker="^",
        color="#d62728",
        alpha=0.68,
        label="Shiwilu",
        zorder=3,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Proyeccion PCA del espacio bilingue v3 ({len(rows)} pares)")
    plt.grid(alpha=0.18)
    plt.legend(fontsize=8, loc="best")
    save_current("embedding_v3_space_pca")


def main() -> None:
    rows = metric_rows()
    write_csv(rows)
    plot_evolution(rows)
    plot_final_recall(rows)
    plot_rank_distribution(rows)
    plot_negative_mix()
    plot_v3_embedding_space()


if __name__ == "__main__":
    main()
