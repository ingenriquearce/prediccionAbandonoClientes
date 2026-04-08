from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experimentos import REGISTRY_DIR, HISTORIAL_PATH, asegurar_estructura_mlops

CHAMPION_PATH = REGISTRY_DIR / 'champion.json'
METRICA_PRINCIPAL = 'f1_churn'


def cargar_champion_actual() -> dict | None:
    if not CHAMPION_PATH.exists():
        return None
    return json.loads(CHAMPION_PATH.read_text(encoding='utf-8'))


def promover_a_champion(
    run_id: str,
    nombre_modelo: str,
    metricas: dict,
    pipeline_path: str,
    metadata: dict,
) -> dict:
    asegurar_estructura_mlops()
    champion = {
        'run_id': run_id,
        'modelo': nombre_modelo,
        'metrica_principal': METRICA_PRINCIPAL,
        'valor_metrica': metricas[METRICA_PRINCIPAL],
        'ruta_pipeline': pipeline_path,
        'accuracy': metricas['accuracy'],
        'precision_churn': metricas['precision_churn'],
        'recall_churn': metricas['recall_churn'],
        'f1_churn': metricas['f1_churn'],
        'roc_auc': metricas['roc_auc'],
        'cross_val_mean': metricas['cross_val_mean'],
        'fecha_entrenamiento': metadata['fecha_entrenamiento'],
    }
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2, ensure_ascii=False), encoding='utf-8')
    return champion


def evaluar_promocion(
    run_id: str,
    nombre_modelo: str,
    metricas: dict,
    pipeline_path: str,
    metadata: dict,
) -> dict:
    champion_actual = cargar_champion_actual()

    if champion_actual is None:
        nuevo_champion = promover_a_champion(run_id, nombre_modelo, metricas, pipeline_path, metadata)
        return {
            'fue_promovido': True,
            'motivo': 'No existía un champion previo. Este experimento se convirtió en el primer modelo productivo.',
            'champion_anterior': None,
            'champion_actual': nuevo_champion,
        }

    nuevo_valor = metricas[METRICA_PRINCIPAL]
    valor_actual = champion_actual['valor_metrica']

    if nuevo_valor > valor_actual:
        nuevo_champion = promover_a_champion(run_id, nombre_modelo, metricas, pipeline_path, metadata)
        return {
            'fue_promovido': True,
            'motivo': (
                f"El nuevo experimento mejoró {METRICA_PRINCIPAL} de "
                f"{valor_actual:.4f} a {nuevo_valor:.4f}."
            ),
            'champion_anterior': champion_actual,
            'champion_actual': nuevo_champion,
        }

    return {
        'fue_promovido': False,
        'motivo': (
            f"El champion actual mantiene mejor {METRICA_PRINCIPAL}: "
            f"{valor_actual:.4f} vs {nuevo_valor:.4f}."
        ),
        'champion_anterior': champion_actual,
        'champion_actual': champion_actual,
    }


def obtener_historial_experimentos(limit: int = 10) -> pd.DataFrame:
    if not HISTORIAL_PATH.exists():
        return pd.DataFrame()

    historial = pd.read_csv(HISTORIAL_PATH)
    historial = historial.sort_values(by='fecha', ascending=False).head(limit)
    return historial
