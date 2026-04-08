from __future__ import annotations

import csv
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MLOPS_DIR = BASE_DIR / 'mlops'
EXPERIMENTOS_DIR = MLOPS_DIR / 'experimentos'
REGISTRY_DIR = MLOPS_DIR / 'registry'
HISTORIAL_PATH = REGISTRY_DIR / 'historial_modelos.csv'


def asegurar_estructura_mlops() -> None:
    EXPERIMENTOS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def slug_modelo(nombre_modelo: str) -> str:
    texto = unicodedata.normalize('NFKD', nombre_modelo).encode('ascii', 'ignore').decode('ascii')
    return texto.lower().replace(' ', '_')


def generar_run_id(nombre_modelo: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'exp_{timestamp}_{slug_modelo(nombre_modelo)}'


def guardar_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def guardar_experimento(
    run_id: str,
    pipeline,
    metricas: dict[str, Any],
    parametros: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    asegurar_estructura_mlops()
    run_dir = EXPERIMENTOS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = run_dir / 'pipeline.joblib'
    metricas_path = run_dir / 'metricas.json'
    parametros_path = run_dir / 'parametros.json'
    metadata_path = run_dir / 'metadata.json'

    joblib.dump(pipeline, pipeline_path)
    guardar_json(metricas_path, metricas)
    guardar_json(parametros_path, parametros)
    guardar_json(metadata_path, metadata)

    return {
        'run_dir': str(run_dir.relative_to(BASE_DIR)),
        'pipeline_path': str(pipeline_path.relative_to(BASE_DIR)),
        'metricas_path': str(metricas_path.relative_to(BASE_DIR)),
        'parametros_path': str(parametros_path.relative_to(BASE_DIR)),
        'metadata_path': str(metadata_path.relative_to(BASE_DIR)),
    }


def registrar_en_historial(
    run_id: str,
    nombre_modelo: str,
    metricas: dict[str, Any],
    fue_promovido: bool,
) -> None:
    asegurar_estructura_mlops()

    archivo_existe = HISTORIAL_PATH.exists()
    with HISTORIAL_PATH.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'run_id',
                'fecha',
                'modelo',
                'accuracy',
                'precision_churn',
                'recall_churn',
                'f1_churn',
                'roc_auc',
                'cross_val_mean',
                'promovido',
            ],
        )
        if not archivo_existe:
            writer.writeheader()

        writer.writerow(
            {
                'run_id': run_id,
                'fecha': datetime.now().isoformat(timespec='seconds'),
                'modelo': nombre_modelo,
                'accuracy': metricas['accuracy'],
                'precision_churn': metricas['precision_churn'],
                'recall_churn': metricas['recall_churn'],
                'f1_churn': metricas['f1_churn'],
                'roc_auc': metricas['roc_auc'],
                'cross_val_mean': metricas['cross_val_mean'],
                'promovido': fue_promovido,
            }
        )
