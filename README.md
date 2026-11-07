# Clasificación de mamografías 2D: benigna (BI-RADS 2) vs alta sospecha (BI-RADS 4-5)

Tesis de Ingeniería en Ciencia de Datos, Universidad Privada del Valle.
Autora: Amira Jimena Roman Juchani.

> **Aviso:** proyecto de investigación académica. No es un dispositivo médico ni debe usarse para decisiones clínicas.

## Qué hace
Entrena y compara tres CNN preentrenadas (ResNet-50, DenseNet-121, EfficientNet-B4) que reciben las 4 vistas de una paciente
(L-CC, L-MLO, R-CC, R-MLO) y predicen *benigna* (BI-RADS 2) vs *alta sospecha* (BI-RADS 4-5). Sigue la guía CLAIM 2024.

## Protocolo (no se modifica entre corridas)
- Unidad de análisis = **paciente** (371 pacientes: 311 benignas, 60 alta sospecha; 1484 imágenes).
- Validación cruzada de 5 pliegues estratificada por BI-RADS; semillas de partición 42-44; semilla de entrenamiento 123.
- Entrenamiento en dos fases (cabeza lr 1e-3, luego todo lr 1e-4), BatchNorm congelada, sobremuestreo por clase con corrección de prior.
- Aumento de datos solo en entrenamiento; sin SMOTE, CutMix, mixup ni volteo.
- Umbral de Youden fijado **solo en validación**; métricas por paciente con IC bootstrap, DeLong + Holm, calibración y Grad-CAM.

## Estructura
```
src/config.py            # configuración única (rutas, hiperparámetros)
notebooks/               # entrenamiento_tesis.ipynb (reanudable)
<splits_publico>.csv     # paciente_id, etiqueta y pliegues por semilla (ver CONFIG["paths"]["splits_publico"])
resultados/tablas/       # métricas OOF, DeLong, calibración, Grad-CAM cuantitativo
resultados/figuras/      # figuras sueltas (sin imágenes de mamografías)
resultados/entorno.txt   # versiones, GPU y commit usados
```

## Datos
Las mamografías y `dataset_final_v1.csv` **no se publican** (datos clínicos privados). Solo se incluye la tabla de particiones.
Sin esos datos el cuaderno no puede reentrenar; los resultados publicados salen de `resultados/`.

## Cómo reproducir
1. Colab con GPU; secreto `GITHUB_TOKEN` (solo si vas a hacer push).
2. Montar Drive con los PNG y el dataset; ajustar `DRIVE_ROOT` en `src/config.py`.
3. Ejecutar el cuaderno en orden. Es reanudable: salta las corridas con `metrics.json`.
4. La reproducibilidad con AMP fp16 es razonable, no bit a bit; cambiar de GPU puede mover las métricas en el tercer decimal.

## Resultados
Ver `resultados/tablas/oof_metricas_seed42.csv` (cifra principal con IC95 %). Commit/tag usado: `<completar al publicar>`.

## Licencia y cita
`<elegir licencia del código>` · Cómo citar: `<completar al tener el documento final de la tesis>`.
