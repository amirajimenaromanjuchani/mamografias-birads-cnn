# Clasificación de mamografías 2D: benigna (BI-RADS 2) vs alta sospecha (BI-RADS 4-5)

Tesis de Ingeniería en Ciencia de Datos e Inteligencia de Negocios, Universidad Privada del Valle.
Autora: Amira Jimena Roman Juchani.

> **Aviso:** proyecto de investigación académica. No es un dispositivo médico ni debe usarse para decisiones clínicas.

## Qué hace
Entrena y compara tres CNN preentrenadas en ImageNet (ResNet-50, DenseNet-121, EfficientNet-B4) que reciben las 4 vistas de una
paciente (L-CC, L-MLO, R-CC, R-MLO) y predicen *benigna* (BI-RADS 2) vs *alta sospecha* (BI-RADS 4-5).
Sigue la guía CLAIM 2024. Datos: Hospital Univalle Zona Norte (Cochabamba).

## Protocolo (no se modifica entre corridas)
- Unidad de análisis = **paciente** (371 pacientes: 311 benignas, 60 alta sospecha; 1484 imágenes).
- Validación cruzada de 5 pliegues estratificada por BI-RADS (2, 4, 5); semilla de entrenamiento 123.
- Entrenamiento en dos fases (cabeza lr 1e-3, luego todo lr 1e-4), BatchNorm congelada, sobremuestreo por clase con corrección de prior.
- Aumento de datos solo en entrenamiento; sin SMOTE, CutMix, mixup ni volteo.
- Umbral de Youden fijado **solo en validación**; métricas por paciente con IC95 % bootstrap (2000 réplicas), DeLong con corrección de Holm,
  calibración (ECE, pendiente, intercepto, Platt anidado) y Grad-CAM (cualitativo y cuantitativo).

## Alcance de los resultados publicados
- **Semilla de partición 42: completa** (3 arquitecturas x 5 pliegues = 15 rondas).
- `splits/splits_publico.csv` incluye también las particiones de las semillas 43 y 44, pero **solo se ejecutó la 42**.
  La corrida de la semilla 43 quedó incompleta y no se publica; se declara como trabajo futuro.
- No hay validación externa ni validación del Grad-CAM con radiólogos.

## Estructura
```
src/config.py                                  # configuración única (rutas, hiperparámetros)
notebooks/                                     # cuaderno de entrenamiento y análisis (reanudable)
splits/splits_publico.csv                      # paciente_id (anónimo), etiqueta y pliegue por semilla
resultados/tablas/                             # métricas OOF, DeLong, subgrupos BI-RADS, vistas, calibración, Grad-CAM cuantitativo
resultados/figuras/                            # calibración, ROC por subgrupo, distribución de probabilidades, curvas de entrenamiento
resultados/{arquitectura}/fold{k}/seed42/      # por ronda: log_epocas.csv, train_ok.json, pred_test.csv, metrics.json
resultados/entorno.txt                         # versiones, GPU y commit usados en el entrenamiento
```

## Datos
Las mamografías, `dataset_final_v1.csv`, los pesos (`best.pt`) y las figuras Grad-CAM por paciente **no se publican**
(datos clínicos privados). Solo se incluye la tabla de particiones con identificadores anónimos.
Sin esos datos el cuaderno no puede reentrenar; los resultados publicados salen de `resultados/`.

## Cómo reproducir
1. Colab con GPU (T4); secreto `GITHUB_TOKEN` (solo si vas a hacer push).
2. Montar Drive con los PNG y el dataset; ajustar `DRIVE_ROOT` en `src/config.py`.
3. Ejecutar el cuaderno en orden. Es reanudable: salta las corridas que ya tienen `metrics.json`.
4. La reproducibilidad con precisión mixta fp16 es razonable, no bit a bit; cambiar de GPU puede mover las métricas en el tercer decimal.

## Resultados
Cifra principal con IC95 %: `resultados/tablas/oof_metricas_seed42.csv`.
Comparación entre arquitecturas: `resultados/tablas/delong_seed42.csv` (usar `p_holm`).
Predicciones por paciente de cada ronda: `resultados/{arquitectura}/fold{k}/seed42/pred_test.csv`.
Versión del código usada en el entrenamiento: campo `commit_repo` de `resultados/entorno.txt`.

## Licencia y cita
Licencia del código: _por definir_. Cómo citar: _se completará al tener el documento final de la tesis_.
