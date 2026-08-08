

# FREESH: Planificación Justa, Eficiente en Recursos y Energía para el Servicio de LLM en GPUs Heterogéneas

> Un sistema modular de **enrutamiento + planificación** para inferencia de LLM que optimiza conjuntamente la **energía/carbono**, los **SLOs** y la **equidad** en clústeres de GPUs heterogéneas y geo-distribuidos.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00807-b31b1b.svg)](https://arxiv.org/abs/2511.00807)

## Aspectos Destacados

- **Diseño coordinado en tres capas**
  - **Nivel de pool (larga escala temporal)**: Optimizador basado en MILP para la ubicación entre regiones, partición de solicitudes y selección del modo de paralelismo de tensores (TP), con objetivos intercambiables de **energía** o **carbono**.
  - **Nivel de GPU (segundos)**: Control dinámico de frecuencia **MIAD (Incremento Multiplicativo, Decremento Aditivo)** para reducir el consumo de energía mientras se cumplen los SLOs.
  - **Nivel de solicitud (subsegundo)**: Planificador preemptivo **LLF (Primero la Menor Holgura)** para garantizar equidad y reducir la latencia de cola (reduce el bloqueo de cabeza de cola de FCFS).
- **Seis “parámetros de control”**: selección de sitio, partición, tipo de solicitud, modo de ejecución (TP / emparejamiento modelo-GPU), frecuencia (DVFS/MIAD) y política de planificación (LLF).
- **Enrutamiento consciente de la intensidad de carbono**: aprovecha las señales de carbono espacio-temporales de la red para desplazar la carga de trabajo a regiones/periodos más limpios.
- **Listo para usar (plug-and-play)**: se integra con entornos de ejecución de inferencia comunes (p. ej., vLLM) y múltiples solucionadores (p. ej., Gurobi / backends de código abierto).

> Método completo y evaluaciones: **https://arxiv.org/abs/2511.00807**
