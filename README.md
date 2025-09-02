# Análisis y Predicción de Tsunamis: Un Enfoque Educativo

Este proyecto aplica **Machine Learning** a datos históricos de terremotos y tsunamis con un enfoque **didáctico y divulgativo**. No busca predecir tsunamis reales, sino mostrar cómo la ciencia de datos puede ayudar a comprender patrones y apoyar en la educación científica.

## Contenido
- **Dataset:** Datos abiertos de NOAA (NCEI). Mediante APIs 
- **Notebooks:** Obtención, limpieza, EDA, modelado y evaluación.  
- **Tipo de problema:** Clasificación
- **Modelos probados:** Logistic Regression, Random Forest, XGBoost, **LightGBM**, CatBoost, SVM, kNN.  
- **Técnicas:** SMOTE para balanceo, GridSearch para optimización.  
- **Resultados:** Recall ≈ 0,75 y F1 ≈ 0,73 (modelo LightGBM).  
- **App Streamlit:** Visualización en mapa y predicción puntual (en desarrollo).  
- **Presentación:** Versión final en PDF 

## Limitaciones
El modelo solo considera datos sísmicos. No integra factores geológicos ni oceánicos necesarios para predicciones reales.

## Futuro
- Integración de datos de placas tectónicas y batimetría.  
- Modelos más complejos (deep learning).  
- Ampliar funcionalidades de la aplicación.  
