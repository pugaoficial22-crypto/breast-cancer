Clasificación de Cáncer de Mama - Random Forest Classifier

clasificación para predecir si un tumor es Benigno (B) o Maligno (M) usando 30 características de imágenes de mama.

Resultados finales

Accuracy: 96.5%
Precision: 97.6%
Recall: 93.0%
F1-Score: 95.2%
ROC-AUC: 99.7%

import joblib

modelo = joblib.load('best_rf_classifier.pkl')

# Ejemplo: primera paciente del dataset (es maligno)
prediccion = modelo.predict([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
                              0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
                              0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38,
                              17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654,
                              0.4601, 0.1189]])

print("Resultado:", "Maligno" if prediccion[0] == 1 else "Benigno")
# Salida: Maligno
