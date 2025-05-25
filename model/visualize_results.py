import pandas as pd
import matplotlib.pyplot as plt
import os

# Ruta al archivo de resultados
csv_path = "runs/detect/recycling_detection_v1/results.csv"

# Leer CSV
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Limpiar columnas

# Graficar pérdidas
plt.figure(figsize=(10, 6))
plt.plot(df['train/box_loss'], label='Box Loss')
plt.plot(df['train/cls_loss'], label='Cls Loss')
plt.plot(df['train/dfl_loss'], label='DFL Loss')
plt.title("Pérdidas de entrenamiento por época")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Graficar métricas de validación
plt.figure(figsize=(10, 6))
plt.plot(df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.title("Precisión media (mAP) por época")
plt.xlabel("Época")
plt.ylabel("mAP")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación automática del comportamiento
def evaluar_entrenamiento(df):
    mAP50 = df['metrics/mAP50(B)']
    box_loss = df['train/box_loss']

    if mAP50.iloc[-1] < 0.6:
        print("❌ Possible underfitting: final mAP50 is low.")
    elif box_loss.iloc[-1] > 1.0:
        print("❌ Possible underfitting: cash loss remains high.")
    elif mAP50.iloc[-1] < mAP50.max() * 0.9:
        print("⚠️ Possible overfitting: Accuracy has decreased from its peak.")
    elif mAP50.is_monotonic_increasing:
        print("✅ No clear signs of overfitting: mAP rises consistently.")
    elif box_loss.iloc[-1] < 0.5 and mAP50.iloc[-1] < 0.8:
        print("⚠️ Very little loss but low mAP — possible overfitting without generalization.")
    
    else:
         print("✅ For the moment everything seems to be fine.")

evaluar_entrenamiento(df)
