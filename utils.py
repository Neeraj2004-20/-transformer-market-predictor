import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")  # Save the plot as an image
    plt.show()
