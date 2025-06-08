import os 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from torch.nn import Module
from torch.utils.data import DataLoader

class Prediction:
    def __init__(self) -> None:
        self.path_save_test = "results_test"
        self.y_true = []
        self.y_pred = []
    
    def test_model(
            self,
            model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            clases: list,
            device: str = "cpu",
            number_samper: int = 5,
        ) -> None:
        os.makedirs(self.path_save_test, exist_ok=True)

        images_shown = 0
        _, axs = plt.subplots(1, number_samper, figsize=(15, 5))

        model.eval()
        with torch.no_grad():
            for batch in test_loader:   
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)

                self.y_true.extend(labels.cpu().numpy())
                self.y_pred.extend(preds.cpu().numpy())

                for i in range(images.size(0)):
                    if images_shown < number_samper:
                        img = images[i].cpu().permute(1, 2, 0).numpy()
                        axs[images_shown].imshow(img)
                        axs[images_shown].set_title(
                            f"True: {clases[labels[i]]}\nPred: {clases[preds[i]]}"
                        )
                        axs[images_shown].axis("off")
                        images_shown += 1
                    else:
                        break
                if images_shown >= number_samper:
                    break

        plt.tight_layout()
        plt.savefig(f"{self.path_save_test}/reporte_visual.png")
        plt.close()
    
    def generate_visual_metrics_report(
            self, 
            clases: list) -> None:
        
        # save classification whint image
        report_dict = classification_report(
            self.y_true,
            self.y_pred, 
            target_names=clases, 
            output_dict=True
        )

        df_report = pd.DataFrame(report_dict).transpose()
        df_report = df_report.round(2)

        fig, ax = plt.subplots(figsize=(10, len(clases) * 0.6 + 2))
        ax.axis("off")
        table = ax.table(
            cellText=df_report.values,
            colLabels=df_report.columns,
            rowLabels=df_report.index,
            loc="center",
            cellLoc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title("clasification report", fontsize=14)
        plt.savefig(f"{self.path_save_test}/clasification_report.png")
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(f"{self.path_save_test}/Confusion_Matrix.png")
        plt.close()


        # F1-score bar chart
        _, _, f1_scores, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, labels=range(len(clases))
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(x=clases, y=f1_scores)
        plt.ylabel("F1 Score")
        plt.ylim(0, 1)
        plt.title("F1 Score for Clas")
        plt.grid(True)
        plt.savefig(f"{self.path_save_test}/f1_score_bar_chart.png")
        plt.close()