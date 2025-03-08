import os
import json
import matplotlib.pyplot as plt

def load_evaluation_data(base_path, models):
    data = {}
    for model in models:
        file_path = os.path.join(base_path, model, "evaluation_results.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data[model] = json.load(f)
    return data

def plot_results(data, metric="F1-Score", filter_by="confidence"):
    categories = list(next(iter(data.values())).keys())
    fig, axes = plt.subplots(1, len(categories), figsize=(15, 5), sharey=True)
    
    category_names = {325135381: "Village", 324385398: "Countryside", 325135402: "City", 325015381: "Forrest"}
    model_names = {"output_urban": "Urban", 
                   "output_forrest": "Forest", 
                   "output_ablation_no_segmentation": "w/o Segmentation", 
                   "output_ablation_no_pretraining": "w/o Pretraining", 
                   "output_ablation_no_postprocessing": "w/o Postprocessing *", 
                   "output_100m_forest": "Forest (100m)", 
                   "output_100m_combined": "Combined (100m)", 
                   "output_150m_forest": "Forest (150m)", 
                   "output_150m_combined": "Combined (150m)",                    
                   "output_combined": "Combined"}
    for i, (ax, category) in enumerate(zip(axes, categories)):
        for model, values in data.items():
            
            points = values[category]
            if filter_by == "confidence" and metric == "F1-Score":
                # Remove entries with iou != 0.5
                points = [(c, f) for i, c, p, f in points if i == 0.5]
                confidence, f1 = zip(*points)                
                ax.plot(confidence, f1, linestyle='dashed', marker='o', label=model_names[model])
            if filter_by == "iou" and metric == "F1-Score":
                points = [(i, f) for i, c, p, f in points if c == 0.3]
                iou, f1 = zip(*points)
                ax.plot(iou, f1, linestyle='dashed', marker='o', label=model_names[model])   
            if metric == "Precision":
                points = [(c, p) for i, c, p, f in points if i == 0.5]
                confidence, prec = zip(*points)
                ax.plot(confidence, prec, linestyle='dashed', marker='o', label=model_names[model])   
            for xi, yi in points:
                ax.text(xi, yi, f"{yi:.2f}", fontsize=8, verticalalignment='bottom')
        
        ax.set_xlabel(f"{filter_by.capitalize()} Threshold")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(category_names[int(category)])
    if filter_by == "confidence":
        axes[0].set_ylabel(f"{metric} @ {'IoU=0.5'}")
    if filter_by == "iou":
        axes[0].set_ylabel(f"{metric} @ {'Confidence=0.3'}")
    if metric == "Precision":
        axes[0].set_ylabel(f"{metric} @ {'IoU=0.5'}")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

# Example usage
base_path = "visualizations"

selected_models = ["output_urban", "output_forrest", "output_combined"]
data = load_evaluation_data(base_path, selected_models)
plot_results(data, metric="F1-Score", filter_by="confidence")

selected_models = ["output_combined", "output_100m_combined", "output_150m_combined"]
data = load_evaluation_data(base_path, selected_models)
plot_results(data, metric="F1-Score", filter_by="confidence")

selected_models = ["output_combined", "output_ablation_no_pretraining", "output_ablation_no_postprocessing", "output_ablation_no_segmentation"]
data = load_evaluation_data(base_path, selected_models)
plot_results(data, metric="F1-Score", filter_by="confidence")

selected_models = ["output_combined", "output_ablation_no_segmentation", "output_ablation_no_postprocessing", "output_urban"]
data = load_evaluation_data(base_path, selected_models)
plot_results(data, metric="F1-Score", filter_by="iou")

selected_models = ["output_combined", "output_ablation_no_postprocessing", "output_ablation_no_segmentation", "output_urban"]
data = load_evaluation_data(base_path, selected_models)
plot_results(data, metric="Precision", filter_by="confidence")

