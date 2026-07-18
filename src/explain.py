import shap
import matplotlib.pyplot as plt

from src.predictor import model

explainer = shap.TreeExplainer(model)


# Global Feature Importance
def plot_global_shap(sample):

    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10,8))

    shap.summary_plot(
        shap_values,
        sample,
        plot_type="bar",
        show=False
    )

    return plt.gcf()


# Local Patient Explanation
def plot_local_shap(sample):

    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10,6))

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=sample.iloc[0],
            feature_names=sample.columns
        ),
        show=False
    )

    return plt.gcf()