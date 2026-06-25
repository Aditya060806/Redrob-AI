import shap
import numpy as np

print("SHAP version:", shap.__version__)
# Create a dummy Explanation object and see how we can plot waterfall
shap_val = np.random.randn(10)
base_value = 0.5
data = np.random.randn(10)
feature_names = [f"feat_{i}" for i in range(10)]

shap_exp = shap.Explanation(
    values=shap_val,
    base_values=base_value,
    data=data,
    feature_names=feature_names
)

try:
    print("Trying shap.plots.waterfall...")
    shap.plots.waterfall(shap_exp, show=False)
    print("Success with shap.plots.waterfall!")
except Exception as e:
    print("Failed shap.plots.waterfall:", e)

try:
    print("Trying shap.waterfall_plot...")
    shap.waterfall_plot(shap_exp, show=False)
    print("Success with shap.waterfall_plot!")
except Exception as e:
    print("Failed shap.waterfall_plot:", e)
