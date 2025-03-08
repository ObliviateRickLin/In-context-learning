import sys; import os; sys.path.append("."); import eval; model_path = sys.argv[1]; print(f"Evaluating model: {model_path}"); metrics = eval.get_run_metrics(model_path); print(metrics)
