import os

import json

OUTPUT_DIR = "output"


def save_model_results_to_csv(results_df, scenario_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    safe_scenario_name = scenario_name.lower().replace(" ", "_")
    output_file = os.path.join(
        output_dir, f"{safe_scenario_name}_model_results.csv")

    results_to_save = results_df.copy()
    if 'Best Params' in results_to_save.columns:
        results_to_save['Best Params'] = results_to_save['Best Params'].apply(
            lambda x: json.dumps(
                x, sort_keys=True) if isinstance(x, dict) else x
        )
    results_to_save.to_csv(output_file, index=False)
    print(f"\nSaved model results to: {output_file}")
