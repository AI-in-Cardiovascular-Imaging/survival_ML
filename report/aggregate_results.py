import os


def aggregate_results(config, results):
        out_file = os.path.join(config.meta.out_dir, 'aggregate_results.xlsx')
        results.drop(columns=["auc", 'evaluation_times', 'truncation_time'], inplace=True)
        mean_results = results.groupby(["Scaler", "Selector", "Model"]).mean().drop('Seed', axis=1).reset_index()
        std_results = results.groupby(["Scaler", "Selector", "Model"]).std().drop('Seed', axis=1).reset_index()

        aggregated_results = mean_results.merge(
            std_results, on=["Scaler", "Selector", "Model"], suffixes=('_mean', '_std')
        )
        aggregated_results.to_excel(out_file, index=False, float_format='%.3f')

        return aggregated_results