import itertools
import papermill as pm

# custom_embeddings_dim_params = [32, 64]
lstm_units_params = [16, 32]
hidden_layer_units_params = [32, 64]
dropout_params = [0.5]
batch_size_params = [32, 128]
iter_count_params = [1, 2]

combinations = list(itertools.product(lstm_units_params, hidden_layer_units_params,
                                      dropout_params, batch_size_params, iter_count_params))

for c in combinations:
    print(c)
    pm.execute_notebook(
        'lstm.ipynb',
        f'lstm_out_l{c[0]}_h{c[1]}_d{c[2]}_b{c[3]}_i{c[4]}.ipynb',
        parameters=dict(lstm_layer_units=c[0], hidden_layer_units=c[1],
                        dropout_rate=c[2], batch_size=c[3], iter_count=c[4])
    )
