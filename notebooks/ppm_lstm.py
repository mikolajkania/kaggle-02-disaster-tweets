import itertools
import papermill as pm

custom_embeddings_dim_params = [32, 64]
lstm_units_params = [16, 32]
hidden_layer_units_params = [64, 32, 96]
dropout_params = [0.2, 0.5]
batch_size_params = [32]
iter_count_params = [1, 2]

combinations = list(itertools.product(custom_embeddings_dim_params, lstm_units_params, hidden_layer_units_params,
                                      dropout_params, batch_size_params, iter_count_params))

for c in combinations:
    print(c)
    pm.execute_notebook(
        'lstm.ipynb',
        f'lstm_out_e{c[0]}_l{c[1]}_h{c[2]}_d{c[3]}_b{c[4]}_i{c[5]}.ipynb',
        parameters=dict(embedding_dim=c[0], lstm_layer_units=c[1], hidden_layer_units=c[2],
                        dropout_rate=c[3], batch_size=c[4], iter_count=c[5])
    )
