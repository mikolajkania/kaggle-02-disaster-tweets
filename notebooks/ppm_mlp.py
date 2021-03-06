import itertools
import papermill as pm

custom_embeddings_dim_params = [16, 32, 64]
mlp_hidden_layer_units_params = [16, 32, 64, 128, 256]
dropout_params = [0.0, 0.2, 0.33, 0.5]
batch_size_param = [32, 64, 128]
iter_count_param = [1, 2]

combinations = list(itertools.product(custom_embeddings_dim_params, mlp_hidden_layer_units_params, dropout_params,
                                      batch_size_param, iter_count_param))

for c in combinations:
    print(c)
    pm.execute_notebook(
        'mlp.ipynb',
        f'mlp_out_e{c[0]}_h{c[1]}_d{c[2]}_b{c[3]}_i{c[4]}.ipynb',
        parameters=dict(embedding_dim=c[0], hidden_layer_units=c[1], dropout_rate=c[2],
                        batch_size=c[3], iter_count=c[4])
    )
