import papermill as pm
import os
import itertools

notebook_path = 'eval-roberta-base-go_emotions.ipynb'
output_html_dir = 'output_html'
os.makedirs(output_html_dir, exist_ok=True)

# Define the parameter combinations
model_names = ['hallucination_evaluation_model','albert-base-V2']
max_lengths = [16, 32, 64]
learning_rates = [2.00E-05, 3.00E-05, 5.00E-05]
batch_sizes = [16, 32, 64]

# Iterate over all combinations
for model_name, max_length, lr, batch_size in itertools.product(model_names, max_lengths, learning_rates, batch_sizes):
    output_notebook = f'executed_{model_name}_maxlen{max_length}_lr{lr}_batch{batch_size}.ipynb'
    output_html = f'{model_name}_maxlen{max_length}_lr{lr}_batch{batch_size}.html'

    # Check if the output notebook file already exists
    if not os.path.exists(output_notebook):
        # Execute the notebook with the current set of parameters
        pm.execute_notebook(
            notebook_path,
            output_notebook,
            parameters={
                'argspretrained_model_name': model_name,
                'argsmaxlengthToPass': max_length,
                'argslr': lr,
                'argsbatch_size': batch_size
            }
        )
    else:
        print(f"Skipped: Combination {model_name}, maxlen {max_length}, lr {lr}, batch {batch_size} already exists.")
