from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pandas as pd

from datasets import Dataset
from patched_generation import get_hidden_states, generate_with_patching_all_layers
from utils import get_layer_names, load_model, load_tokenizer, last_relation_word


def activation_patching(entries, model, tokenizer, source, default_decoding):
    target_token_str = "x"
    if source == "patch_e":
        source_str = entries["patch_e"]
    elif source == "patch_r":
        source_str = entries['patch_r']
    else:
        raise ValueError(f"Source {source} not supported")
    target_token_str = "is"
    tokenizer.padding_side = "right"
    hidden_states = get_hidden_states(model, tokenizer, source_str, entries["patch_prompt"])
    tokenizer.padding_side = "left"
    generations = generate_with_patching_all_layers(model, tokenizer, hidden_states, entries["source_prompt"],
                                                    target_token_str, default_decoding)

    entry_count = len(entries["source_prompt"])
    layer_count = len(get_layer_names(model))
    new_entries = {}
    for k in entries.keys():
        new_entries[k] = []
    new_entries["source_layer"] = []
    new_entries["target_layer"] = []
    new_entries["generation"] = []
    for source_layer in range(layer_count):
        for target_layer in range(layer_count):
            for i in range(entry_count):
                for k, v in entries.items():
                    new_entries[k].append(v[i])
                new_entries["source_layer"].append(source_layer)
                new_entries["target_layer"].append(target_layer)
                new_entries["generation"].append(generations[source_layer, target_layer][i])

    return new_entries


def create_source_prompt(dataset, source):
    source_prompt = (
            "Syria: Syria, " +
            "Leonardo DiCaprio: Leonardo DiCaprio, " +
            "Samsung: Samsung,"
        )
    if source == "patch_e":
        return dataset.apply(lambda row: f"{row['r2_template'].format(row['patch_e'])} is", axis=1)
    elif source == "patch_r":
        return dataset.apply(lambda row: row['patch_r_template'].format(row['e2_label']), axis=1)
    else:
        raise ValueError(f"Source {source} combination not supported")

def main(args):
    print(args)
    if not args.input_path:
        args.input_path = f"datasets/{args.model_name}/relation_patch.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)
    dataset["patch_prompt"] = create_source_prompt(dataset, args.source)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    generations = dataset.map(
        activation_patching,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": args.source,
                   "default_decoding": args.default_decoding},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    generations = generations.to_pandas()
    generations = generations.sort_values(["id", "source_layer", "target_layer"])
    generations = generations.reset_index(drop=True)
    if not args.output_path:
        args.output_path = f"datasets/{args.model_name}/entity_relation_patching/{args.source}_test_patching.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2","meta-llama/Meta-Llama-3-8B-Instruct","Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("source", choices=["patch_e", "patch_r"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--default-decoding", action=BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
