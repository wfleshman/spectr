import argparse


from importlib.metadata import entry_points

from llm_merging.evaluation import * 
from llm_merging.data import * 

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    discovered_merges = entry_points(group="llm_merging.merging.Merges")
    loaded_merges = {ep.name: ep.load() for ep in discovered_merges}
    return loaded_merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str, required=True)
    parser.add_argument("--dataset_filepaths", type=str, default=None, nargs='+')
    parser.add_argument("--eval_types", type=str, default=None, nargs='+')
    args = parser.parse_args()
    print(args.merging_method)
    # Load correct merging method 
    loaded_merges = all_merge_handlers()
    merge_method = loaded_merges[args.merging_method](args.merging_method)

    # Call the merge function. The merged model is stored under merging_method object 
    # merge_method.merge()

    if args.dataset_filepaths is not None:
        assert args.eval_types is not None, "If dataset_filepaths is passed, eval_types"

        assert len(args.dataset_filepaths) == len(args.eval_types), "All lists should be of the same length"

    else:
        assert args.eval_types is None, "If dataset_filepaths is not passed, eval_types should not be passed in"
        # Evaluate method on fixed datasets
        evaluate_model(
            merge_method,
            ['data/dbpedia_test.json', 'data/glue_sst2_test.json','data/glue_cola_test.json', 'data/glue_mrpc_test.json', 'data/hellaswag_test.json', 'data/glue_qnli_test.json', 'data/agnews_test.json', 'data/glue_qqp_test.json', 'data/glue_mnli_test.json'],
            ["multiple_choice"] * 9, 
            ["accuracy"] * 9,
        )
