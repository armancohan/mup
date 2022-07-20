import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="")
    parser.add_argument("output_json", help="")
    args = parser.parse_args()

    with open(args.input_json) as fin:
        data = [json.loads(line) for line in fin]

    example_by_id = {}
    for example in data:
        id_ = example["paper_id"]
        if id_ in example_by_id:
            example_by_id[id_]["all_summaries"].append(example["summary"])
        else:
            example_by_id[id_] = {}
            example_by_id[id_]["all_summaries"] = [example["summary"]]
            example_by_id[id_]["paper"] = example["paper"]

    with open(args.output_json, "w") as fout:
        for id_, example in example_by_id.items():
            example["paper"].pop("id")
            fout.write(json.dumps({"paper_id": id_, "summaries": example["all_summaries"], **example["paper"]}) + "\n")


if __name__ == "__main__":
    main()
