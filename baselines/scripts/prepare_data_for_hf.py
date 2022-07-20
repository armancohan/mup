import argparse
import json
import pathlib
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset


def get_text_from_sections(
    sections: List[Dict[str, str]], abstract: Optional[str] = None, section_sep_token="</sec>"
) -> str:
    """get a flat text from list of sections"""
    text = ""
    if abstract is not None:
        text = abstract + f" {section_sep_token}"
    for section in sections:
        section_text = section["text"]
        text += section_text + f" {section_sep_token}"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="allenai/mup-full", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()

    data = load_dataset(args.dataset_name)

    example_by_id = {}
    splits = ["train", "validation"]
    for split in splits:
        results = []
        for example in data[split]:
            text = get_text_from_sections(example["sections"], example["abstractText"])
            paper_id = example["paper_id"]
            summaries = example.get("summaries") or None
            title = example.get("title") or ""

            for summary_index, summary in enumerate(summaries):
                results.append(
                    {
                        "paper_name": title,
                        "text": text,
                        "summary": summary,
                        "paper_id": paper_id + f"___{summary_index}",
                    }
                )

        df = pd.DataFrame(results)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_csv_path = f"{args.output_dir}/{split}.csv"
        df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
