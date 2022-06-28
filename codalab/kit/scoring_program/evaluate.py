#!/usr/bin/env python
import argparse
import os.path
import sys

import numpy as np
from tqdm import tqdm


def install_dependencies():
    try:
        os.system("pip install rouge-score")
        os.system("pip install pandas")
    except Exception as e:
        print(e)
        print("Error occurred while installing dependencies. Please install manually.")
        sys.exit(1)


try:
    # installing dependencies
    import pandas as pd
    from rouge_score import rouge_scorer

    # os.system("pip install bert-score")
except Exception as e:
    install_dependencies()

# input_dir = sys.argv[1]
# output_dir = sys.argv[2]

# submit_dir = os.path.join(input_dir, "res")
# truth_dir = os.path.join(input_dir, "ref")

# if not os.path.isdir(submit_dir):
#     print(f"{submit_dir} doesn't exist")


def evaluate_metrics(test_annotation_file, user_submission_file, use_bertscore=False):

    metrics = ["rouge1", "rouge2", "rougeL"]
    ground_truth_df = pd.read_csv(test_annotation_file)
    print(f"processing ground truth file + {test_annotation_file}")
    submission_df = pd.read_csv(user_submission_file)
    print(f"processing submission file {user_submission_file}")

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f": [], "rouge1_r": [], "rouge2_f": [], "rouge2_r": [], "rougeL_f": [], "rougeL_r": []}
    if len(ground_truth_df["paper_id"].unique()) != len(submission_df["paper_id"].unique()):
        print(
            "Warning, number of unique 'paper_id's in submission is not equal to number of unique 'paper_id's in "
            "ground truth"
        )

    bertscore_summaries = ([], [])
    for index, ground_truth_row in tqdm(
        ground_truth_df.iterrows(), total=ground_truth_df.shape[0], desc="evaluating summaries..."
    ):
        ground_truth_summary = ground_truth_row["summary"]
        article_id = ground_truth_row["paper_id"]
        submission_summary_row = submission_df.loc[submission_df["paper_id"] == article_id]
        if submission_summary_row.empty:
            print(f"paper with id '{article_id}' wasn't found in submission")
            raise Exception(f"paper with id '{article_id}' wasn't found in submission")
        elif len(submission_summary_row.index) != 1:
            print(f"More than one summary submission for paper with id '{article_id}'")
            raise Exception(f"More than one summary submission for paper with id '{article_id}'")

        submission_summary = submission_summary_row.iloc[0]["summary"]

        # print(f"evaluating summary for article with id '{article_id}'")
        scores = scorer.score(ground_truth_summary.strip(), submission_summary.strip())

        for rouge_metric in metrics:
            results[rouge_metric + "_f"].append(scores[rouge_metric].fmeasure)
            results[rouge_metric + "_r"].append(scores[rouge_metric].recall)

        if use_bertscore:
            from bert_score import score as bert_score_func

            bertscore_summaries[0].append(submission_summary.strip())
            bertscore_summaries[1].append(ground_truth_summary.strip())

    metrics_scores = {
        rouge_metric: np.average(rouge_metric_scores) for (rouge_metric, rouge_metric_scores) in results.items()
    }
    avg_score = [metric_score for metric_name, metric_score in metrics_scores.items() if "_f" in metric_name]

    if use_bertscore:
        (P, R, F) = bert_score_func(cands=bertscore_summaries[0], refs=bertscore_summaries[1], lang="en")
        metrics_scores["BERTScore_P"] = P.mean().item()
        metrics_scores["BERTScore_R"] = R.mean().item()
        metrics_scores["BERTScore_F"] = F.mean().item()
        avg_score.append(F.mean().item())

    metrics_scores["Metrics_Avg"] = sum(avg_score) / len(avg_score)
    return metrics_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_file", help="path to submission file")
    parser.add_argument("ground_truth", help="path to ground truth file")
    parser.add_argument("--output_dir", help="path to output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = os.path.join(args.output_dir, "scores.txt")

    truth_file = args.ground_truth
    submission_answer_file = args.submission_file
    if not os.path.exists(submission_answer_file):
        print(
            f"Submission file with name 'testing.csv' doesn't exist, please make sure to submit a single zip file that contains 'testing.csv'"
        )
        raise Exception(
            f"Submission file with name 'testing.csv' doesn't exist, please make sure to submit a single zip file that contains 'testing.csv'"
        )

    eval_scores = evaluate_metrics(test_annotation_file=truth_file, user_submission_file=submission_answer_file)
    with open(output_filename, "w") as output_file:
        for metric, metric_score in eval_scores.items():
            output_file.write(f"{metric}:{(metric_score * 100):.2f}\n")
    output_file.close()


if __name__ == "__main__":
    main()
