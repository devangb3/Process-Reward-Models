import json
import re

import pandas as pd

DEBUG_FILE = "./Experiments/prm_debug_steps-250.jsonl"

final_answer_re = re.compile(r"Final Answer:\s*(\S+)")
processed_rows = []

with open(DEBUG_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)

        # Extract final answer from text
        text = obj.get("text", "")

        m = final_answer_re.search(text)
        if m:
          answer = m.group(1)
          obj["final_answer"] = answer

        processed_rows.append(obj)


df = pd.DataFrame(processed_rows)
print(df)

def find_tiebreak_case_studies(df):
    """
    Finds problems where the there is a tie in answer voting.
    """
    valid_groups = []

    # Group by problem text
    for problem_text, group in df.groupby("problem"):

        # Count occurrences per final answer
        answer_counts = group.groupby("final_answer").size()

        # Only keep problems with >1 unique answer
        if len(answer_counts) <= 1:
            continue

        # Check if all counts are equal
        if len(set(answer_counts.values)) == 1:

            # Collect agg_scores per final_answer
            answer_agg_scores = group.groupby("final_answer")["agg_score"].apply(list).to_dict()

            valid_groups.append({
                "problem": problem_text,
                "unique_answers": list(answer_counts.index),
                "counts": answer_counts.to_dict(),
                "agg_scores": answer_agg_scores,
                "rows": group,  # optional: the full row subset
            })

    return pd.DataFrame(valid_groups)

tiebreak_case_studies = find_tiebreak_case_studies(df)
tiebreak_case_studies.to_csv('tiebreak_case_studies.csv')

def find_majority_not_most_plausible(df):
    """
    Finds problems where the most frequent final answer is NOT the one with the highest total verifier score (agg_score).
    """
    mismatched_problems = []

    for problem_text, group in df.groupby("problem"):

        # Count occurrences per final answer
        answer_counts = group.groupby("final_answer").size()

        # Skip problems with only 1 unique answer
        if len(answer_counts) <= 1:
            continue

        # Majority answer (most frequent)
        majority_answer = answer_counts.idxmax()

        # Collect list of agg_scores per answer
        answer_agg_scores = group.groupby("final_answer")["agg_score"].apply(list).to_dict()

        # Compute total agg_score per answer for comparison
        total_agg_score = {ans: sum(scores) for ans, scores in answer_agg_scores.items()}

        # Answer with highest total agg_score
        highest_score_answer = max(total_agg_score, key=total_agg_score.get)

        if majority_answer != highest_score_answer:
            mismatched_problems.append({
                "problem": problem_text,
                "majority_answer": majority_answer,
                "highest_agg_score_answer": highest_score_answer,
                "counts": answer_counts.to_dict(),
                "agg_scores": answer_agg_scores
            })

    return pd.DataFrame(mismatched_problems)

frequent_case_studies = find_majority_not_most_plausible(df)
frequent_case_studies.to_csv('frequent_case_studies.csv')
