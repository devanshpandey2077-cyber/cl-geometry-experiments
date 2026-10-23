from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    repo: str
    config: str | None
    fields: tuple[str, ...]
    labels: tuple[str, ...]
    instruction: str

TASKS = {
    "yelp": Task("fancyzhx/yelp_polarity", "plain_text", ("text",),
                 ("negative", "positive"), "Classify the sentiment of the review."),
    "qqp": Task("nyu-mll/glue", "qqp", ("question1", "question2"),
                ("different meanings", "same meaning"),
                "Decide whether the two questions have the same meaning."),
    "qnli": Task("nyu-mll/glue", "qnli", ("question", "sentence"),
                 ("answers the question", "does not answer the question"),
                 "Decide whether the sentence answers the question."),
    "mnli": Task("nyu-mll/glue", "mnli", ("premise", "hypothesis"),
                 ("entailment", "neutral", "contradiction"),
                 "Determine the relation of the hypothesis to the premise."),
    "ag_news": Task("fancyzhx/ag_news", None, ("text",),
                    ("world", "sports", "business", "science and technology"),
                    "Classify the news topic."),
    "paws": Task("google-research-datasets/paws", "labeled_final", ("sentence1", "sentence2"),
                 ("different meanings", "same meaning"),
                 "Decide whether the sentences have the same meaning."),
}
CODES = "ABCD"

