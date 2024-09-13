

## LLM reranking
- `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
