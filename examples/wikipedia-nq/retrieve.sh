ENCODE_QUERY_DIR=embeddings-nq-queries
ENCODE_DIR=embeddings-nq-corpus
SAVE_RANKING_PATH=retrieval.txt
TOP_K=100

python -m retrive \
    --query_reps=$ENCODE_QUERY_DIR/query.pt \
    --passage_reps $ENCODE_DIR/'*.index' \
    --batch_size -1 \
    --top_k $TOP_K \
    --save_text \
    --save_ranking_to $SAVE_RANKING_PATH
