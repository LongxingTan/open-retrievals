ENCODE_QUERY_DIR=./scifact/query-embeddings
ENCODE_CORPUS_DIR=./scifact/corpus-embeddings
SAVE_RANKING_PATH=./scifact/recall_out/retrieval.txt
TOP_K=10

python -m retrievals.pipelines.retrieve \
    --query_reps=$ENCODE_QUERY_DIR/query.pkl \
    --passage_reps $ENCODE_CORPUS_DIR/'*.pkl' \
    --batch_size 128 \
    --top_k $TOP_K \
    --save_ranking_file $SAVE_RANKING_PATH
