# Wikipedia Natural Questions & TriviaQA

## 1. prepare data
```shell
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz
```

## 2. Convert train data format
```shell
python prepare_retrieve_data.py --input ./biencoder-nq-train.json --output ./nq-train-data
```

## 3. Embedding fine-tuning
```shell
sh embed_pairwise_train.sh
```

## 4. Retrieval

Download data
```shell
# queries
wget https://www.dropbox.com/s/x4abrhszjssq6gl/nq-test-queries.json
wget https://www.dropbox.com/s/b64e07jzlji8zhl/trivia-test-queries.json

# corpus
wget https://www.dropbox.com/s/8ocbt0qpykszgeu/wikipedia-corpus.tar.gz
tar -xvf wikipedia-corpus.tar.gz
```

Build corpus index
```shell

```

Build query index
```shell

```

Search

```python

```

## 5. Evaluation
```python

```


## Reference
- [Dense](https://github.com/luyug/Dense)
