# Open-Retrievals Enhancement Plan

## Vision

Transform the open-retrievals repository into a unified training framework that combines RAG-retrieval capabilities with verl-style configuration management and training workflows, while supporting both BERT and decoder LLM models.

## Key Requirements

1. **YAML-Based Configuration**: Enable verl-style training via YAML files
2. **Dual Training Paradigms**: Support both verl training workflows and LLaMA-Factory style training
3. **Model Universal Support**: Extend support to both BERT (encoder) and decoder-only LLMs, VLM, RL-embedding
4. **Full Feature Coverage**: Preserve all existing RAG-retrieval features (retrieval, ranking, ColBERT, distillation)


## Quick start
```
make style
```

```
make test
```


## Architecture Design

### 1. Configuration Layer (`src/retrievals/config/`)

#### YAML Configuration Components:
- `BaseConfig`: Common configuration fields
- `ModelConfig`: Model selection and parameters for BERT/LLM
- `DataConfig`: Dataset paths, formats, preprocessing
- `TrainingConfig`: Hyperparameters, optimization settings
- `LossConfig`: Loss function configurations (contrastive, ranking, distillation)
- `ColBERTConfig`: ColBERT-specific parameters
- `DistillationConfig`: Teacher-student training settings

#### Configuration Structure:
```yaml
# Model Configuration
model:
  type: "bert"  # [bert, llm, mixed]
  model_name: "bert-base-uncased"
  tokenizer_name: "bert-base-uncased"
  pooling_strategy: "cls"
  max_length: 512

# Training Configuration
training:
  framework: "verl"  # [verl, llama_factory]
  output_dir: "./outputs"
  num_epochs: 3
  learning_rate: 2e-5
  batch_size: 32
  gpu_num: 1
  mixed_precision: "bf16"
  gradient_accumulation_steps: 1

# Loss Configuration
loss:
  type: "multiple_negatives_ranking"  # [mnrl, triplet, colbert, margin_mse, distillation]
  temperature: 0.05
  margin: 0.5

# Data Configuration
data:
  train_file: "train.jsonl"
  valid_file: "valid.jsonl"
  query_col: "query"
  doc_col: "document"
  label_col: "label"
  negative_samples: 8

# Retrieval Configuration
retrieval:
  mode: "dense"  # [dense, sparse, hybrid]
  index_path: "./index"
  top_k: 100

# ColBERT Configuration
colbert:
  dim: 128
  doc_maxlen: 512
  mask_punctuation: true

# Distillation Configuration
distillation:
  teacher_model: "bert-base-uncased"
  teacher_checkpoint: "./teacher_ckpt"
  temperature: 2.0
  alpha: 0.7
  loss_types: ["logits", "hidden_states"]

# Ranking Configuration
ranking:
  cross_encoder: false
  rerank_top_k: 10
  objective: "classification"  # [classification, regression, listwise]
```

### 2. Unified Training Engine (`src/retrievals/trainer/unified_trainer.py`)

The refactored trainer should support:

#### Multiple Frameworks:
- **verl-style**: FSDP, Ray, custom training loops
- **LLaMA-Factory**: LoRA/QLoRA, efficient fine-tuning
- **transformers**: Standard HuggingFace trainer

#### Training Modes:
- Dense Retrieval Training (Bi-Encoder, DPR, SimCSE)
- ColBERT Training (Late interaction, token-level encoding)
- Cross-Encoder Training (Pointwise, Pairwise, Listwise)
- Knowledge Distillation (Logits, Hidden States, Attention)
- Multi-task Learning

#### Key Methods:
```python
class UnifiedTrainer:
    """Unified trainer supporting multiple frameworks and model types"""
    
    def __init__(self, config: TrainingConfig, model, train_dataset, val_dataset):
        self.config = config
        self.framework = config.framework  # "verl" or "llama_factory"
        self.setup_framework()
    
    def setup_framework(self):
        """Initialize training framework"""
        if self.framework == "verl":
            self.setup_verl()
        elif self.framework == "llama_factory":
            self.setup_llama_factory()
    
    def train_dense_retrieval(self):
        """Dense passage retrieval training"""
        pass
    
    def train_colbert(self):
        """ColBERT late interaction training"""
        pass
    
    def train_cross_encoder(self):
        """Cross-encoder ranking training"""
        pass
    
    def train_distillation(self, teacher_model, student_model):
        """Knowledge distillation training"""
        pass
```

### 3. Model Registry (`src/retrievals/models/registry.py`)

Centralized model loading supporting:
- BERT family (bert, roberta, distilbert, etc.)
- Encoder-Decoder models (T5, BART)
- Decoder-only LLMs (Llama, Mistral, Qwen, etc.)
- Custom models with specific pooling strategies

### 4. Loss Function Factory (`src/retrievals/losses/factory.py`)

Support all existing loss functions:
- **Contrastive**: InfoNCE, MultipleNegativesRanking, Triplet Loss
- **Ranking**: MarginMSE, PairKL, CosineSimilarityLoss
- **ColBERT**: ColBERT-specific token-level loss
- **Distillation**: LogitsDistillation, HiddenStateDistillation
- **Specialized**: ArcFace, Circle Loss, CoSent, SimCSE

### 5. Pipeline Management (`src/retrievals/pipelines/`)

Enhanced pipelines supporting different tasks:
- `DenseRetrievalPipeline`: Vector-based retrieval
- `ColBERTRetrievalPipeline`: Late interaction retrieval
- `CrossEncoderRerankPipeline`: Re-ranking
- `DistillationPipeline`: Teacher-student training
- `RAGPipeline`: End-to-end RAG

## Implementation Roadmap

### Phase 1: Foundation (Priority: HIGH)

1. **Create Configuration System**
   - [ ] Design YAML config schemas and validation
   - [ ] Implement `ConfigParser` and `ConfigValidator`
   - [ ] Add Hydra-style configuration management
   - [ ] Create default configs for different training scenarios

2. **Model Registry & Loader**
   - [ ] Centralize model loading logic
   - [ ] Support automatic model type detection (bert vs llm)
   - [ ] Implement pooling strategy abstractions
   - [ ] Add model quantization support (for LLMs)

3. **Unified Dataset Interface**
   - [ ] Create flexible dataset loading (JSON, CSV, Parquet, HF datasets)
   - [ ] Support query-passage and query-doc formats
   - [ ] Implement in-batch negative sampling
   - [ ] Add data augmentation pipelines

### Phase 2: Training Framework Integration (Priority: HIGH)

1. **verl-style Training**
   - [ ] Implement FSDP support for large models
   - [ ] Add Ray-based distributed training
   - [ ] Integrate verl's training methodology
   - [ ] Support overlap communication and computation

2. **LLaMA-Factory Integration**
   - [ ] Add LoRA/QLoRA adapter support
   - [ ] Implement efficient fine-tuning strategies
   - [ ] Support quantization (bitsandbytes)
   - [ ] Integrate Po+ scheduling and RLHF

3. **Training Engine Refactoring**
   - [ ] Extract common training logic from existing trainers
   - [ ] Implement multi-mode training (dense, colbert, ranking, distill)
   - [ ] Add checkpoint management and resuming
   - [ ] Support mixed precision training

### Phase 3: Feature Enhancement (Priority: MEDIUM)

1. **ColBERT Enhancements**
   - [ ] Optimize ANN indexing and search
   - [ ] Add centroid-based compression
   - [ ] Support multi-vector encoding
   - [ ] Integrate with retrieval pipelines

2. **Cross-Encoder & Ranking**
   - [ ] Support multiple ranking objectives
   - [ ] Implement ListMLE, ListNet for ranking
   - [ ] Add teacher forcing during training
   - [ ] Support ICD-specific training

3. **Distillation Improvements**
   - [ ] Progressive layer distillation
   - [ ] Attention-based distillation
   - [ ] Multi-teacher distillation
   - [ ] Distillation with different architectures

### Phase 4: Comprehensive Testing (Priority: HIGH)

1. **Unit & Integration Tests**
   - [ ] Test all loss functions with different models
   - [ ] Validate retrieval metrics (MRR, NDCG, Recall)
   - [ ] Test YAML config parsing and validation
   - [ ] Verify checkpoint saving and loading

2. **End-to-End Tests**
   - [ ] Run dense retrieval training from YAML
   - [ ] Execute ColBERT training and retrieval
   - [ ] Test cross-encoder ranking
   - [ ] Validate distillation pipeline

3. **Performance Benchmarks**
   - [ ] Compare training speed with vLLM/DeepSpeed
   - [ ] Measure retrieval quality on benchmarks (MS Marco, BEIR)
   - [ ] Evaluate memory footprint
   - [ ] Test scaling across multiple GPUs

### Phase 5: Documentation & Examples (Priority: MEDIUM)

1. **Comprehensive Documentation**
   - [ ] Update README with new features
   - [ ] Create API documentation
   - [ ] Write migration guide from old API
   - [ ] Document configuration options

2. **Example YAML Files**
   - [ ] Dense retrieval on MS Marco
   - [ ] ColBERT training setup
   - [ ] Cross-encoder ranking
   - [ ] Knowledge distillation
   - [ ] RAG end-to-end pipeline

3. **Tutorials**
   - [ ] Custom data format specification
   - [ ] Multi-GPU training setup
   - [ ] Memory-efficient training
   - [ ] Fine-tuning with LoRA
   - [ ] Large-scale indexing

## Key Considerations

### YAML-First API
- YAML configuration is the supported public training interface
- Backward compatibility with older Python model, loss, and trainer APIs is not required
- Remove compatibility shims when simplifying the implementation
- Update examples to use YAML workflows

### Performance Optimizations
- Enable gradient checkpointing
- Support sequence packing/flash attention
- Implement efficient negative mining
- Use FSDP/DeepSpeed for large models

### Extensibility
- Plugin system for custom loss functions
- Interface for new retrieval methods
- Support for custom data loaders
- Hook system for training callbacks

## Success Metrics

1. **Functionality**: All existing models/losses work with new system
2. **Flexibility**: Users can configure any training scenario via YAML
3. **Performance**: Training speed matches or exceeds current implementation
4. **Scalability**: Supports distributed training on 8+ GPUs
5. **Quality**: Retrieval benchmarks maintain or improve scores
6. **User Adoption**: Clear documentation and working examples