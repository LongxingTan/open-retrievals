import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import ray
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.config import HTTPOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME_OR_PATH_RERANKER = os.getenv("RERANKER_MODEL_PATH", "models/Qwen3-Reranker-0.6B")
MODEL_NAME_OR_PATH_EMBEDDING = os.getenv("EMBEDDING_MODEL_PATH", "models/Qwen3-Embedding-0.6B")
NUM_REPLICAS = int(os.getenv("NUM_REPLICAS", "2"))
NUM_GPUS = float(os.getenv("NUM_GPUS", "0.9"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4008"))


# Pydantic models for request validation
class RerankerInput(BaseModel):
    questions: List[str] = Field(..., min_items=1, description="List of questions")
    texts: List[str] = Field(..., min_items=1, description="List of texts to rerank")
    instruction: Optional[str] = Field(
        default="Given the user query, retrieval the relevant passages", description="Instruction for reranking"
    )


class EmbeddingInput(BaseModel):
    input: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to embed")
    is_query: bool = Field(default=False, description="Whether input is a query")


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: float


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]


class RerankerResponse(BaseModel):
    scores: List[float]
    model: str
    usage: Dict[str, int]


# Global app instance for proper FastAPI integration
app = FastAPI(
    title="Embedding and Reranking Service",
    description="Production-ready embedding and reranking API using Ray Serve",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_gpus": NUM_GPUS},
    max_concurrent_queries=100,  # Limit concurrent requests per replica
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": NUM_REPLICAS * 2,
        "target_num_ongoing_requests_per_replica": 10,
    },
)
@serve.ingress(app)
class BatchCombineInferModel:
    def __init__(self, model_name_or_path_reranker: str, model_name_or_path_embedding: str):
        """Initialize the model with proper error handling and logging."""
        try:
            logger.info("Initializing embedding model...")
            self.model_embedding = Qwen3Embedding(
                model_name_or_path=model_name_or_path_embedding,
            )
            logger.info("Embedding model initialized successfully")

            logger.info("Initializing reranker model...")
            self.model_reranker = Qwen3Reranker(
                model_name_or_path=model_name_or_path_reranker,
                instruction="Retrieval document that can answer user's query",
                max_length=2048,
            )
            logger.info("Reranker model initialized successfully")

            # Model metadata
            self.embedding_model_name = model_name_or_path_embedding
            self.reranker_model_name = model_name_or_path_reranker

        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    @app.get("/health")
    async def health_check(self) -> HealthResponse:
        """Health check endpoint for load balancers and monitoring."""
        return HealthResponse(status="healthy", message="Service is running", timestamp=time.time())

    @app.get("/")
    async def root(self):
        """Root endpoint with service information."""
        return {
            "service": "Embedding and Reranking API",
            "version": "1.0.0",
            "endpoints": {"embedding": "/embedding/api", "reranker": "/reranker/api", "health": "/health"},
        }

    @app.post("/embedding/api", response_model=EmbeddingResponse)
    async def embedding(self, request: EmbeddingInput) -> EmbeddingResponse:
        """Generate embeddings for input texts."""
        try:
            start_time = time.time()

            # Validate input length
            if len(request.input) > 1000:
                raise HTTPException(status_code=400, detail="Too many texts. Maximum 1000 texts per request.")

            # Generate embeddings
            with torch.inference_mode():
                embeddings = self.model_embedding.encode(request.input, is_query=request.is_query)
                # Convert to list format
                embeddings_list = embeddings.cpu().detach().numpy().tolist()

            processing_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {processing_time:.2f}s for {len(request.input)} texts")

            return EmbeddingResponse(
                embeddings=embeddings_list,
                model=self.embedding_model_name,
                usage={"total_texts": len(request.input), "processing_time_ms": int(processing_time * 1000)},
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            raise HTTPException(status_code=503, detail="GPU memory exhausted. Please try with fewer texts.")
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @app.post("/reranker/api", response_model=RerankerResponse)
    async def reranker(self, request: RerankerInput) -> RerankerResponse:
        """Rerank texts based on questions."""
        try:
            start_time = time.time()

            # Validate input lengths match
            if len(request.questions) != len(request.texts):
                raise HTTPException(status_code=400, detail="Number of questions must match number of texts")

            # Validate reasonable batch size
            if len(request.questions) > 100:
                raise HTTPException(
                    status_code=400, detail="Too many pairs. Maximum 100 question-text pairs per request."
                )

            # Create pairs and compute scores
            with torch.inference_mode():
                pairs = list(zip(request.questions, request.texts))
                scores = self.model_reranker.compute_scores(pairs, request.instruction)

                # Ensure scores is a list
                if not isinstance(scores, list):
                    scores = scores.tolist() if hasattr(scores, 'tolist') else [float(scores)]

            processing_time = time.time() - start_time
            logger.info(f"Reranking completed in {processing_time:.2f}s for {len(pairs)} pairs")

            return RerankerResponse(
                scores=scores,
                model=self.reranker_model_name,
                usage={"total_pairs": len(pairs), "processing_time_ms": int(processing_time * 1000)},
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            raise HTTPException(status_code=503, detail="GPU memory exhausted. Please try with fewer pairs.")
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def main():
    """Main function to start the Ray Serve application."""
    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Start Ray Serve
        serve.start(
            http_options=HTTPOptions(
                host=HOST,
                port=PORT,
                request_timeout_s=300,  # 5 minutes timeout
            )
        )

        logger.info(f"Starting Ray Serve on {HOST}:{PORT}")

        # Deploy the application
        serve.run(
            BatchCombineInferModel.bind(MODEL_NAME_OR_PATH_RERANKER, MODEL_NAME_OR_PATH_EMBEDDING),
            route_prefix="/",
            name="embedding_reranker_service",
        )

        logger.info("Service deployed successfully")

        # Keep the service running
        while True:
            time.sleep(60)  # Check every minute instead of every 1000 seconds

    except KeyboardInterrupt:
        logger.info("Shutting down service...")
        serve.shutdown()
        ray.shutdown()
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        raise


if __name__ == "__main__":
    main()
