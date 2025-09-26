import time

import numpy as np
import redis
import json
import hashlib
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from src.experiment import ImageGenerationExperiments
from src.utils import get_root_path


class SemanticCache:
    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 similarity_threshold: float = 0.85):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        # self.embedding_model = SentenceTransformer(embedding_model_path)
        # Use online model to avoid torch.load security issue - caches to ~/.cache automatically
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text"""
        return self.embedding_model.encode(text)

    def _get_cache_keys(self, user_id: str) -> List[str]:
        """Get all cache keys for a user"""
        pattern = f"cache:{user_id}:*"
        return self.redis_client.keys(pattern)

    def get_cached_response(self, user_id: str, query: str) -> Optional[str]:
        """
        Check if we have a semantically similar cached response
        """
        try:
            # Generate embedding for the new query
            query_embedding = self._generate_embedding(query)

            # Get all cached entries for this user
            cache_keys = self._get_cache_keys(user_id)

            if not cache_keys:
                return None

            best_similarity = 0.0
            best_response = None

            # Compare with each cached query
            for cache_key in cache_keys:
                cached_data = self.redis_client.hgetall(cache_key)

                if not cached_data:
                    continue

                # Deserialize cached embedding
                cached_embedding = np.frombuffer(
                    bytes.fromhex(cached_data['embedding']),
                    dtype=np.float32
                )

                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]

                # Track best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = cached_data['response']

            # Return cached response if similarity exceeds threshold
            if best_similarity >= self.similarity_threshold:
                self.logger.info(f"Cache HIT for user {user_id} with similarity {best_similarity:.3f}")
                return best_response

            self.logger.info(f"Cache MISS for user {user_id}, best similarity: {best_similarity:.3f}")
            return None

        except Exception as e:
            self.logger.error(f"Error in cache lookup: {e}")
            return None

    def cache_response(self, user_id: str, query: str, response: str, ttl: int = 86400):
        """
        Store query-response pair in semantic cache
        """
        try:
            # Generate embedding
            embedding = self._generate_embedding(query)

            # Create unique cache key
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            cache_key = f"cache:{user_id}:{query_hash}"

            # Store in Redis hash
            cache_data = {
                'query': query,
                'response': response,
                'embedding': embedding.tobytes().hex(),
                'timestamp': str(int(time.time()))
            }

            self.redis_client.hset(cache_key, mapping=cache_data)
            self.redis_client.expire(cache_key, ttl)  # Set TTL

            self.logger.info(f"Cached response for user {user_id}")

        except Exception as e:
            self.logger.error(f"Error caching response: {e}")


if __name__ == "__main__":
    cache = SemanticCache()
    experiment = ImageGenerationExperiments(test_id=1)
    test_cases = [
        experiment.pos_prompt,
        experiment.pos_prompt_2,
        experiment.pos_prompt_3,
    ]

    embeddings = []
    start = time.time()
    for i in range(0, len(test_cases)):
        embeddings.append(cache._generate_embedding(test_cases[i]))
    start = time.time()
    for i in range(0, len(embeddings)):
        high_smilarity_prompts = []
        for j in range(i + 1, len(embeddings)):
            e1 = embeddings[i]
            e2 = embeddings[j]
            similarity = cosine_similarity(
                e1.reshape(1, -1),
                e2.reshape(1, -1)
            )[0][0]
            if similarity > 0.80:
                high_smilarity_prompts.append(test_cases[j])
            print(f"similarity between prompt {i} and prompt {j}: {similarity}")
    now = time.time()
    print(f"Took {now - start} seconds to compare embeddings")

