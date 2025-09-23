import time

import numpy as np
import redis
import json
import hashlib
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from src.utils import get_root_path


class SemanticCache:
    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.85):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        # self.embedding_model = SentenceTransformer(str(get_root_path() / "models" / "all-MiniLM-L6-v2"))
        self.embedding_model = SentenceTransformer(str(get_root_path() / "models" / "all-mpnet-base-v2"))
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text"""
        print
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
    test_cases = [
        "A skeleton warrior with rusty sword and shield, standing, front view, idle pose",
        "A skeleton warrior wielding a jagged great axe, crouched, three-quarters view, attack stance",
        "A skeleton archer with cracked bow, aiming, side view",
        "A skeleton mage holding a glowing staff, casting spell, top-down angle",
        "A skeleton knight in tattered armor, holding longsword, standing, back view",
        "A skeleton dual wielding curved daggers, crouched, front view, ready stance",
        "A skeletal paladin with glowing shield, kneeling, front view",
        "A skeleton warrior with spiked halberd, leaping forward, dynamic angle",
        "A skeleton rogue with short sword, sneaking, three-quarters view",
        "A skeleton warrior with battle hammer, raised overhead, side view, attack pose",
        "A skeleton with rusty pike, marching, front view",
        "A skeleton swordsman holding longsword, standing defensive, back view",
        "A skeleton warrior with chain whip, swinging, side view",
        "A skeleton warrior dual wielding axes, roaring, front view",
        "A skeleton with flaming sword, crouched, top-down view, casting fire",
        "A skeleton with frost sword, standing, icy aura, three-quarters view",
        "A skeleton with cursed blade, glowing red eyes, front view, battle stance",
        "A skeleton warrior with spiked mace, jumping, side angle",
        "A skeleton carrying a long curved blade, running, dynamic angle",
        "A skeleton with double-headed axe, raising weapon, front view",
        "A skeleton mounted on skeletal horse, holding lance, charging, side view",
        "A skeleton warrior with cracked shield and sword, crouched, back view",
        "A skeleton archer drawing bow, aiming, front view",
        "A skeleton wielding jagged dagger, sneaking, three-quarters view",
        "A skeleton holding massive two-handed sword, standing, side view",
        "A skeleton with morning star, swinging, front view",
        "A skeleton warrior with hook blade, crouched, side angle",
        "A skeleton with halberd, standing tall, top-down view",
        "A skeleton warrior dual wielding short swords, leaping, three-quarters angle",
        "A skeleton with scythe, grim reaper style, front view, floating",
        "A skeleton with battle banner, standing, side angle",
        "A skeleton with broken sword, idle stance, back view",
        "A skeleton wielding spear and shield, defensive, front view",
        "A skeleton with dual katanas, crouched, ready stance, dynamic angle",
        "A skeleton holding cleaver, hunched, front view, menacing stance",
        "A skeleton warrior with warhammer, standing firm, three-quarters view",
        "A skeleton dual wielding curved daggers, sneaky, side angle",
        "A skeleton with glowing sword, combat ready, top-down view",
        "A skeleton with chain armor, holding sword, standing, front view",
        "A skeleton with spiked mace, leaping, three-quarters angle",
        "A skeleton holding longsword and cracked shield, marching forward, side view",
        "A skeleton in ancient armor, standing, sword upright, front view",
        "A skeleton holding jagged spear, ready stance, dynamic angle",
        "A skeleton mage raising skeletal minions, front view, casting spell",
        "A skeleton warrior swinging flail, crouched, side view",
        "A skeleton with spiked axe, roaring, three-quarters view",
        "A skeleton holding crossbow, aiming, side angle",
        "A skeleton archer with glowing arrows, crouched, top-down view",
        "A skeleton warrior with cracked armor, holding battle axe, front view",
        "A skeleton knight with flaming sword, kneeling, side view",
        "A skeleton rogue dual wielding daggers, mid-leap, dynamic angle",
        "A skeleton with rusty club, leaning forward, three-quarters view",
        "A skeleton warrior with jagged sword, charging, front view",
        "A skeleton mage with glowing staff, floating, top-down angle",
        "A skeleton with spiked shield, crouched, side view, defending",
        "A skeleton warrior mounted on undead steed, raising sword, three-quarters view",
        "A skeleton with enchanted blade, standing, glowing aura, front view",
        "A skeleton archer shooting multiple arrows, leaping, dynamic angle",
        "A skeleton warrior holding broken halberd, crouched, back view",
        "A skeleton knight wielding longsword, standing, side angle",
        "A skeleton warrior swinging dual axes, roaring, front view",
        "A skeleton holding curved dagger and shield, crouched, three-quarters view",
        "A skeleton with spiked gauntlets, punching, side angle",
        "A skeleton warrior with flaming halberd, charging, front view",
        "A skeleton mage casting dark magic, floating, top-down view",
        "A skeleton holding war banner, standing proudly, side view",
        "A skeleton with jagged sword, mid-air leap, dynamic angle",
        "A skeleton dual wielding scimitars, crouched, three-quarters view",
        "A skeleton mounted on skeletal wolf, holding spear, charging, front view",
        "A skeleton with glowing axe, standing tall, back view",
        "A skeleton rogue with short swords, sneaking, side angle",
        "A skeleton warrior wielding massive hammer, mid-swing, dynamic angle",
        "A skeleton with frost sword, crouched, ready to strike, front view",
        "A skeleton mage holding dark tome, casting spell, side angle",
        "A skeleton with broken shield and jagged sword, marching, three-quarters view",
        "A skeleton archer drawing glowing bow, crouched, top-down view",
        "A skeleton holding giant cleaver, hunched, dynamic angle",
        "A skeleton warrior swinging flail, standing, side view",
        "A skeleton dual wielding axes, roaring, front view, battle stance",
        "A skeleton with flaming sword, leaping, three-quarters view",
        "A skeleton mage summoning skeletal minions, floating, top-down angle",
        "A skeleton holding spiked spear and shield, crouched, side view",
        "A skeleton warrior mounted on undead steed, charging, dynamic angle",
        "A skeleton rogue dual wielding daggers, sneaking, front view",
        "A skeleton warrior wielding jagged halberd, standing, three-quarters view",
        "A skeleton mage with glowing staff, casting fire spell, side angle",
        "A skeleton with curved sword and shield, crouched, back view",
        "A skeleton dual wielding short swords, mid-leap, dynamic angle",
        "A skeleton warrior holding massive two-handed sword, roaring, front view",
        "A skeleton with spiked gauntlets, attacking, three-quarters view",
        "A skeleton knight wielding glowing sword, kneeling, side view",
        "A skeleton archer shooting multiple arrows, crouched, top-down angle"
    ]

    embeddings = []
    start = time.time()
    for i in range(0, len(test_cases)):
        embeddings.append(cache._generate_embedding(test_cases[i]))
    now = time.time()
    print(f"Took {now - start} seconds to genearate embeddings")
    start = time.time()
    for i in range(0, len(embeddings)):
        high_smilarity_prompts = []
        for j in range(i, len(embeddings)):
            e1 = embeddings[i]
            e2 = embeddings[j]
            similarity = cosine_similarity(
                e1.reshape(1, -1),
                e2.reshape(1, -1)
            )[0][0]
            if similarity > 0.80:
                high_smilarity_prompts.append(test_cases[j])
        print(f"{test_cases[i]}'s high similarity is: {"\n".join(high_smilarity_prompts)}")
        print("========================")
    now = time.time()
    print(f"Took {now - start} seconds to compare embeddings")

