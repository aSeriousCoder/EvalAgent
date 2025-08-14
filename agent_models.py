import os
import json
import re
import numpy as np

NUM_RETRIEVAL = 5
NUM_MAX_RETRY = 3

def build_prompt(news_list, short_term_memory_str=None, long_term_memory_str=None):
    news_list_str = "\n".join([f"Article {i+1}: {news['title']}" for i, news in enumerate(news_list)])
    prompt = f"You are a user browsing a news feed application, You are now come up with a list of news articles:\n{news_list_str}\n"
    if short_term_memory_str is not None:
        prompt += f"You have previously reading records:\n{short_term_memory_str}\n"
    if long_term_memory_str is not None:
        prompt += f"Your long-term preferences include:\n{long_term_memory_str}\n"
    prompt += "Rank the news articles from most to least interesting. "
    prompt += "Directly respond with the ranked list of news ids,  e.g., [4, 1, 3, 5, 2] (the 4th news attracts you the most, and the 2nd one is the least interesting)."
    prompt += "You should response directly without any other text."
    return prompt

def parse_response(response):
    pattern = r'\[[0-9, ]*\]'
    result = re.search(pattern, response)
    if result is None:
        raise ValueError("Invalid response: {response}")
    else:
        rank = json.loads(result.group(0))
        scores = [0] * len(rank)
        for i, news_id in enumerate(rank):
            scores[news_id-1] = (5 - i) / 5
        return scores

class EvalAgent:
    def __init__(self, model_name, llm_frame):
        self.model_name = model_name
        self.llm_frame = llm_frame
        
    def build_memory(self, history_news):
        self.memory = StableMemory(self.model_name, self.llm_frame)
        for i, news in enumerate(history_news):
            self.memory.update(news)

    def _get_short_term_memory_str(self, news_list):
        return self.memory._get_short_term_memory_str(news_list)

    def _get_long_term_memory_str(self, news_list):
        return self.memory._get_long_term_memory_str(news_list)

    def predict(self, news_list):
        for _ in range(NUM_MAX_RETRY):
            prompt = build_prompt(news_list, self._get_short_term_memory_str(news_list), self._get_long_term_memory_str(news_list))
            try:
                response = self.llm_frame.generate(prompt)
                return parse_response(response)
            except Exception as e:
                print(f"Error: {e}")
                continue
        raise Exception(f"Failed to generate prediction with {NUM_MAX_RETRY} retries")


class StableMemory:
    def __init__(self, model_name, llm_frame):
        self.short_term_memory = []  # [(news, timestamp, embedding, density, is_triggered)]
        self.long_term_memory = []   # [(description, embedding)]
        self.N = 20  # Maximum number of news in short-term memory
        self.k = 5   # k-NN parameter
        self.lambda_ = 0.1  # Time decay weight
        self.tau = 3600  # Time constant (seconds)
        self.beta = 1.0  # Sigmoid function parameter
        self.epsilon = 0.5  # Similarity threshold
        self.model_name = model_name
        self.llm_frame = llm_frame
        self.update_counter = 0  # Counter for news count
        self.update_threshold = 10  # Threshold for long-term memory update

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_all_densities(self, embeddings):
        """Compute the density of all embeddings"""
        if len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return [1.0]
            
        # Compute the covariance matrix
        embeddings_array = np.array(embeddings)
        try:
            S = np.cov(embeddings_array.T)
            # Check if the covariance matrix contains nan
            if np.isnan(S).any():
                # If it contains nan, use the identity matrix as the covariance matrix
                S = np.eye(embeddings_array.shape[1])
        except:
            # If the calculation of the covariance matrix fails, use the identity matrix
            S = np.eye(embeddings_array.shape[1])
        
        # Use Silverman's Rule to estimate the bandwidth
        n, d = embeddings_array.shape
        sigma = np.sqrt(n**(-1/(d+4)) * np.trace(S) / d)
        
        # Use broadcasting to compute the distance between all points
        embeddings_i = embeddings_array[:, np.newaxis, :]  # (n, 1, d)
        embeddings_j = embeddings_array[np.newaxis, :, :]  # (1, n, d)
        # Compute the Euclidean distance
        distances = np.sqrt(np.sum((embeddings_i - embeddings_j) ** 2, axis=2))  # (n, n)
        
        # Compute the density for each point
        densities = []
        for i in range(n):
            # Find the k nearest neighbors (excluding itself)
            k = min(self.k, n-1)
            distances_i = distances[i].copy()
            distances_i[i] = np.inf  # Exclude itself
            k_nearest_indices = np.argsort(distances_i)[:k]
            k_nearest = distances_i[k_nearest_indices]
            
            # Compute the Gaussian kernel density
            density = np.mean(np.exp(- self._sigmoid(k_nearest**2) / self._sigmoid(sigma)))
            densities.append(density)
            
        return densities

    def _update_all_densities(self):
        """Update the density of all memories"""
        if not self.short_term_memory:
            return
            
        # Get all embeddings
        embeddings = [item[2] for item in self.short_term_memory]
        
        # Compute all densities
        densities = self._compute_all_densities(embeddings)
        
        # Update the density value directly
        for i, density in enumerate(densities):
            news, timestamp, embedding, _, is_triggered = self.short_term_memory[i]
            self.short_term_memory[i] = (news, timestamp, embedding, density, is_triggered)

    def update(self, news):
        """Update the memory"""
        # Add to short-term memory, initial density set to 1
        self.short_term_memory.append((news, news.get('timestamp', 0), news['embedding'], 1.0, False))

        # Update the density of all memories
        self._update_all_densities()

        # Execute the forgetting mechanism
        self._forget(news.get('timestamp', 0))

        # Update the counter
        self.update_counter += 1
        
        # Update the long-term memory every 10 news
        if self.update_counter >= self.update_threshold:
            self._update_long_term_memory()
            self.update_counter = 0

    def _forget(self, current_time):
        """Execute the forgetting mechanism"""
        if len(self.short_term_memory) <= self.N:
            return
            
        # Compute the forgetting probability
        n = len(self.short_term_memory)
        forget_probs = []
        
        for news, timestamp, _, density, _ in self.short_term_memory:
            e = 1 - density  # Exploration tendency
            time_decay = self.lambda_ * (1 - np.exp(-(current_time - timestamp) / self.tau))
            p = ((n - self.N) * e / sum(1 - item[3] for item in self.short_term_memory)) + time_decay
            forget_probs.append(p)
            
        # Delete items based on probability
        keep_mask = np.random.random(len(forget_probs)) > forget_probs
        self.short_term_memory = [item for i, item in enumerate(self.short_term_memory) if keep_mask[i]]

    def _update_long_term_memory(self):
        """Update the long-term memory"""
        if not self.short_term_memory:
            return
            
        # Compute the average density
        avg_density = np.mean([item[3] for item in self.short_term_memory])
        
        # Collect all triggered memories
        triggered_news = []
        for i, (news, _, embedding, density, is_triggered) in enumerate(self.short_term_memory):
            if is_triggered:
                continue
                
            p_trigger = 1 / (1 + np.exp(-self.beta * (density - avg_density)))
            
            if np.random.random() < p_trigger:
                # Mark as triggered
                self.short_term_memory[i] = (news, news.get('timestamp', 0), embedding, density, True)
                triggered_news.append((news, embedding))
        
        if not triggered_news:
            return

        # Batch recall related descriptions
        recalled_descriptions = []
        for news, embedding in triggered_news:
            for desc in self.long_term_memory:
                desc_emb = desc['embedding']
                similarity = np.dot(embedding, desc_emb) / (np.linalg.norm(embedding) * np.linalg.norm(desc_emb))
                if similarity > self.epsilon:
                    recalled_descriptions.append(desc)
                    self.long_term_memory.remove(desc)
        
        # Update all memories using LLM
        if recalled_descriptions or len(self.long_term_memory) == 0:
            recalled_descriptions_str = "\n".join([f"Preference {i+1}: {desc['description']}" for i, desc in enumerate(recalled_descriptions)])
            triggered_news_str = "\n".join([f"News {i+1}: {news['title']}" for i, (news, _) in enumerate(triggered_news)])
            
            prompt = f"""
You are an intelligent user modeling assistant. Your goal is to incrementally update the user's high-level information based on the news reading records. 
The high-level information is organized as a list of concise descriptions reflecting the user's interests, preferences, and behavioral patterns, detailed as:
{recalled_descriptions_str}

The user just read the following news articles:
{triggered_news_str}

Rules:
1. Descriptions should be concise and clear, avoiding unnecessary complexity.
2. You can update the descriptions by modify, add or merge.
3. Directly respond with the updated descriptions with each description in a new line."""
            try:
                response = self.llm_frame.generate(prompt)
                updated_descriptions = response.split("\n")
                updated_description_embeddings = self.llm_frame.embed(updated_descriptions)
                for desc, embedding in zip(updated_descriptions, updated_description_embeddings):
                    self.long_term_memory.append({
                        'description': desc,
                        'embedding': embedding
                    })
            except Exception as e:
                self.long_term_memory.extend(recalled_descriptions)  # roll back
                print(f"Error updating long-term memory: {e}")

    def retrieve_from_short(self, news):
        """Retrieve related memories"""
        query_embedding = news['embedding']
        
        # Retrieve from short-term memory
        short_term_results = []
        for item in self.short_term_memory:
            news, _, embedding, _, _ = item
            distance = np.linalg.norm(query_embedding - embedding)
            short_term_results.append((news, distance))
        
        # Sort by distance and select the top K
        short_term_results.sort(key=lambda x: x[1])
        K = min(5, len(short_term_results))
        retrieved_short_term = [item[0] for item in short_term_results[:K]]
        
        return retrieved_short_term

    def retrieve_from_long(self, news):
        """Retrieve related memories"""
        query_embedding = news['embedding']
        # Retrieve from long-term memory
        long_term_results = []
        for desc in self.long_term_memory:
            similarity = np.dot(query_embedding, desc['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(desc['embedding']))
            long_term_results.append((desc, similarity))
        
        # Sort by similarity and select the top T
        long_term_results.sort(key=lambda x: x[1], reverse=True)
        T = min(3, len(long_term_results))
        retrieved_long_term = [item[0] for item in long_term_results[:T]]
        
        return retrieved_long_term

    def _get_short_term_memory_str(self, news_list):
        """Get the string representation of short-term memory"""
        all_retrieved_news = []
        for news in news_list:
            retrieved_news = self.retrieve_from_short(news)
            all_retrieved_news.extend(retrieved_news)
        
        # Remove duplicates while maintaining order
        seen = set()
        unique_news = []
        for news in all_retrieved_news:
            if news['title'] not in seen:
                seen.add(news['title'])
                unique_news.append(news)
        
        # Limit the number
        K = min(5, len(unique_news))
        return "\n".join([f"History {i+1}: {news['title']}" for i, news in enumerate(unique_news[:K])])

    def _get_long_term_memory_str(self, news_list):
        """Get the string representation of long-term memory"""
        all_retrieved_descriptions = []
        for news in news_list:
            retrieved_descriptions = self.retrieve_from_long(news)
            all_retrieved_descriptions.extend(retrieved_descriptions)
        # Remove duplicates while maintaining order
        seen = set()
        unique_descriptions = []
        for desc in all_retrieved_descriptions:
            if desc['description'] not in seen:
                seen.add(desc['description'])
                unique_descriptions.append(desc['description'])
        # Limit the number
        T = min(3, len(unique_descriptions))
        return "\n".join([f"Preference {i+1}: {desc}" for i, desc in enumerate(unique_descriptions[:T])])
