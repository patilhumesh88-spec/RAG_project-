import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load API key securely
load_dotenv("API.env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env file")

# 2. Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")  # Fast & free-tier friendly

def run_pipeline_1(query: str) -> dict:
    """Pipeline 1: LLM-Only (No retrieval, raw prompt → answer)"""
    start_time = time.time()

    # Call Gemini
    response = model.generate_content(query)
    answer = response.text.strip()

    # Calculate latency
    latency = time.time() - start_time

    # Extract token counts from Gemini's built-in metadata
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage else 0
    completion_tokens = usage.candidates_token_count if usage else 0
    total_tokens = prompt_tokens + completion_tokens

    # Cost calculation (Free tier = $0, but here's the formula for later)
    # Pricing: ~$0.075/1M input, ~$0.30/1M output
    cost_usd = 0.0  # Free tier

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_sec": round(latency, 3),
        "cost_usd": cost_usd
    }

if __name__ == "__main__":
    # Test query matching hackathon theme
    test_query = "What is GraphRAG and how does it reduce token usage compared to standard vector RAG?"
    
    result = run_pipeline_1(test_query)
    
    print("="*50)
    print("🤖 PIPELINE 1: LLM-ONLY BASELINE")
    print("="*50)
    print(f"✅ Answer:\n{result['answer']}\n")
    print("📊 Metrics:")
    print(f"  Prompt Tokens:      {result['prompt_tokens']}")
    print(f"  Completion Tokens:  {result['completion_tokens']}")
    print(f"  Total Tokens:       {result['total_tokens']}")
    print(f"  ⏱️ Latency:          {result['latency_sec']}s")
    print(f"  💰 Cost:            ${result['cost_usd']} (Free Tier)")
    print("="*50)
