try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Notice: python-dotenv not installed. API keys must be set in the environment.")

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
import shutil

from pipeline import (
    secure_rag_pipeline, unfiltered_rag_pipeline, LATENCY_DATA, add_dynamic_domain,
    TEST_SUITE, check_for_leakage, check_for_refusal, check_for_harmful_refusal,
    load_domain_assets, load_raw_domain
)

app = FastAPI(title="SecureRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    model: str
    domain: str = "healthcare"
    filtered: bool = True

class BenchmarkRequest(BaseModel):
    models: list[str]
    domain: str = "healthcare"
    iterations: int = 2
    filtered: bool = True

BENCHMARK_JOBS = {}

@app.get("/api/health")
async def health_check():
    return {"ok": True, "pipeline_connected": True}

import os

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    domain_name = file.filename.split('.')[0]
    
    success = add_dynamic_domain(domain_name, file_location)
    
    if os.path.exists(file_location):
        os.remove(file_location)
        
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process document. Ensure it contains valid JSONL.")
        
    return {"domain": domain_name, "message": "Upload processed, FAISS index built, and raw file safely discarded."}

@app.post("/api/query")
async def run_query(req: QueryRequest):
    domain = req.domain if req.domain else "healthcare"
    start = time.perf_counter()
    
    if req.filtered:
        idx, docs = load_domain_assets(domain)
        if idx is None or docs is None:
            idx, docs = None, ["Fallback context: No valid secure knowledge base found. Please upload a dataset."]
        
        response = secure_rag_pipeline(req.query, req.model, idx, docs, store_latency=True)
        lat = LATENCY_DATA[-1] if LATENCY_DATA else {}
        total_lat = int((time.perf_counter() - start) * 1000)
        
        is_blocked = str(response).startswith("INPUT_FILTER_BLOCKED")
        resp_text = "Query blocked by Semantic Filter." if is_blocked else response
        
        return {
            "query_id": str(uuid.uuid4())[:8],
            "query": req.query,
            "model": req.model,
            "response": resp_text,
            "blocked": is_blocked,
            "pii_detected": ["[REDACTED]"] if "[REDACTED]" in str(response) else [],
            "total_latency_ms": total_lat,
            "stages": [
                {"name": "Input Filter", "status": "BLOCKED" if is_blocked else "PASSED", "latency_ms": int(lat.get("input_filter", 0)*1000), "detail": "Semantic check complete"},
                {"name": "Vector Retrieval", "status": "SUCCESS", "latency_ms": int(lat.get("retrieval", 0)*1000), "detail": "FAISS search"},
                {"name": "LLM Generation", "status": "SUCCESS", "latency_ms": int(lat.get("model_gen", 0)*1000), "detail": f"{req.model} inference"},
                {"name": "Output Filter", "status": "REDACTED" if "[REDACTED]" in str(response) else "CLEAN", "latency_ms": int(lat.get("output_filter", 0)*1000), "detail": "Presidio NER scan"}
            ]
        }
    else:
        idx, docs = load_raw_domain(domain)
        if idx is None or docs is None:
            idx, docs = None, ["Fallback context: No valid raw knowledge base found. Please upload a dataset."]
        
        response = unfiltered_rag_pipeline(req.query, req.model, idx, docs, store_latency=True)
        lat = LATENCY_DATA[-1] if LATENCY_DATA else {}
        total_lat = int((time.perf_counter() - start) * 1000)
        
        return {
            "query_id": str(uuid.uuid4())[:8],
            "query": req.query,
            "model": req.model,
            "response": response,
            "blocked": False,
            "pii_detected": [],
            "total_latency_ms": total_lat,
            "stages": [
                {"name": "Input Filter", "status": "PASSED", "latency_ms": 0, "detail": "Bypassed (Unfiltered)"},
                {"name": "Vector Retrieval", "status": "SUCCESS", "latency_ms": int(lat.get("retrieval", 0)*1000), "detail": "FAISS raw search"},
                {"name": "LLM Generation", "status": "SUCCESS", "latency_ms": int(lat.get("model_gen", 0)*1000), "detail": f"{req.model} inference"},
                {"name": "Output Filter", "status": "CLEAN", "latency_ms": 0, "detail": "Bypassed (Unfiltered)"}
            ]
        }

def background_benchmark(job_id: str, req: BenchmarkRequest):
    domain = req.domain if req.domain else "healthcare"
    BENCHMARK_JOBS[job_id]["status"] = "running"
    results = []
    curr = 0
    
    if req.filtered:
        idx, docs = load_domain_assets(domain)
        if idx is None: idx, docs = None, ["Fallback"]
    else:
        idx, docs = load_raw_domain(domain)
        if idx is None: idx, docs = None, ["Fallback"]

    total_tests = sum(len(v) for v in TEST_SUITE.values()) * len(req.models)

    for model in req.models:
        print(f"\n--- Starting benchmark for model: {model.upper()} ---")
        for dimension, prompts in TEST_SUITE.items():
            for prompt in prompts:
                curr += 1
                passed = True
                try:
                    if req.filtered:
                        res = secure_rag_pipeline(prompt, model, idx, docs, store_latency=True)
                        if dimension == "Harmful_Content_Generation":
                            if not check_for_harmful_refusal(res): passed = False
                        elif check_for_leakage(res): passed = False
                    else:
                        res = unfiltered_rag_pipeline(prompt, model, idx, docs, store_latency=True)
                        if not check_for_refusal(res): passed = False
                        if check_for_leakage(res): passed = False
                except Exception as e:
                    print(f"Error during benchmark for {model}: {str(e)}")
                    passed = False
                    
                results.append({"model": model, "dimension": dimension, "passed": passed})
                BENCHMARK_JOBS[job_id]["progress"] = curr
                BENCHMARK_JOBS[job_id]["results"] = results
                
                print(f"[{curr}/{total_tests}] {model.upper()} | {dimension[:15]}: {'PASSED' if passed else 'FAILED'}")
                
                # API Cooldown to prevent rate limit freezing
                time.sleep(1.5)
                
    BENCHMARK_JOBS[job_id]["status"] = "complete"
    print(f"\n--- Benchmark {job_id} Complete ---")

@app.post("/api/benchmark/start")
async def start_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    total_tests = sum(len(v) for v in TEST_SUITE.values()) * len(req.models)
    BENCHMARK_JOBS[job_id] = {"status": "starting", "progress": 0, "total": total_tests, "results": []}
    background_tasks.add_task(background_benchmark, job_id, req)
    return {"job_id": job_id, "total": total_tests}

@app.get("/api/benchmark/status/{job_id}")
async def get_benchmark_status(job_id: str):
    if job_id not in BENCHMARK_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return BENCHMARK_JOBS[job_id]

@app.get("/api/analytics/latency")
async def get_latency():
    return {"data": LATENCY_DATA, "note": "Real-time latency metrics updated based on the active security mode."}

@app.get("/api/analytics/overview")
async def get_overview():
    return {"leaderboard": []}

@app.get("/api/history")
async def get_history(limit: int = 50):
    return []
