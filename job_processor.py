"""
This file works for the main processing of the prompt from start to end - 
caching on the redis server for queuing, prompt analysis to trigger the llm and servers
and then triggering functions to store data in the DB
"""
import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import multiprocessing
import signal
import sys

import redis

from llm_providers import OllamaProvider
from mcp_integration import MCPClient
from config import config
from firestore import FirestoreJobStore, FirestoreMCPStore

firestore_jobs_db = FirestoreJobStore(config.GOOGLE_CLOUD_PROJECT)
firestore_mcp_db = FirestoreMCPStore(config.GOOGLE_CLOUD_PROJECT)

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Defining the dataclass Job for redis caching till everything 
# related to the prompt is completed
@dataclass
class Job:
    job_id: str
    user_id: str
    username: str
    prompt: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0
    progress_message: str = "Queued"
    # Storing the queue position in Redis to process on a FIFO basis
    queue_position: int = 0
    model: str = "llama3.2"

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'username': self.username,
            'prompt': self.prompt,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'progress': self.progress,
            'progress_message': self.progress_message,
            'queue_position': self.queue_position,
            'model': self.model
        }

    """
    This function is being used by Redis to convert the data stored 
    from a specific format to a desired format for the web rendering.
    This will be deprecated later once the DB transactions are successfully saved
    as Redis will not store the data later, hence no need to fetch and transform
    """
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            job_id=data['job_id'],
            user_id=data['user_id'],
            username=data['username'],
            prompt=data['prompt'],
            status=JobStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data['started_at'] else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
            result=data.get('result'),
            error=data.get('error'),
            progress=data.get('progress', 0),
            progress_message=data.get('progress_message', 'Queued'),
            queue_position=data.get('queue_position', 0),
            model=data.get('model', 'llama3.2')
        )

class RedisJobQueue:
    """Redis-based job queue for multi-worker environments"""
    
    def __init__(self, redis_url: str = None):
        if redis_url is None:
            redis_url = config.REDIS_URL
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.jobs_key = "jobs"
        self.queue_key = "job_queue"
        self.processing_key = "processing_jobs"

    """
    Adding the job to the Redis Queue
    """    
    def add_job(self, user_id: str, username: str, prompt: str, job_id: str, model: str = "llama3.2") -> None:
        job = Job(
            job_id=job_id,
            user_id=user_id,
            username=username,
            prompt=prompt,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            model=model
        )
        
        # Store job data
        self.redis_client.hset(self.jobs_key, job_id, json.dumps(job.to_dict()))
        
        # Add to queue
        self.redis_client.lpush(self.queue_key, job_id)
        
        # Update queue positions
        self._update_queue_positions()
    
    def get_job(self, job_id: str) -> Optional[Job]:
        job_data = self.redis_client.hget(self.jobs_key, job_id)
        if job_data:
            return Job.from_dict(json.loads(job_data))
        return None
    
    def update_job(self, job_id: str, **kwargs):
        job_data = self.redis_client.hget(self.jobs_key, job_id)
        if job_data:
            job_dict = json.loads(job_data)
            for key, value in kwargs.items():
                if key == 'status' and isinstance(value, JobStatus):
                    job_dict[key] = value.value
                elif key in ['started_at', 'completed_at'] and isinstance(value, datetime):
                    job_dict[key] = value.isoformat()
                else:
                    job_dict[key] = value
            
            self.redis_client.hset(self.jobs_key, job_id, json.dumps(job_dict))
    
    def get_next_job(self) -> Optional[str]:
        """Get next job from queue (blocking pop)"""
        result = self.redis_client.brpop(self.queue_key, timeout=1)
        if result:
            _, job_id = result

            # Move to processing set
            self.redis_client.sadd(self.processing_key, job_id)
            self._update_queue_positions()
            return job_id
        return None
    
    def complete_job(self, job_id: str):
        """Remove job from processing set when completed"""
        self.redis_client.srem(self.processing_key, job_id)
        self._update_queue_positions()
    
    def get_user_jobs(self, user_id: str) -> List[Job]:
        """Get all jobs for a specific user"""
        all_jobs = self.redis_client.hgetall(self.jobs_key)
        user_jobs = []
        
        for job_data in all_jobs.values():
            job = Job.from_dict(json.loads(job_data))
            if job.user_id == user_id:
                user_jobs.append(job)
        
        return sorted(user_jobs, key=lambda x: x.created_at, reverse=True)
    
    def _update_queue_positions(self):
        """Update queue positions for all pending jobs"""
        pending_jobs = self.redis_client.lrange(self.queue_key, 0, -1)
        
        for position, job_id in enumerate(reversed(pending_jobs), 1):
            job_data = self.redis_client.hget(self.jobs_key, job_id)
            if job_data:
                job_dict = json.loads(job_data)
                job_dict['queue_position'] = position
                self.redis_client.hset(self.jobs_key, job_id, json.dumps(job_dict))
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        pending_count = self.redis_client.llen(self.queue_key)
        processing_count = self.redis_client.scard(self.processing_key)
        total_jobs = self.redis_client.hlen(self.jobs_key)
        
        return {
            'pending': pending_count,
            'processing': processing_count,
            'total': total_jobs
        }
    
    def get_queue_position(self, job_id: str) -> int:
        """Get the position of a job in the queue (1-based index)"""
        try:
            pending_jobs = self.redis_client.lrange(self.queue_key, 0, -1)
            
            for i, pending_job_id in enumerate(pending_jobs):
                if pending_job_id == job_id:
                    return i + 1  # 1-based position
            
            # If not in pending queue, check if it's processing or completed
            job = self.get_job(job_id)
            if job and job.status in [JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]:
                return 0  # Not in queue anymore
                
            return -1  # Job not found
            
        except Exception as e:
            print(f"Error getting queue position for job {job_id}: {e}")
            return -1
    
    def get_estimated_time(self, job_id: str) -> int:
        """Get estimated time until job completion in seconds"""
        try:
            position = self.get_queue_position(job_id)
            if position <= 0:
                return 0  # Job not in queue or not found
            
            # Estimate 30 seconds per job ahead in queue
            return (position - 1) * 30
            
        except Exception as e:
            print(f"Error getting estimated time for job {job_id}: {e}")
            return 0

class LLMProcessor:
    """LLM processor that runs in a separate process"""
    
    def __init__(self):
        self.mcp_client = MCPClient()
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama is running and has models available"""
        try:
            import httpx
            from config import config
            ollama_url = config.effective_ollama_url
            
            print(f"🔍 Checking Ollama health at: {ollama_url}")
            
            # Use longer timeout for Cloud Run internal communication
            timeout = 30.0 if config.is_production else 10.0
            client = httpx.Client(timeout=timeout)
            
            # Check if service is responding
            response = client.get(f"{ollama_url}/api/tags")
            if response.status_code != 200:
                print(f"❌ Ollama health check failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
            
            # Check if models are available
            models_data = response.json()
            models = models_data.get("models", [])
            
            if not models:
                print("⚠️ Ollama is running but no models are loaded yet")
                # In production, we might want to be more lenient about this
                if config.is_production:
                    print("   Allowing processing to continue in production mode")
                    return True
                return False
            
            print(f"✅ Ollama health check passed - {len(models)} models available")
            for model in models[:3]:  # Show first 3 models
                model_name = model.get('name', 'unknown')
                print(f"   - {model_name}")
            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more models")
            
            return True
            
        except httpx.ConnectError as e:
            print(f"❌ Ollama connection failed: {e}")
            print(f"   URL: {ollama_url}")
            if config.is_production:
                print("   This might be a temporary network issue in Cloud Run")
            return False
        except httpx.TimeoutException as e:
            print(f"❌ Ollama request timeout: {e}")
            print(f"   URL: {ollama_url}")
            return False
        except Exception as e:
            print(f"❌ Ollama health check failed: {e}")
            print(f"   URL: {ollama_url}")
            return False
    
    def process_prompt(self, job_id: str, prompt: str, job_queue, user_id: str, model: str = "llama3.2") -> Dict[str, Any]:
        """Process a prompt with LLM and MCP integration"""
        try:
            # Update progress
            job_queue.update_job(job_id, progress=2, progress_message="Checking service health...")
            
            # Check if Ollama is running with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                if self._check_ollama_health():
                    break
                elif attempt < max_retries - 1:
                    print(f"⏳ Ollama health check failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    job_queue.update_job(job_id, progress=2, progress_message=f"Waiting for Ollama service (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    raise Exception("Ollama service is not running after multiple attempts. Please check the service status.")
            
            job_queue.update_job(job_id, progress=5, progress_message="Initializing MCP clients...")
            
            # Get MCP tools with error handling
            try:
                tools_by_server = asyncio.run(self.mcp_client.list_tools())
            except Exception as e:
                print(f"Warning: MCP servers not available: {e}")
                tools_by_server = {}
            
            tool_to_server_map = {}
            
            job_queue.update_job(job_id, progress=10, progress_message="Preparing tools...")
            
            ollama_tools = []
            for server, tools in tools_by_server.items():
                for tool in tools:
                    tool_to_server_map[tool.name] = server
                    
                    ollama_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                        }
                    }
                    
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        ollama_tool["function"]["parameters"] = tool.inputSchema
                    
                    ollama_tools.append(ollama_tool)
            
            job_queue.update_job(job_id, progress=20, progress_message="Calling LLM...")
            
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": "You are a security analysis assistant. Analyze user prompts and use available tools when appropriate for security operations. Be decisive about tool usage."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Call LLM with streaming progress
            llm = OllamaProvider(model=model)
            
            # Create progress callback for streaming updates
            def progress_callback(progress: int, message: str):
                job_queue.update_job(job_id, progress=progress, progress_message=message)
            
            # Add a small delay to ensure "Calling LLM..." is visible
            time.sleep(0.5)
            
            response = llm.generate_with_tools_streaming(messages, ollama_tools, progress_callback)
            
            # Check if we got a valid response or an error
            if "Error:" in response.get("content", ""):
                raise Exception(f"LLM Error: {response.get('content', 'Unknown error')}")
            
            job_queue.update_job(job_id, progress=80, progress_message="Processing tool calls...")
            
            # Process tool calls
            try:
                available_servers = self.mcp_client.get_available_servers()
            except Exception as e:
                print(f"Warning: Could not get available servers: {e}")
                available_servers = {}
            
            total_mcps = len(available_servers)
            used_servers = []
            tool_call_results = []
            llm_message = response.get("content", "No response from LLM")
            
            tool_calls = response.get("tool_calls", [])
            
            if tool_calls:
                progress_step = 15 / len(tool_calls)  # 15% range for tool execution (80-95%)
                current_progress = 80
                
                for i, tool_call in enumerate(tool_calls):
                    try:
                        job_queue.update_job(
                            job_id, 
                            progress=int(current_progress + (i * progress_step)), 
                            progress_message=f"Executing tool {i+1}/{len(tool_calls)}..."
                        )
                        
                        if "function" in tool_call:
                            function_info = tool_call["function"]
                            tool_name = function_info.get("name")
                            
                            args_raw = function_info.get("arguments", {})
                            if isinstance(args_raw, str):
                                args = json.loads(args_raw)
                            else:
                                args = args_raw
                            
                            if tool_name in tool_to_server_map:
                                server_name = tool_to_server_map[tool_name]
                                
                                # Type conversion logic
                                tool_schema = None
                                for server, tools in tools_by_server.items():
                                    if server == server_name:
                                        for tool in tools:
                                            if tool.name == tool_name:
                                                tool_schema = tool.inputSchema
                                                break
                                
                                if tool_schema and 'properties' in tool_schema:
                                    converted_args = {}
                                    for arg_name, arg_value in args.items():
                                        if arg_name in tool_schema['properties']:
                                            expected_type = tool_schema['properties'][arg_name].get('type', 'string')
                                            
                                            try:
                                                if expected_type == 'number' and isinstance(arg_value, str):
                                                    converted_args[arg_name] = float(arg_value)
                                                elif expected_type == 'integer' and isinstance(arg_value, str):
                                                    converted_args[arg_name] = int(arg_value)
                                                elif expected_type == 'boolean' and isinstance(arg_value, str):
                                                    converted_args[arg_name] = arg_value.lower() in ('true', '1', 'yes')
                                                else:
                                                    converted_args[arg_name] = arg_value
                                            except (ValueError, TypeError):
                                                converted_args[arg_name] = arg_value
                                        else:
                                            converted_args[arg_name] = arg_value
                                    args = converted_args
                                
                                used_servers.append({"server":server_name, "tool":tool_name})
                                try:
                                    result = asyncio.run(self.mcp_client.call_tool(server_name, tool_name, args))
                                    # Changed the tool_call_results structure that is acceptable in GCP Firestore
                                    # Tuples are not accepted in Firestore
                                    tool_call_results.append({
                                        "tool": tool_name,
                                        "result": result
                                    })
                                except Exception as tool_error:
                                    error_msg = f"Tool execution failed: {str(tool_error)}"
                                    print(f"Error calling tool {tool_name}: {tool_error}")
                                    tool_call_results.append({
                                        "tool": tool_name,
                                        "result": error_msg
                                    })
                            else:
                                print(f"Tool {tool_name} not found in server mapping")
                            
                    except Exception as e:
                        error_msg = f"Error processing tool call: {str(e)}"
                        print(f"Error in tool call processing: {e}")
                        tool_call_results.append({
                            "tool": tool_call.get("function", {}).get("name", "unknown"),
                            "result": error_msg
                        })
            
            job_queue.update_job(job_id, progress=95, progress_message="Finalizing results...")
            
            # Prepare final result
            used_servers = used_servers
            num_servers_triggered = len(used_servers)
            risk_level = 'HIGH' if used_servers else 'SAFE'
            risk_color = 'red' if used_servers else 'green'
            
            result = {
                'prompt': prompt,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'total_mcps': total_mcps,
                'used_servers': used_servers,
                'num_servers_triggered': num_servers_triggered,
                'llm_message': llm_message,
                'llm_content': response.get("content", ""),
                'llm_tool_calls': response.get("tool_calls", []),
                'tool_call_results': tool_call_results,
                'analysis': llm_message
            }            

            # Updated the Redis Queue - this will be removed in further commits when the DB structure has been thoroughly tested
            job_queue.update_job(job_id, progress=100, progress_message="Completed")
            
            # Updated the firestore with final results of the prompt to be stored in the database
            firestore_jobs_db.update_job(job_id, progress=100, progress_message="Completed", result=result, completed_at=datetime.now())
            
            # Updated the triggered MCP server collection with used servers for that specific user id
            firestore_mcp_db.update_mcp_triggered(user_id, used_servers)
            
            return result
            
        except Exception as e:
            raise Exception(f"Error processing prompt: {str(e)}")

def worker_process(redis_url: str):
    """Worker process that processes jobs from the queue"""
    print(f"🟢 Starting worker process (PID: {os.getpid()})")
    
    # Create job queue and processor
    try:
        job_queue = RedisJobQueue(redis_url)
        processor = LLMProcessor()
        print(f"✅ Worker {os.getpid()} initialized successfully")
    except Exception as e:
        print(f"❌ Worker {os.getpid()} failed to initialize: {e}")
        return
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        print(f"🔴 Worker {os.getpid()} received shutdown signal {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    # Process jobs
    while True:
        try:
            job_id = job_queue.get_next_job()
            if job_id:
                print(f"📋 Worker {os.getpid()} processing job {job_id}")
                job = job_queue.get_job(job_id)
                user_id = job.user_id
                if not job:
                    print(f"⚠️ Job {job_id} not found, skipping")
                    job_queue.complete_job(job_id)
                    continue
                
                try:
                    # Update job status
                    job_queue.update_job(
                        job_id,
                        status=JobStatus.PROCESSING,
                        started_at=datetime.now(),
                        progress=5,
                        progress_message="Starting processing..."
                    )
                    
                    # Process the job
                    print("user_id", user_id)
                    result = processor.process_prompt(job_id, job.prompt, job_queue, user_id, job.model)
                    
                    # Update job with result
                    job_queue.update_job(
                        job_id,
                        status=JobStatus.COMPLETED,
                        completed_at=datetime.now(),
                        result=result,
                        progress=100,
                        progress_message="Completed successfully"
                    )
                    
                    print(f"✅ Worker {os.getpid()} completed job {job_id}")
                    consecutive_errors = 0  # Reset error counter on success
                    
                except Exception as e:
                    # Update job with error
                    error_message = str(e)
                    job_queue.update_job(
                        job_id,
                        status=JobStatus.FAILED,
                        completed_at=datetime.now(),
                        error=error_message,
                        progress=0,
                        progress_message=f"Failed: {error_message}"
                    )
                    
                    print(f"❌ Worker {os.getpid()} failed job {job_id}: {e}")
                    consecutive_errors += 1
                
                finally:
                    # Mark job as no longer processing
                    job_queue.complete_job(job_id)
            else:
                # No job available, reset error counter
                consecutive_errors = 0
            
            # Check if we've had too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"❌ Worker {os.getpid()} has {consecutive_errors} consecutive errors, restarting...")
                break
            
        except Exception as e:
            print(f"❌ Worker {os.getpid()} error: {e}")
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"❌ Worker {os.getpid()} has {consecutive_errors} consecutive errors, stopping...")
                break
            
            time.sleep(min(consecutive_errors * 2, 30))  # Exponential backoff, max 30 seconds
    
    print(f"🔴 Worker {os.getpid()} exiting")

class JobManager:
    """Main job manager that coordinates between web app and workers"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", num_workers: int = 1):
        self.redis_url = redis_url
        self.num_workers = num_workers
        self.workers = []
        
        # Create job queue
        self.job_queue = RedisJobQueue(redis_url)
    
    def start_workers(self):
        """Start worker processes"""
        print(f"Starting {self.num_workers} worker processes...")
        
        for i in range(self.num_workers):
            worker = multiprocessing.Process(
                target=worker_process,
                args=(self.redis_url,),
                name=f"JobWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)
            print(f"✅ Started worker {i} (PID: {worker.pid})")
    
    def stop_workers(self):
        """Stop all worker processes"""
        print("Stopping worker processes...")
        
        for worker in self.workers:
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
        
        self.workers = []
        print("✅ All workers stopped")
    
    def get_job_queue(self):
        """Get the job queue instance"""
        return self.job_queue
    
    def get_queue_stats(self):
        """Get queue statistics"""
        return self.job_queue.get_queue_stats()
    
    def check_worker_health(self):
        """Check health of worker processes and return list of dead workers"""
        dead_workers = []
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                dead_workers.append(i)
        return dead_workers
    
    def restart_dead_workers(self):
        """Restart any dead worker processes"""
        dead_workers = self.check_worker_health()
        for worker_index in dead_workers:
            try:
                # Terminate the dead worker
                old_worker = self.workers[worker_index]
                if old_worker.is_alive():
                    old_worker.terminate()
                    old_worker.join(timeout=5)
                
                # Create new worker
                new_worker = multiprocessing.Process(
                    target=worker_process,
                    args=(self.redis_url,),
                    name=f"JobWorker-{worker_index}"
                )
                new_worker.start()
                self.workers[worker_index] = new_worker
                print(f"✅ Restarted worker {worker_index} (PID: {new_worker.pid})")
                
            except Exception as e:
                print(f"❌ Failed to restart worker {worker_index}: {e}")
        
        return len(dead_workers)

# Global job manager instance
job_manager = None

def init_job_manager(redis_url: str = "redis://localhost:6379/0", num_workers: int = 1):
    """Initialize the global job manager"""
    global job_manager
    job_manager = JobManager(redis_url, num_workers)
    return job_manager

def get_job_queue(redis_url: str = None):
    """Get the job queue from the global manager or create a new one"""
    if job_manager:
        return job_manager.get_job_queue()
    else:
        # Fallback for development - requires Redis
        return RedisJobQueue(redis_url)

# Health check functions
def check_services():
    """Check if required services are running"""
    print("🔍 Checking service health...")
    
    # Check Redis
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running")
    except Exception as e:
        print(f"❌ Redis not accessible: {e}")
        print("   Please start Redis with: redis-server")
    
    # Check Ollama
    try:
        import httpx
        from config import config
        
        # Use the configured Ollama URL, falling back to cloud URL if available
        ollama_url = config.effective_ollama_url
            
        print(f"🔍 Checking Ollama health at: {ollama_url}")
        
        client = httpx.Client(timeout=5.0)
        response = client.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            print("✅ Ollama service is running")
        else:
            print("⚠️  Ollama service responded with error")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Ollama service is not running")
        print("   Please start Ollama with: ollama serve")
    
    # Check MCP servers
    try:
        mcp_client = MCPClient()
        servers = mcp_client.get_available_servers()
        if servers:
            print(f"✅ MCP servers available: {list(servers.keys())}")
        else:
            print("⚠️  No MCP servers available")
    except Exception as e:
        print(f"MCP servers not accessible: {e}")

if __name__ == "__main__":
    check_services()
