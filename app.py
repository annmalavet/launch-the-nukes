from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response
import os
import uuid
from datetime import datetime
from job_processor import get_job_queue, JobStatus, worker_process
import time
from config import config
from mcp_integration import MCPClient
import asyncio
from firestore import FirestoreJobStore, FirestoreMCPStore, Job


app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Instantiating Firestore
firestore_jobs_db=FirestoreJobStore(config.GOOGLE_CLOUD_PROJECT)
firestore_mcp_db=FirestoreMCPStore(config.GOOGLE_CLOUD_PROJECT)

def get_mcp_servers():
    """Get MCP servers using the existing MCP integration - no server execution needed"""
    try:
        print("Getting MCP servers using mcp_integration.list_tools()")
        
        # Create MCP client (this just reads YAML configs, doesn't run servers)
        client = MCPClient()
        
        # Get tools from all servers (this uses the YAML factory, no server execution)
        tools_by_server = asyncio.run(client.list_tools())
        
        print(f"Found {len(tools_by_server)} servers")
        
        servers = {}
        for server_name, tool_list in tools_by_server.items():
            if not tool_list:
                continue
                
            # Get server description from the client's available servers
            available_servers = client.get_available_servers()
            description = available_servers.get(server_name, 'MCP Server')
            
            tools = []
            for tool in tool_list:
                tool_dict = {
                    'name': tool.name,
                    'description': tool.description,
                    'properties': []
                }
                
                # Extract properties from the tool's input schema
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    schema = tool.inputSchema
                    if isinstance(schema, dict) and 'properties' in schema:
                        required = schema.get('required', [])
                        for prop_name, prop_info in schema['properties'].items():
                            tool_dict['properties'].append({
                                'name': prop_name,
                                'type': prop_info.get('type', 'string'),
                                'description': prop_info.get('description', 'No description'),
                                'required': prop_name in required
                            })
                
                tools.append(tool_dict)
            
            servers[server_name] = {
                'description': description,
                'tools': tools
            }
            
            print(f"Loaded {server_name}: {len(tools)} tools")
        
        print(f"Final result: {len(servers)} servers loaded")
        return servers
        
    except Exception as e:
        print(f"Error loading MCP servers: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_user_id():
    """Get or create anonymous user ID from cookie"""
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    return user_id

def set_user_cookie(response, user_id):
    """Set user ID cookie with 30 day expiration (refresh on each visit)"""
    response.set_cookie('user_id', user_id, max_age=30*24*60*60, httponly=True, secure=False, samesite='Lax')

@app.route('/')
def index():
    """Redirect to login if not authenticated, dashboard if authenticated"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    user_id = get_user_id()
    
    # Get MCP servers 
    mcp_servers = get_mcp_servers()

    # Fetch used servers from Firestore collection triggered_mcp_severs
    used_servers = firestore_mcp_db.get_used_servers(user_id)
    
    response = make_response(render_template('dashboard.html', 
                                           username=f'User-{user_id[:8]}',
                                           mcp_servers=mcp_servers,
                                           used_servers=used_servers 
                                           ))
    set_user_cookie(response, user_id)
    return response

@app.route('/submit', methods=['POST'])
def submit():
    user_id = get_user_id()
    user_input = request.form.get('user_input', '').strip()
    model = request.form.get('model', 'llama3.2')

    if not user_input:
        flash('Please enter some text', 'error')
        return redirect(url_for('dashboard'))

    # Get job queue (connects to Redis directly)
    job_queue = get_job_queue(config.REDIS_URL)
    if not job_queue:
        flash('Job processing service unavailable', 'error')
        return redirect(url_for('dashboard'))

    react_mode = request.form.get('react_mode') == 'true'
    previous_job_id = request.form.get('previous_job_id', '')
    react_turn_count = int(request.form.get('react_turn_count', 0)) + 1

    # Load prior conversation messages if continuing a ReAct chain
    prior_messages = []
    if react_mode and previous_job_id:
        prev_job = job_queue.get_job(previous_job_id)
        if prev_job and prev_job.result:
            prior_messages = prev_job.result.get('messages', [])

    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id,
                user_id=user_id,
                username=f'User-{user_id[:8]}',
                prompt=user_input,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                started_at=datetime.now())

    # This will create a DB document to be stored on GCP Firestore
    firestore_jobs_db.create_job(job)

    # This will add the job to queue for processing the prompt in the Redis Caching System
    job_queue.add_job(user_id, f'User-{user_id[:8]}', user_input, job_id,
                      model=model, react_mode=react_mode, prior_messages=prior_messages,
                      react_turn_count=react_turn_count)
    
    response = make_response(redirect(url_for('job_status', job_id=job_id)))
    set_user_cookie(response, user_id)
    return response

@app.route('/job/<job_id>')
def job_status(job_id):
    user_id = get_user_id()
    job_queue = get_job_queue(config.REDIS_URL)
    
    if not job_queue:
        flash('Job processing service unavailable', 'error')
        return redirect(url_for('dashboard'))
    
    job = job_queue.get_job(job_id)
    
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('dashboard'))
    
    response = make_response(render_template('job_status.html', 
                                           job=job, 
                                           job_id=job_id,
                                           username=f'User-{user_id[:8]}'))
    set_user_cookie(response, user_id)
    return response

@app.route('/api/job/<job_id>/status')
def api_job_status(job_id):
    job_queue = get_job_queue(config.REDIS_URL)
    
    if not job_queue:
        return jsonify({'error': 'Job processing service unavailable'}), 503
    
    job = job_queue.get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify({
        'id': job_id,
        'status': job.status.value,
        'progress': job.progress,
        'progress_message': job.progress_message,
        'result': job.result,
        'queue_position': job_queue.get_queue_position(job_id),
        'estimated_time': job_queue.get_estimated_time(job_id)
    })

@app.route('/api/mcp/servers')
def api_mcp_servers():
    """API endpoint to get MCP servers"""
    servers = get_mcp_servers()
    return jsonify(servers)

@app.route('/api/queue/stats')
def api_queue_stats():
    job_queue = get_job_queue(config.REDIS_URL)
    
    if not job_queue:
        return jsonify({'error': 'Job processing service unavailable'}), 503
    
    stats = job_queue.get_queue_stats()
    return jsonify(stats)

@app.route('/results/<job_id>')
def results(job_id):
    user_id = get_user_id()
    job_queue = get_job_queue(config.REDIS_URL)
    job = job_queue.get_job(job_id)
    
    if not job or job.status != JobStatus.COMPLETED:
        flash('Job not found or not completed', 'error')
        return redirect(url_for('dashboard'))
    
    response = make_response(render_template('results.html',
                                           **job.result,
                                           job_id=job_id,
                                           username=f'User-{user_id[:8]}'))
    set_user_cookie(response, user_id)
    return response

@app.route('/my-jobs')
def my_jobs():
    user_id = get_user_id()
    job_queue = get_job_queue(config.REDIS_URL)
    user_jobs = job_queue.get_user_jobs(user_id)
    
    response = make_response(render_template('my_jobs.html', 
                                           jobs=user_jobs, 
                                           username=f'User-{user_id[:8]}'))
    set_user_cookie(response, user_id)
    return response

@app.route('/health')
def health_check():
    """Health check endpoint - doesn't require user tracking"""
    job_queue = get_job_queue(config.REDIS_URL)
    redis_status = "healthy" if job_queue else "unavailable"
    
    return jsonify({
        'status': 'healthy',
        'redis': redis_status,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    if config.DEBUG and os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        import multiprocessing
        worker = multiprocessing.Process(target=worker_process, args=(config.REDIS_URL,), daemon=True)
        worker.start()
        print('🟢 Dev worker started')
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
