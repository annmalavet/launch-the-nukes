"""
This module provides a structured interface for managing job records in Firestore,
defining a Job data model and a FirestoreJobStore class to handle create, update,
retrieve, and list operations. The object-oriented design keeps Firestore logic
organized and encapsulated, making the system easier to extend and maintain.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from google.cloud import firestore
from enum import Enum
from dataclasses import dataclass

# Setting up an enumeration to status keyword in the system for generalization purpose
class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Defining the data schema for the Job Storage
@dataclass
class Job:
    # Created on runtime when the job request is submitted on the server,
    # used to reference when details on a specific prompt is needed
    job_id: str

    # for now, this user id is being generated out of the cookie value
    # by stripping to 8 characters, later user will have option to set the user name 
    # once login page is setup
    user_id: str

    # this is also being set as 'user-{cookie value}', the changes are similar to user_id
    username: str

    # storing the prompt given by the user for analysis purposes 
    prompt: str

    # Will be set as {JobStatus} which has been declared above based on the status of the job
    status: JobStatus

    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Storing the result of the prompt, with multiple parameters that gives
    # a full understanding of how the LLM responded to the prompt
    result: Optional[Dict[str, Any]] = None

    # this will be set as 0 at the start and 100 once ended, no intermediate 
    # stage is updated to limit writes to the DB
    progress: int = 0

    # Will be set as Queued Or Completed based on the status of the job
    progress_message: str = "Queued"

class FirestoreJobStore:
    def __init__(self, project_id: str):
        # Initialize Firestore client for the GCP project
        self.db = firestore.Client(project=project_id)
        self.col = self.db.collection("jobs")
    
    """
    This function converts the DB document to a dictionary that is readable 
    for the frontend so that it can be rendered precisely
    """
    def _doc_to_job(self, data) -> Job:
        d = data.to_dict()
        return Job(
            job_id=data.id,
            user_id=d["user_id"],
            username=d["username"],
            prompt=d["prompt"],
            status=JobStatus(d["status"]),
            created_at=d["created_at"],
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            result=d.get("result"),
            progress=d.get("progress", 0),
            progress_message=d.get("progress_message", "Queued"),
        )



    def create_job(self, job: Job) -> None:
        """
        Creating the document and setting it with the key value as the Job ID
        """
        self.col.document(job.job_id).set({
            "user_id": job.user_id,
            "username": job.username,
            "prompt": job.prompt,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result": job.result,
            "progress": job.progress,
            "progress_message": job.progress_message,
            "model": getattr(job, "model", "llama3.2"),
        })

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by its ID. Returns None if not found.
        """
        data = self.col.document(job_id).get()
        return self._doc_to_job(data) if data.exists else None

    def update_job(self, job_id: str, **fields) -> None:
        """
        Updating the Job Fields once the status changes from Queued to Completed
        Make the least use of this function to keep the system cost efficient and less queried on the DB
        """
        if "status" in fields and isinstance(fields["status"], JobStatus):
            fields["status"] = fields["status"].value
        self.col.document(job_id).update(fields)
    
    # This function has not been used yet but will be referenced once the Redis Implementation is limited to caching
    def list_jobs_for_user(self, user_id: str) -> List[Job]:
        q = (self.col.where("user_id", "==", user_id)
                   .order_by("created_at", direction=firestore.Query.DESCENDING))
        return [self._doc_to_job(s) for s in q.stream()]

class FirestoreMCPStore:
    def __init__(self, project_id: str):
        # Initialize Firestore client for the GCP project
        self.db = firestore.Client(project=project_id)
        self.col = self.db.collection("triggered_mcp_servers")

    def update_mcp_triggered(self, user_id: str, used_servers: List[Dict[str, str]]) -> None:
        """
        Updates the 'triggered_mcp_servers' collection for a user.

        If the user_id exists, appends tools under each MCP server.
        If the user_id does not exist, creates a new document.
        """
        user_doc = self.col.document(user_id)
        doc = user_doc.get()

        if doc.exists:
            data = doc.to_dict()
            existing_servers = data.get("used_servers", {})

            for entry in used_servers:
                server = entry["server"]
                tool = entry["tool"]

                if server not in existing_servers:
                    existing_servers[server] = []
                if tool not in existing_servers[server]:
                    existing_servers[server].append(tool)

            user_doc.update({"used_servers": existing_servers})

        else:
            # Build dict structure from input
            servers_dict = {}
            for entry in used_servers:
                server = entry["server"]
                tool = entry["tool"]
                servers_dict.setdefault(server, []).append(tool)

            user_doc.set({"used_servers": servers_dict})


    def get_used_servers(self, user_id):
        """
        Fetch the list of used servers for a given user ID from Firestore.

        Args:
            user_id (str): The ID of the user.

        Returns:
            list: A list of used servers for the user. Returns an empty list if the user does not exist.
        """
        user_doc = self.col.document(user_id)
        doc = user_doc.get()
        if doc.exists:
            return doc.to_dict().get("used_servers", {})
        return {}

