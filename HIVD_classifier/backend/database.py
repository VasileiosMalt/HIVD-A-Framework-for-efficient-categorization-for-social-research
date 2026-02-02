"""
Database manager for HIVD Classifier using SQLite
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

import aiosqlite
import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from models import ClassDefinition, ClassificationJob, ClassificationResult, ClassificationStatus


class Database:
    """Async SQLite database manager"""
    
    def __init__(self, db_path: str = "hivd_classifier.db"):
        self.db_path = db_path
        
    async def init_db(self):
        """Initialize database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Classes table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    reference_images TEXT NOT NULL,
                    parent_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_id) REFERENCES classes (id)
                )
            """)

            # Safe migration: Add parent_id if it doesn't exist
            try:
                await db.execute("ALTER TABLE classes ADD COLUMN parent_id INTEGER REFERENCES classes (id)")
            except:
                pass
            
            # Jobs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT NOT NULL,
                    total_images INTEGER DEFAULT 0,
                    processed_images INTEGER DEFAULT 0,
                    progress_percentage REAL DEFAULT 0.0,
                    high_confidence_count INTEGER DEFAULT 0,
                    low_confidence_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_class_id INTEGER NOT NULL,
                    predicted_class_name TEXT NOT NULL,
                    predicted_subclass_id INTEGER,
                    predicted_subclass_name TEXT,
                    probability REAL NOT NULL,
                    is_high_confidence BOOLEAN NOT NULL,
                    safety_net_applied BOOLEAN DEFAULT FALSE,
                    safety_net_action TEXT,
                    detected_objects TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id),
                    FOREIGN KEY (predicted_class_id) REFERENCES classes (id),
                    FOREIGN KEY (predicted_subclass_id) REFERENCES classes (id)
                )
            """)

            # Safe migration: Add predicted_subclass info if it doesn't exist
            try:
                await db.execute("ALTER TABLE results ADD COLUMN predicted_subclass_id INTEGER REFERENCES classes (id)")
                await db.execute("ALTER TABLE results ADD COLUMN predicted_subclass_name TEXT")
            except:
                pass
            
            await db.commit()
    
    # ===== Classes CRUD =====
    
    async def create_class(self, class_def: ClassDefinition) -> int:
        """Create a new class"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO classes (name, description, reference_images, parent_id)
                VALUES (?, ?, ?, ?)
                """,
                (class_def.name, class_def.description, 
                 json.dumps(class_def.reference_images), class_def.parent_id)
            )
            await db.commit()
            return cursor.lastrowid
    
    async def get_class(self, class_id: int) -> Optional[ClassDefinition]:
        """Get a class by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM classes WHERE id = ?", (class_id,)
            )
            row = await cursor.fetchone()
            if row:
                return ClassDefinition(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    reference_images=json.loads(row['reference_images']),
                    parent_id=row['parent_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
    
    async def get_all_classes(self) -> List[ClassDefinition]:
        """Get all classes"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM classes ORDER BY created_at")
            rows = await cursor.fetchall()
            return [
                ClassDefinition(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    reference_images=json.loads(row['reference_images']),
                    parent_id=row['parent_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]
    
    async def update_class(self, class_id: int, class_def: ClassDefinition) -> bool:
        """Update a class"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE classes 
                SET name = ?, description = ?, reference_images = ?, parent_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (class_def.name, class_def.description,
                 json.dumps(class_def.reference_images), class_def.parent_id, class_id)
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def delete_class(self, class_id: int) -> bool:
        """Delete a class"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM classes WHERE id = ?", (class_id,))
            await db.commit()
            return cursor.rowcount > 0
    
    # ===== Jobs CRUD =====
    
    async def create_job(self, job: ClassificationJob) -> int:
        """Create a new classification job"""
        async with aiosqlite.connect(self.db_path) as db:
            # Handle both Enum and raw string
            status_val = job.status.value if hasattr(job.status, 'value') else str(job.status)
            
            cursor = await db.execute(
                """
                INSERT INTO jobs (job_name, status, config)
                VALUES (?, ?, ?)
                """,
                (job.job_name, status_val, job.config.model_dump_json())
            )
            await db.commit()
            return cursor.lastrowid
    
    async def get_job(self, job_id: int) -> Optional[ClassificationJob]:
        """Get a job by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
            if row:
                from models import ClassificationConfig
                return ClassificationJob(
                    id=row['id'],
                    job_name=row['job_name'],
                    status=ClassificationStatus(row['status'].split('.')[-1].lower()),
                    config=ClassificationConfig.model_validate_json(row['config']),
                    total_images=row['total_images'],
                    processed_images=row['processed_images'],
                    progress_percentage=row['progress_percentage'],
                    high_confidence_count=row['high_confidence_count'],
                    low_confidence_count=row['low_confidence_count'],
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    error_message=row['error_message']
                )
            return None
    
    async def get_all_jobs(self) -> List[ClassificationJob]:
        """Get all jobs ordered by creation time (newest first)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            
            from models import ClassificationConfig
            return [
                ClassificationJob(
                    id=row['id'],
                    job_name=row['job_name'],
                    status=ClassificationStatus(row['status'].split('.')[-1].lower()),
                    config=ClassificationConfig.model_validate_json(row['config']),
                    total_images=row['total_images'],
                    processed_images=row['processed_images'],
                    progress_percentage=row['progress_percentage'],
                    high_confidence_count=row['high_confidence_count'],
                    low_confidence_count=row['low_confidence_count'],
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    error_message=row['error_message']
                )
                for row in rows
            ]

    async def delete_job(self, job_id: int) -> bool:
        """Delete a job and its results"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete results first (foreign key constraint might not be enforced but good practice)
            await db.execute("DELETE FROM results WHERE job_id = ?", (job_id,))
            # Delete job
            cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def delete_all_jobs(self) -> int:
        """Delete all jobs and their results"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete all results
            await db.execute("DELETE FROM results")
            # Delete all jobs
            cursor = await db.execute("DELETE FROM jobs")
            await db.commit()
            return cursor.rowcount

    async def update_job_status(self, job_id: int, status: ClassificationStatus,
                               error_message: Optional[str] = None):
        """Update job status"""
        async with aiosqlite.connect(self.db_path) as db:
            # Handle both Enum and raw string
            status_val = status.value if hasattr(status, 'value') else str(status)
            
            if status == ClassificationStatus.RUNNING:
                await db.execute(
                    "UPDATE jobs SET status = ?, started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status_val, job_id)
                )
            elif status == ClassificationStatus.COMPLETED:
                await db.execute(
                    "UPDATE jobs SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status_val, job_id)
                )
            elif status == ClassificationStatus.FAILED:
                await db.execute(
                    "UPDATE jobs SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status_val, error_message, job_id)
                )
            else:
                await db.execute(
                    "UPDATE jobs SET status = ? WHERE id = ?",
                    (status_val, job_id)
                )
            await db.commit()
    
    async def update_job_progress(self, job_id: int, processed: int, total: int):
        """Update job progress"""
        progress = (processed / total * 100) if total > 0 else 0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE jobs 
                SET processed_images = ?, total_images = ?, progress_percentage = ?
                WHERE id = ?
                """,
                (processed, total, progress, job_id)
            )
            await db.commit()
    
    async def save_result(self, result: ClassificationResult):
        """Save a classification result"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO results (
                    job_id, image_path, predicted_class_id, predicted_class_name,
                    predicted_subclass_id, predicted_subclass_name,
                    probability, is_high_confidence, safety_net_applied,
                    safety_net_action, detected_objects
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.job_id, result.image_path, result.predicted_class_id,
                    result.predicted_class_name, result.predicted_subclass_id,
                    result.predicted_subclass_name, result.probability,
                    result.is_high_confidence, result.safety_net_applied,
                    result.safety_net_action,
                    json.dumps(result.detected_objects) if result.detected_objects else None
                )
            )
            await db.commit()
    
    async def get_job_results(self, job_id: int) -> List[ClassificationResult]:
        """Get all results for a job"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM results WHERE job_id = ?", (job_id,)
            )
            rows = await cursor.fetchall()
            return [
                ClassificationResult(
                    job_id=row['job_id'],
                    image_path=row['image_path'],
                    predicted_class_id=row['predicted_class_id'],
                    predicted_class_name=row['predicted_class_name'],
                    predicted_subclass_id=row['predicted_subclass_id'],
                    predicted_subclass_name=row['predicted_subclass_name'],
                    probability=row['probability'],
                    is_high_confidence=row['is_high_confidence'],
                    safety_net_applied=row['safety_net_applied'],
                    safety_net_action=row['safety_net_action'],
                    detected_objects=json.loads(row['detected_objects']) if row['detected_objects'] else None
                )
                for row in rows
            ]
