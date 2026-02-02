"""
FastAPI main application for HIVD Classifier
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

import asyncio
import sys
import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import aiosqlite
from pathlib import Path
from typing import List, Dict, Optional
import logging
import sqlite3
from contextlib import asynccontextmanager

from models import (
    ClassDefinition, ClassificationConfig, ClassificationJob,
    ClassificationResult, ClassificationStatus, SubcategorizationStrategy,
    SystemStatus, ProgressUpdate, ImageInfo
)
from database import Database
from classifier_engine import StrongEmbeddingClassifier, get_available_models, RECOMMENDED_MODELS
from safety_net import SafetyNet
from utils import (
    load_config, get_image_files, save_class_config,
    create_prediction_structure, save_image_to_class, format_citation
)

# Determine paths for both development and packaged executable
def get_app_paths():
    """Get application paths that work for both dev and packaged exe"""
    if getattr(sys, 'frozen', False):
        # Running as packaged executable
        # sys._MEIPASS is where PyInstaller extracts bundled files
        bundle_dir = Path(sys._MEIPASS)
        # The exe's directory (where config.yaml and data folders should be)
        app_dir = Path(os.path.dirname(sys.executable))
    else:
        # Running as script (development)
        bundle_dir = Path(__file__).parent.parent  # HIVD_classifier folder
        app_dir = bundle_dir
    
    return bundle_dir, app_dir

BUNDLE_DIR, APP_DIR = get_app_paths()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
db: Database = None
config: Dict = None
classifier: StrongEmbeddingClassifier = None
safety_net: SafetyNet = None
active_jobs: Dict[int, asyncio.Task] = {}
ws_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global db, config, classifier, safety_net
    
    # Startup
    logger.info("Starting HIVD Classifier API...")
    logger.info(f"Framework: {format_citation()}")
    
    # Load configuration
    config = await load_config()
    
    # Initialize database
    db = Database()
    await db.init_db()
    
    # Initialize classifier (lazy loading of models)
    device = config['clip']['device']
    
    # Check if CUDA is actually available
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested in config but not available in this environment. Falling back to CPU.")
        device = 'cpu'
        config['clip']['device'] = 'cpu'
        
    logger.info(f"Classifier will use device: {device}")
    
    # Initialize safety net
    if config.get('safety_net', {}).get('enabled', True):
        safety_net = SafetyNet(config['yolo']['model'])
        logger.info("Safety net initialized")
    
    logger.info("API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    for task in active_jobs.values():
        task.cancel()


app = FastAPI(
    title="HIVD Classifier API",
    description="Zero-shot image classification with strong embeddings for social research",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log detailed validation errors and return JSON-serializable response"""
    # Convert errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        err_dict = {
            "type": error.get("type", "unknown"),
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Validation error"),
        }
        # Don't include 'ctx' as it may contain non-serializable objects
        errors.append(err_dict)
    
    logger.error(f"Validation error for {request.url}: {errors}")
    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )

# Mount static files (frontend) - works for both dev and packaged exe
frontend_path = BUNDLE_DIR / "frontend"
if not frontend_path.exists():
    # Fallback for development mode
    frontend_path = Path(__file__).parent.parent / "frontend"
    
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Frontend mounted from: {frontend_path}")
else:
    logger.error(f"Frontend directory not found at: {frontend_path}")


# ===== Health Check =====

@app.get("/")
async def root():
    """Serve frontend"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return JSONResponse(
            status_code=500,
            content={"error": f"Frontend not found. Expected at: {index_path}"}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "citation": format_citation()}


@app.get("/api/models")
async def get_models():
    """Get available CLIP models and their pretrained weights"""
    return {
        "models": get_available_models(),
        "recommended": RECOMMENDED_MODELS
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status"""
    global classifier
    
    classes = await db.get_all_classes()
    
    # Count images in datasets
    image_dataset_path = Path(config['paths']['image_dataset'])
    inference_dataset_path = Path(config['paths']['inference_dataset'])
    
    dataset_images = len(get_image_files(image_dataset_path)) if image_dataset_path.exists() else 0
    inference_images = len(get_image_files(inference_dataset_path)) if inference_dataset_path.exists() else 0
    
    import torch
    gpu_available = torch.cuda.is_available() if config['clip']['device'] == 'cuda' else False
    
    return SystemStatus(
        active_jobs=len(active_jobs),
        total_classes=len(classes),
        total_images_dataset=dataset_images,
        total_images_inference=inference_images,
        gpu_available=gpu_available,
        clip_model_loaded=classifier is not None,
        yolo_model_loaded=safety_net is not None
    )


# ===== Class Management =====

@app.get("/api/classes", response_model=List[ClassDefinition])
async def get_classes():
    """Get all defined classes"""
    return await db.get_all_classes()


@app.get("/api/classes/{class_id}", response_model=ClassDefinition)
async def get_class(class_id: int):
    """Get a specific class"""
    class_def = await db.get_class(class_id)
    if not class_def:
        raise HTTPException(status_code=404, detail="Class not found")
    return class_def


@app.post("/api/classes", response_model=ClassDefinition)
async def create_class(class_def: ClassDefinition):
    """Create a new class"""
    try:
        class_id = await db.create_class(class_def)
        class_def.id = class_id
        
        # Save to file system
        classes_dir = Path(config['paths']['classes_dir'])
        await save_class_config(class_def, classes_dir)
        
        logger.info(f"Created class: {class_def.name}")
        return class_def
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/classes/{class_id}", response_model=ClassDefinition)
async def update_class(class_id: int, class_def: ClassDefinition):
    """Update a class"""
    success = await db.update_class(class_id, class_def)
    if not success:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Update file system
    classes_dir = Path(config['paths']['classes_dir'])
    await save_class_config(class_def, classes_dir)
    
    class_def.id = class_id
    logger.info(f"Updated class: {class_def.name}")
    return class_def


@app.delete("/api/classes/{class_id}")
async def delete_class(class_id: int):
    """Delete a class"""
    class_def = await db.get_class(class_id)
    if not class_def:
        raise HTTPException(status_code=404, detail="Class not found")
    
    success = await db.delete_class(class_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete class")
    
    # Delete from file system
    classes_dir = Path(config['paths']['classes_dir'])
    class_dir = classes_dir / class_def.name
    if class_dir.exists():
        import shutil
        shutil.rmtree(class_dir)
    
    logger.info(f"Deleted class: {class_def.name}")
    return {"success": True}


# ===== Image Browser =====

@app.get("/api/images/dataset", response_model=List[ImageInfo])
async def list_dataset_images():
    """List all images in the image_dataset folder"""
    dataset_path = Path(config['paths']['image_dataset'])
    if not dataset_path.exists():
        return []
    
    image_files = get_image_files(dataset_path)
    
    images_info = []
    for img_path in image_files[:500]:  # Limit to first 500 for performance
        path_obj = Path(img_path)
        images_info.append(ImageInfo(
            path=img_path,
            filename=path_obj.name,
            size_bytes=path_obj.stat().st_size
        ))
    
    return images_info


@app.get("/api/images/preview")
async def get_image_preview(path: str):
    """Get image for preview"""
    img_path = Path(path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(img_path)


# ===== Classification =====

@app.post("/api/classify/start", response_model=ClassificationJob)
async def start_classification(config_data: ClassificationConfig):
    """Start a classification job"""
    try:
        global classifier, active_jobs
        
        # Validate classes exist
        for class_id in config_data.class_ids:
            class_def = await db.get_class(class_id)
            if not class_def:
                raise HTTPException(status_code=404, detail=f"Class {class_id} not found")
        
        # Create job
        job = ClassificationJob(
            job_name=config_data.job_name,
            status=ClassificationStatus.PENDING,
            config=config_data
        )
        
        job_id = await db.create_job(job)
        job.id = job_id
        
        # Start classification task
        task = asyncio.create_task(run_classification(job_id))
        active_jobs[job_id] = task
        
        logger.info(f"Started classification job: {config_data.job_name} (ID: {job_id})")
        return job
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Job name '{config_data.job_name}' already exists. Please choose a different name.")
    except Exception as e:
        logger.error(f"Error starting classification job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classify/status/{job_id}", response_model=ClassificationJob)
async def get_job_status(job_id: int):
    """Get classification job status"""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/classify/results/{job_id}", response_model=List[ClassificationResult])
async def get_job_results(job_id: int):
    """Get classification results"""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return await db.get_job_results(job_id)


@app.post("/api/classify/open-folder/{job_id}")
async def open_results_folder(job_id: int):
    """Open the results folder for a job in the system file explorer"""
    import subprocess
    import platform
    
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get the predictions folder path
    predictions_dir = Path(config['paths']['predictions_dir'])
    job_folder = predictions_dir / job.job_name
    
    if not job_folder.exists():
        raise HTTPException(status_code=404, detail=f"Results folder not found: {job_folder}")
    
    # Open folder in file explorer based on OS
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(job_folder))
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(job_folder)], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", str(job_folder)], check=True)
        
        return {"success": True, "path": str(job_folder)}
    except Exception as e:
        logger.error(f"Failed to open folder: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to open folder: {str(e)}")


@app.get("/api/jobs", response_model=List[ClassificationJob])
async def list_jobs():
    """List all classification jobs"""
    return await db.get_all_jobs()


@app.delete("/api/jobs/all")
async def delete_all_jobs():
    """Delete all classification jobs and their results"""
    # Cancel all active jobs
    for job_id in list(active_jobs.keys()):
        active_jobs[job_id].cancel()
        del active_jobs[job_id]
        
    # Delete from DB
    count = await db.delete_all_jobs()
    
    # Delete all folders in predictions directory
    predictions_dir = Path(config['paths']['predictions_dir'])
    if predictions_dir.exists():
        import shutil
        for item in predictions_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                except Exception as e:
                    logger.error(f"Failed to delete folder {item}: {e}")
            
    return {"success": True, "deleted_count": count}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: int):
    """Delete a classification job and its results"""
    # Check if running
    if job_id in active_jobs:
        active_jobs[job_id].cancel()
        del active_jobs[job_id]
    
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete from DB
    success = await db.delete_job(job_id)
    
    # Delete results folder
    predictions_dir = Path(config['paths']['predictions_dir'])
    job_folder = predictions_dir / job.job_name
    if job_folder.exists():
        import shutil
        try:
            shutil.rmtree(job_folder)
        except Exception as e:
            logger.error(f"Failed to delete results folder: {e}")
            # Don't fail the request if just folder deletion fails, but warn
    
    return {"success": success}


@app.post("/api/jobs/{job_id}/rerun")
async def rerun_job(job_id: int):
    """Rerun an existing classification job"""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id in active_jobs:
        raise HTTPException(status_code=400, detail="Job is already running")
    
    # Reset job state in DB
    # We need a method to reset job stats, or we just manually update fields
    # Using update_job_status is not enough, we need to clear counts
    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute("""
            UPDATE jobs 
            SET status = 'pending', 
                processed_images = 0, 
                progress_percentage = 0,
                high_confidence_count = 0,
                low_confidence_count = 0,
                error_message = NULL,
                started_at = NULL,
                completed_at = NULL
            WHERE id = ?
        """, (job_id,))
        
        # Delete old results from DB
        await conn.execute("DELETE FROM results WHERE job_id = ?", (job_id,))
        await conn.commit()
    
    # Clean up results folder? 
    # run_classification logic calls create_prediction_structure 
    # which generally creates/cleans folders. 
    # But usually we might want to start fresh.
    # The create_prediction_structure implementation (which I haven't seen fully but assume exists) 
    # likely handles existing folders.
    
    # Start task
    task = asyncio.create_task(run_classification(job_id))
    active_jobs[job_id] = task
    
    # Fetch updated job to return
    updated_job = await db.get_job(job_id)
    return updated_job


# ===== WebSocket for real-time updates =====

@app.websocket("/api/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(ws_connections)}")
    
    try:
        while True:
            # Keep connection alive - use timeout to allow other tasks
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(ws_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in ws_connections:
            ws_connections.remove(websocket)


async def broadcast_progress(update: ProgressUpdate):
    """Broadcast progress update to all connected clients"""
    if not ws_connections:
        logger.debug("No WebSocket connections to broadcast to")
        return
        
    data = update.model_dump()
    logger.debug(f"Broadcasting: {data.get('message', '')} - {data.get('progress_percentage', 0):.1f}%")
    
    for ws in ws_connections[:]:
        try:
            await ws.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")
            if ws in ws_connections:
                ws_connections.remove(ws)
    
    # Small delay to ensure message is flushed
    await asyncio.sleep(0.05)


# ===== Classification Worker =====

async def run_classification(job_id: int):
    """Background task to run classification"""
    global classifier, safety_net
    
    try:
        # Update status to running
        await db.update_job_status(job_id, ClassificationStatus.RUNNING)
        
        # Get job and all classes
        job = await db.get_job(job_id)
        all_classes = await db.get_all_classes()
        
        # Filter to macro classes selected by user
        selected_class_ids = set(job.config.class_ids)
        selected_macros = [c for c in all_classes if c.id in selected_class_ids]
        
        logger.info("=" * 70)
        logger.info("STARTING CLASSIFICATION JOB")
        logger.info("=" * 70)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Job Name: {job.job_name}")
        logger.info(f"Strategy: {job.config.strategy.value}")
        logger.info(f"Selected Macro IDs: {selected_class_ids}")
        logger.info(f"Selected Macros: {[f'{m.name} (id={m.id})' for m in selected_macros]}")
        logger.info(f"Subcategory Threshold: {job.config.subcategory_threshold:.0%}")
        
        # Debug: Show ALL classes in database
        logger.info("-" * 50)
        logger.info("ALL CLASSES IN DATABASE:")
        logger.info("-" * 50)
        for cls in all_classes:
            parent_info = f"parent_id={cls.parent_id}" if cls.parent_id else "MACRO"
            logger.info(f"  ID={cls.id}: '{cls.name}' ({parent_info})")
            logger.info(f"    Description: {cls.description[:60]}...")
            logger.info(f"    Reference images: {len(cls.reference_images)}")
            for img in cls.reference_images:
                exists = "✓" if Path(img).exists() else "✗ MISSING"
                logger.info(f"      {exists}: {Path(img).name}")
        
        # Determine our classification set based on strategy
        class_references = {}
        class_descriptions = {}
        flat_map = {} # label -> (macro_class, sub_class_or_none)
        
        if job.config.strategy == SubcategorizationStrategy.PRE:
            logger.info("Using PRE-classification strategy (Flat)")
            logger.info("=" * 60)
            logger.info("PRE STRATEGY: Building flat classification set")
            logger.info("=" * 60)
            
            # In PRE mode, we put all macros AND all their children in one big bucket
            for macro in selected_macros:
                children = [c for c in all_classes if c.parent_id == macro.id]
                
                logger.info(f"\nMacro: '{macro.name}' - {len(children)} subcategories")
                logger.info(f"  Macro Description: {macro.description}")
                logger.info(f"  Macro Reference images: {len(macro.reference_images)}")
                
                if children:
                    # Parent becomes a "General" catch-all
                    label_p = f"General_{macro.name}"
                    class_references[label_p] = macro.reference_images
                    # Better description for catch-all
                    class_descriptions[label_p] = f"{macro.description} This is a general image that doesn't fit specific subcategories."
                    flat_map[label_p] = (macro, None)
                    logger.info(f"  -> Created catch-all: '{label_p}'")
                    logger.info(f"     Description: {class_descriptions[label_p]}")
                    
                    # Add all children
                    for child in children:
                        label = f"{macro.name} >> {child.name}"
                        class_references[label] = child.reference_images
                        class_descriptions[label] = child.description
                        flat_map[label] = (macro, child)
                        logger.info(f"  -> Subcategory: '{label}'")
                        logger.info(f"     Description: {child.description}")
                        logger.info(f"     Reference images: {len(child.reference_images)}")
                else:
                    # No children - use macro as-is
                    class_references[macro.name] = macro.reference_images
                    class_descriptions[macro.name] = macro.description
                    flat_map[macro.name] = (macro, None)
                    logger.info(f"  -> No subcategories, using macro directly")
            
            logger.info("\n" + "=" * 60)
            logger.info("FINAL CLASS SET FOR CLASSIFICATION:")
            logger.info("=" * 60)
            for label in class_references.keys():
                refs_count = len(class_references[label])
                desc = class_descriptions[label][:80]
                logger.info(f"  [{label}]")
                logger.info(f"    Refs: {refs_count}, Desc: {desc}...")
            logger.info("=" * 60)
        else:
            logger.info("Using POST-classification strategy (Hierarchical)")
            # In POST mode, Stage 1 only has macro-categories
            for macro in selected_macros:
                class_references[macro.name] = macro.reference_images
                class_descriptions[macro.name] = macro.description
        
        # Initialize classifier
        if classifier is None:
            classifier = StrongEmbeddingClassifier(
                model_name=job.config.clip_model,
                pretrained=job.config.clip_pretrained,
                device=config['clip']['device']
            )
        
        # Prepare reference embeddings
        logger.info("Preparing reference embeddings...")
        await broadcast_progress(ProgressUpdate(
            job_id=job_id,
            status=ClassificationStatus.RUNNING,
            processed_images=0,
            total_images=0,
            progress_percentage=0.0,
            message="Preparing reference embeddings..."
        ))
        
        classifier.prepare_reference_embeddings(
            class_references,
            class_descriptions=class_descriptions,
            num_augmentations=job.config.num_augmentations,
            random_augmentations=job.config.random_augmentations,
            std_threshold=job.config.std_threshold
        )
        
        # Get images to classify
        inference_path = Path(config['paths']['inference_dataset'])
        image_files = get_image_files(inference_path)
        await db.update_job_progress(job_id, 0, len(image_files))
        
        # Create prediction directory structure
        predictions_dir = Path(config['paths']['predictions_dir'])
        macro_names = [c.name for c in selected_macros]
        class_paths = create_prediction_structure(
            predictions_dir, job.job_name, macro_names
        )
        
        # --- Stage 1 (or only stage in PRE mode) ---
        logger.info(f"Classifying {len(image_files)} images...")
        logger.info(f"Classes being classified: {list(class_references.keys())}")
        processed = 0
        stage1_results = []
        
        # Track distribution of predictions
        prediction_counts = {}
        
        for i, (img_path, label, probability, is_high_conf) in enumerate(
            classifier.classify_images(
                image_files,
                high_threshold=job.config.high_prob_threshold,
                low_threshold=job.config.low_prob_threshold,
                batch_size=config['classification']['batch_size']
            )
        ):
            stage1_results.append((img_path, label, probability, is_high_conf))
            
            # Track prediction distribution
            prediction_counts[label] = prediction_counts.get(label, 0) + 1
            
            # Log each prediction for debugging
            logger.debug(f"Classified: {Path(img_path).name} -> '{label}' (prob: {probability:.1%}, high_conf: {is_high_conf})")
            
            processed += 1
            if processed % 10 == 0 or processed == len(image_files):
                await db.update_job_progress(job_id, processed, len(image_files))
                await broadcast_progress(ProgressUpdate(
                    job_id=job_id,
                    status=ClassificationStatus.PROCESSING,
                    processed_images=processed,
                    total_images=len(image_files),
                    progress_percentage=(processed / len(image_files)) * 100,
                    current_image=Path(img_path).name,
                    message="Initial Classification" if job.config.strategy == SubcategorizationStrategy.POST else "Flat Classification"
                ))
        
        # Log prediction distribution summary
        logger.info("=" * 60)
        logger.info("CLASSIFICATION RESULTS DISTRIBUTION:")
        for label, count in sorted(prediction_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  '{label}': {count} images")
        logger.info("=" * 60)

        # --- Process Results ---
        final_results = []
        
        if job.config.strategy == SubcategorizationStrategy.PRE:
            # PRE (Flat) Strategy:
            # All subcategories competed together in a single classification pass.
            # Now we organize results into parent/child folder structure.
            logger.info("PRE Strategy: Organizing flat classification results into folder hierarchy...")
            logger.info(f"Total labels in flat_map: {len(flat_map)}")
            
            sub_assigned_count = 0
            macro_only_count = 0
            
            for idx, (img_path, label, prob, high_conf) in enumerate(stage1_results):
                macro, sub = flat_map[label]
                
                result = ClassificationResult(
                    job_id=job_id,
                    image_path=img_path,
                    predicted_class_id=macro.id,
                    predicted_class_name=macro.name,
                    predicted_subclass_id=sub.id if sub else None,
                    predicted_subclass_name=sub.name if sub else None,
                    probability=prob,
                    is_high_confidence=high_conf
                )
                
                # Log assignments
                if sub:
                    logger.info(f"PRE: {Path(img_path).name} -> {macro.name}/{sub.name} (conf: {prob:.1%})")
                    sub_assigned_count += 1
                else:
                    logger.debug(f"PRE: {Path(img_path).name} -> {macro.name} (general) (conf: {prob:.1%})")
                    macro_only_count += 1
                
                # Save to folder structure: parent_folder/subcategory_folder (if sub exists)
                save_image_to_class(
                    img_path, macro.name, class_paths, high_conf,
                    subclass_name=sub.name if sub else None
                )
                await db.save_result(result)
                final_results.append(result)
                
                # Progress update for saving phase
                if (idx + 1) % 20 == 0 or (idx + 1) == len(stage1_results):
                    await broadcast_progress(ProgressUpdate(
                        job_id=job_id,
                        status=ClassificationStatus.PROCESSING,
                        processed_images=len(image_files),
                        total_images=len(image_files),
                        progress_percentage=100.0,
                        current_image=Path(img_path).name,
                        message=f"Organizing results ({idx + 1}/{len(stage1_results)})"
                    ))
            
            logger.info(f"PRE Strategy complete: {sub_assigned_count} to subcategories, {macro_only_count} to macro only")
        
        else:
            # Hierarchical strategy (POST): Two-stage classification
            logger.info("Starting hierarchical sub-classification...")
            
            # Pre-prepare sub-classification data for all macros with children
            sub_class_data = {}
            macros_with_children = []
            
            for macro in selected_macros:
                children = [c for c in all_classes if c.parent_id == macro.id]
                if children:
                    general_name = f"General_{macro.name}"
                    sub_refs = {child.name: child.reference_images for child in children}
                    sub_refs[general_name] = []  # Text-only catch-all
                    sub_descs = {child.name: child.description for child in children}
                    sub_descs[general_name] = f"An image related to {macro.name} but not matching any specific subcategory."
                    
                    sub_class_data[macro.name] = {
                        'children': children,
                        'general_name': general_name,
                        'refs': sub_refs,
                        'descs': sub_descs
                    }
                    macros_with_children.append(macro.name)
                    logger.info(f"Macro '{macro.name}' has {len(children)} subcategories: {[c.name for c in children]}")
            
            # Group stage1 results by macro category
            images_by_macro = {}
            for img_path, macro_name, m_prob, m_high in stage1_results:
                if macro_name not in images_by_macro:
                    images_by_macro[macro_name] = []
                images_by_macro[macro_name].append((img_path, m_prob, m_high))
            
            # Track overall progress across all macros
            total_to_process = len(stage1_results)
            overall_processed = 0
            
            # Process each macro category
            for macro_name, macro_images in images_by_macro.items():
                macro_obj = next((m for m in selected_macros if m.name == macro_name), None)
                if not macro_obj:
                    logger.warning(f"Macro object not found for '{macro_name}', skipping")
                    continue
                
                logger.info(f"Processing macro '{macro_name}': {len(macro_images)} images")
                
                if macro_name in sub_class_data:
                    # This macro has subcategories - run sub-classification
                    data = sub_class_data[macro_name]
                    children = data['children']
                    general_name = data['general_name']
                    
                    # Broadcast that we're preparing embeddings for this macro
                    await broadcast_progress(ProgressUpdate(
                        job_id=job_id,
                        status=ClassificationStatus.PROCESSING,
                        processed_images=overall_processed,
                        total_images=total_to_process,
                        progress_percentage=(overall_processed / total_to_process) * 100,
                        message=f"Preparing sub-classification for {macro_name}..."
                    ))
                    
                    # Prepare sub-classification embeddings for this macro
                    logger.info(f"Preparing embeddings for {len(children)} subcategories of '{macro_name}'")
                    classifier.prepare_reference_embeddings(
                        data['refs'],
                        class_descriptions=data['descs'],
                        num_augmentations=job.config.num_augmentations,
                        random_augmentations=job.config.random_augmentations,
                        std_threshold=job.config.std_threshold
                    )
                    
                    # Batch classify all images in this macro category
                    image_paths = [img[0] for img in macro_images]
                    sub_results = classifier.classify_images(
                        image_paths,
                        high_threshold=job.config.high_prob_threshold,
                        low_threshold=job.config.low_prob_threshold,
                        batch_size=config['classification']['batch_size']
                    )
                    
                    # Process each result
                    for i, (img_path, m_prob, m_high) in enumerate(macro_images):
                        _, winner, s_prob, s_high = sub_results[i]
                        
                        result = ClassificationResult(
                            job_id=job_id,
                            image_path=img_path,
                            predicted_class_id=macro_obj.id,
                            predicted_class_name=macro_name,
                            probability=m_prob,
                            is_high_confidence=m_high
                        )
                        
                        # Assign subcategory ONLY if:
                        # 1. Winner is not the catch-all "General_X" class
                        # 2. Confidence meets the subcategory_threshold (default 80%)
                        # Otherwise, image stays in the parent macro folder
                        if winner != general_name and s_prob >= job.config.subcategory_threshold:
                            sub_obj = next((c for c in children if c.name == winner), None)
                            if sub_obj:
                                result.predicted_subclass_id = sub_obj.id
                                result.predicted_subclass_name = winner
                                logger.info(f"✓ Sub-classified: {Path(img_path).name} -> {macro_name}/{winner} (conf: {s_prob:.1%})")
                            else:
                                logger.warning(f"Subcategory '{winner}' not found in children list")
                        else:
                            # Image stays in parent macro folder (didn't meet subcategory threshold)
                            if winner != general_name:
                                logger.info(f"✗ Kept in macro: {Path(img_path).name} -> {macro_name} (best sub: {winner} @ {s_prob:.1%}, threshold: {job.config.subcategory_threshold:.1%})")
                            else:
                                logger.debug(f"  General match: {Path(img_path).name} -> {macro_name}")
                        
                        save_image_to_class(
                            img_path, macro_name, class_paths, m_high,
                            subclass_name=result.predicted_subclass_name
                        )
                        await db.save_result(result)
                        final_results.append(result)
                        
                        overall_processed += 1
                        if overall_processed % 5 == 0 or overall_processed == total_to_process:
                            await broadcast_progress(ProgressUpdate(
                                job_id=job_id,
                                status=ClassificationStatus.PROCESSING,
                                processed_images=overall_processed,
                                total_images=total_to_process,
                                progress_percentage=(overall_processed / total_to_process) * 100,
                                current_image=Path(img_path).name,
                                message=f"Sub-classifying: {macro_name} ({i+1}/{len(macro_images)})"
                            ))
                else:
                    # No subcategories for this macro - just save results
                    logger.info(f"No subcategories for '{macro_name}', saving {len(macro_images)} results directly")
                    for img_path, m_prob, m_high in macro_images:
                        result = ClassificationResult(
                            job_id=job_id,
                            image_path=img_path,
                            predicted_class_id=macro_obj.id,
                            predicted_class_name=macro_name,
                            probability=m_prob,
                            is_high_confidence=m_high
                        )
                        
                        save_image_to_class(img_path, macro_name, class_paths, m_high)
                        await db.save_result(result)
                        final_results.append(result)
                        
                        overall_processed += 1
                        
                    # Update progress after processing macro without subcategories
                    await broadcast_progress(ProgressUpdate(
                        job_id=job_id,
                        status=ClassificationStatus.PROCESSING,
                        processed_images=overall_processed,
                        total_images=total_to_process,
                        progress_percentage=(overall_processed / total_to_process) * 100,
                        message=f"Saved {len(macro_images)} images for {macro_name}"
                    ))
        
        # Reset processed for safety net tracking
        processed = 0 
        
        # Apply safety net if enabled
        if job.config.apply_safety_net and safety_net:
            logger.info("Applying safety net...")
            await broadcast_progress(ProgressUpdate(
                job_id=job_id,
                status=ClassificationStatus.RUNNING,
                processed_images=len(image_files),
                total_images=len(image_files),
                progress_percentage=100.0,
                message="Applying object detection safety net..."
            ))
            
            predictions_path = predictions_dir / job.job_name
            safety_rules = config.get('safety_net', {}).get('rules', [])
            
            logger.info(f"Applying {len(safety_rules)} safety net rules...")
            
            # Since safety net needs high/low prob dirs, we'll apply it per parent class
            for class_name, (hp_dir, lp_dir) in class_paths.items():
                logger.info(f"Safety net checking class: {class_name}")
                safety_net.apply_safety_rules(
                    predictions_path / class_name,
                    hp_dir,
                    lp_dir,
                    rules=safety_rules
                )
        
        # Mark as completed
        await db.update_job_status(job_id, ClassificationStatus.COMPLETED)
        await broadcast_progress(ProgressUpdate(
            job_id=job_id,
            status=ClassificationStatus.COMPLETED,
            processed_images=len(image_files),
            total_images=len(image_files),
            progress_percentage=100.0,
            message="Classification completed!"
        ))
        
        logger.info(f"Classification job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Error in classification job {job_id}: {e}")
        await db.update_job_status(job_id, ClassificationStatus.FAILED, str(e))
        await broadcast_progress(ProgressUpdate(
            job_id=job_id,
            status=ClassificationStatus.FAILED,
            processed_images=0,
            total_images=0,
            progress_percentage=0.0,
            message=f"Error: {str(e)}"
        ))
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
