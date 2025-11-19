#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InstanSeg Segmentation Node for nuclei segmentation + embedding generation
"""
import argparse
import os
import sys
import time
import json
import zarr
import uvicorn
import requests
import platform
import numpy as np
from sse_starlette.sse import EventSourceResponse
import asyncio

import multiprocessing
import multiprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path

from instanseg_nuc_seg import SlideSegmentation
# Import embedding if available (reuse StarDist's embedding module)
try:
    from nuc_embedding import NucleiEmbedding
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    print("Warning: nuc_embedding not available, embedding generation will be skipped")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []
progress_value = 0  # Global variable to track progress
progress_complete = False  # New flag to indicate completion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # === segmentation + embedding parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--instanseg_model', default='brightfield_nuclei', type=str,
                        help='InstanSeg model name (e.g., brightfield_nuclei, fluorescence_nuclei)')
    parser.add_argument('--image_reader', default='tiffslide', type=str,
                        help='Image reader for InstanSeg')
    # New arguments for downsampling and bounding box
    parser.add_argument('--target_mpp', default=None, type=float, help='Target microns per pixel for processing')
    parser.add_argument('--bbox', default=None, type=str, help='Bounding box for segmentation in format "x,y,width,height"')
    parser.add_argument('--polygon_points', default=None, type=json.loads, help='Polygon points for segmentation in JSON string format "[[x1,y1],[x2,y2],...]".')
    
    # CPU control parameter
    parser.add_argument('--max_workers', type=int, default=15, help='Maximum number of CPU workers for processing (default: 15)')
    # Embedding data loading workers
    parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers for embedding (default: auto, set to 0 for single-threaded)')

    return parser.parse_args()


def run_segmentation(args):
    """
    Combined "Segmentation + Embedding" logic in one node.
    1) if already have segmentation => skip InstanSeg
    2) or run InstanSeg to get segmentation
    3) according to segmentation, generate embedding
    4) write segmentation + embedding to workflow_data.zarr
    """
    global progress_complete

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        start_time = time.time()

        # Step A: check if already have segmentation
        ALREADY_HAVE_SEG = False
        centroids = None
        contours = None
        probability = None

        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if NODE_NAME in zf:
                try:
                    centroids = zf[f"{NODE_NAME}/centroids"][()]
                    # Attempt to load contours and probability, but don't fail if not present initially
                    if f"{NODE_NAME}/contours" in zf:
                        contours = zf[f"{NODE_NAME}/contours"][()]
                    if f"{NODE_NAME}/probability" in zf:
                        probability = zf[f"{NODE_NAME}/probability"][()]

                    # Check if essential data (centroids) is valid
                    if centroids is not None and centroids.size > 0:
                        ALREADY_HAVE_SEG = True
                        print("Using existing nuclei segmentation => skip InstanSeg.")
                        result["message"] = "Using existing nuclei segmentation"
                        result["nuclei_count"] = len(centroids)
                    else:
                        print("Warning: Existing centroids are missing or empty. Will re-run InstanSeg.")
                        ALREADY_HAVE_SEG = False
                        centroids = None
                        contours = None
                        probability = None
                except KeyError as e:
                    print(f"Warning: Existing segmentation data missing key {e}. Will re-run InstanSeg.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
                except Exception as e:
                    print(f"Warning: Error reading existing segmentation data: {e}. Will re-run InstanSeg.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
            else:
                print(f"Group '{NODE_NAME}' not found in Zarr store. Will run InstanSeg.")
                ALREADY_HAVE_SEG = False

        # Step B: if not have segmentation => run InstanSeg
        if not ALREADY_HAVE_SEG:
            print(f"Working on {args.slidepath} with InstanSeg model={args.instanseg_model}")
            
            # Initialize InstanSeg segmentation
            # Use 1024 tile size for better MPS performance (4x faster than 2048)
            ss = SlideSegmentation(
                args,
                tile_size=1024,  # Optimized for MPS: 1024 is 4x faster than 2048
                overlap=128,     # Reduced overlap for smaller tiles
                model_name=getattr(args, 'instanseg_model', 'brightfield_nuclei'),
                image_reader=getattr(args, 'image_reader', 'tiffslide'),
                verbosity=1,
                progress_callback=lambda x: update_progress(x, "segmentation")
            )
            ss.run_WSI_segmentation()
            
            # Retrieve results from ss object, with checks
            if hasattr(ss, 'final_points') and ss.final_points is not None:
                centroids = ss.final_points.astype(np.int32)
                print(f"[SEG LOG] ss.final_points (centroids) generated. Shape: {centroids.shape}, Dtype: {centroids.dtype}")
            else:
                print("[SEG LOG] ss.final_points (centroids) is None or not generated. Setting to empty.")
                centroids = np.array([]).reshape(0, 2).astype(np.int32)

            if hasattr(ss, 'final_coord') and ss.final_coord is not None:
                contours = ss.final_coord.astype(np.int32)
                print(f"[SEG LOG] ss.final_coord (contours) generated. Shape: {contours.shape}, Dtype: {contours.dtype}")
            else:
                print("[SEG LOG] ss.final_coord (contours) is None or not generated. Setting to None.")
                contours = None

            if hasattr(ss, 'prob_all') and ss.prob_all is not None:
                probability = ss.prob_all.astype(np.float32)
                print(f"[SEG LOG] ss.prob_all (probability) generated. Shape: {probability.shape}, Dtype: {probability.dtype}")
            else:
                print("[SEG LOG] ss.prob_all (probability) is None or not generated. Setting to empty.")
                probability = np.array([]).astype(np.float32)

            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"

        # Step C: generate embedding if not cached; write directly to Zarr
        if centroids is not None and len(centroids) > 0:
            have_cached_embedding = False
            zf = zarr.open_group(ZARR_PATH, mode='a')
            node_grp_path = f"{NODE_NAME}"
            if NODE_NAME in zf and 'embedding' in zf[NODE_NAME]:
                try:
                    existing_len = zf[NODE_NAME]['embedding'].shape[0]
                    if existing_len == len(centroids):
                        have_cached_embedding = True
                        print("found existing embeddings in store => skip embedding calculation")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding and HAS_EMBEDDING:
                print("no cached embeddings => generate new embeddings directly into Zarr")
                # Load contours for bounding box extraction
                contours_for_embedding = None
                if NODE_NAME in zf and 'contours' in zf[NODE_NAME]:
                    try:
                        contours_for_embedding = zf[f"{NODE_NAME}/contours"][()]
                        print(f"Loaded contours for embedding: shape {contours_for_embedding.shape}")
                    except Exception as e:
                        print(f"Warning: Could not load contours for embedding: {e}")
                ne = NucleiEmbedding(args, centroids, contours=contours_for_embedding, progress_callback=lambda x: update_progress(x, "embedding"))
                ne.generate_embeddings(zarr_path=ZARR_PATH, dataset_path=f"{NODE_NAME}/embedding", num_workers=0)
        elif centroids is not None and len(centroids) == 0:
            print("[EMBED LOG] No centroids detected from segmentation, skipping embedding generation.")
        else:
            print("[EMBED LOG] Centroids are None, skipping embedding generation.")

        # Step D: write segmentation (embedding already written if generated)
        if centroids is not None:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            # Do NOT delete the whole group, as it would remove previously written 'embedding'.
            # Instead, create/require the group and overwrite only specific datasets.
            node_grp = zf.require_group(NODE_NAME)

            print(f"[ZARR WRITE] Writing centroids. Shape: {centroids.shape if centroids is not None else 'None'}")
            if 'centroids' in node_grp:
                del node_grp['centroids']
            node_grp.create_dataset('centroids', data=centroids)
            
            if contours is not None:
                print(f"[ZARR WRITE] Writing contours. Shape: {contours.shape}")
                if 'contours' in node_grp:
                    del node_grp['contours']
                node_grp.create_dataset('contours', data=contours)
            else:
                print("[ZARR WRITE] Contours are None, not writing.")
            
            if probability is not None:
                print(f"[ZARR WRITE] Writing probability. Shape: {probability.shape}")
                if 'probability' in node_grp:
                    del node_grp['probability']
                node_grp.create_dataset('probability', data=probability)
            else:
                print("[ZARR WRITE] Probability is None, not writing.")

            # Embedding has been written directly by NucleiEmbedding if needed

            time.sleep(0.5)
        else:
            print("[ZARR WRITE] Centroids are None after segmentation step, nothing to write for this node.")

        progress_complete = True
        update_progress(100, "embedding")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "nuclei_count": 0}


@app.get("/status")
def get_status():
    return {"status": "instanseg_node with embedding running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[InstanSegNode] /init => inited model/resources (with embedding)")
        return {"status": "ok", "message": "InstanSegNode init done"}
    else:
        print("[InstanSegNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS
    NODE_NAME = data.get("node_name", "InstanSegNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    print(f"[InstanSegNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[InstanSegNode] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            instanseg_model="brightfield_nuclei",
            image_reader="tiffslide",
            # Initialize ROI/scaling-related fields to avoid stale carry-over
            target_mpp=None,
            bbox=None,
            polygon_points=None,
            # Initialize z-stack related fields
            z_layer_for_segmentation=None,
            is_zstack=False,
            num_z_layers=1,
            # Initialize num_workers (can be set via userData)
            num_workers=None,
        )
    else:
        # Reset ROI/scaling-related fields on every /read to prevent using values from a previous run
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None
        # Reset z-stack fields (will be auto-detected during segmentation)
        ARGS.z_layer_for_segmentation = None
        ARGS.is_zstack = False
        ARGS.num_z_layers = 1
        # Reset num_workers (will be set from userData if provided)
        ARGS.num_workers = None

    zf = zarr.open_group(ZARR_PATH, mode='r')
    user_data_path = f"{NODE_NAME}/userData"
    if user_data_path in zf:
        for k in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][k][()]
            raw_str = raw_bytes.decode("utf-8")
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            print(f"[InstanSegNode] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "instanseg_model":
                ARGS.instanseg_model = val_json
            elif k == "image_reader":
                ARGS.image_reader = val_json
            elif k == "target_mpp":
                try:
                    ARGS.target_mpp = float(val_json)
                except ValueError:
                    print(f"Warning: Could not parse target_mpp value '{val_json}' as float.")
                    ARGS.target_mpp = None
            elif k == "bbox":
                if isinstance(val_json, str) and len(val_json.split(',')) == 4:
                    ARGS.bbox = val_json
                else:
                    print(f"Warning: bbox value '{val_json}' is not in 'x,y,width,height' format.")
                    ARGS.bbox = None
            elif k == "polygon_points":
                if isinstance(val_json, list) and all(isinstance(p, list) and len(p) == 2 for p in val_json):
                    ARGS.polygon_points = val_json
                else:
                    print(f"Warning: polygon_points value '{val_json}' is not in the expected [[x1,y1],[x2,y2],...] format.")
                    ARGS.polygon_points = None
            elif k == "num_workers":
                try:
                    ARGS.num_workers = int(val_json)
                    print(f"  → num_workers set to {ARGS.num_workers} via userData")
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse num_workers value '{val_json}' as int.")
                    ARGS.num_workers = None

    return {"status": "ok", "message": "InstanSegNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[InstanSegNode] no path => skip.")
        out_val = {
            "status": "ok",
            "message": "no path, skipping.",
            "nuclei_count": 0
        }
    else:
        print(f"[InstanSegNode] /execute => run_segmentation with slidepath={ARGS.slidepath}")
        out_val = run_segmentation(ARGS)

    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{NODE_NAME}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out_val, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_dataset(node_out_path, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)

    return {"status": "ok", "output": out_val}


def update_progress(value, phase="segmentation"):
    """
    Update progress with phase-specific scaling
    - segmentation: 0-50
    - embedding: 50-100
    """
    global progress_value
    
    if phase == "segmentation":
        # Scale segmentation progress from 0-100 to 0-50
        progress_value = int(value * 0.5)
    elif phase == "embedding":
        # Scale embedding progress from 0-100 to 50-100
        progress_value = 50 + int(value * 0.5)
    else:
        # Default behavior for backward compatibility
        progress_value = value


@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_complete
        last_value = -1
        progress_value = 0  # Reset progress to 0 for each new connection
        progress_complete = False  # Reset completion flag
        
        while True:
            # Check if progress changed or if it's the final 100% update
            if progress_value != last_value or (progress_value == 100 and progress_complete):
                if last_value > progress_value:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value

                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if progress_value == 100 and progress_complete:
                    print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break

            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 and completion flag after sending the final update
        progress_value = 0
        progress_complete = False
        print("Progress reset to 0.")

    return EventSourceResponse(event_generator())


def main():
    # Add this line to support multiprocessing in PyInstaller packaged executables
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting InstanSegNode at port={args.port}")

    try:
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(1)  # wait uvicorn start

        # register to manager
        this_file_path = str(Path(__file__).resolve())
        create_payload = {
            "service_name": args.name,
            "file_path": this_file_path,
            "port": args.port
        }
        url_create = f"{args.manager_host}/api/tasks/v1/create_node"

        try:
            resp = requests.post(url_create, json=create_payload, timeout=10)
            resp.raise_for_status()
            print(f"[{args.name}] create_node success => {resp.json()}")
        except Exception as e:
            print(f"[{args.name}] create_node request failed: {e}")
            print("keep running...")

        print(f"[{args.name}] Serving at port={args.port}, Press Ctrl+C to exit.")
        t.join()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    main()

