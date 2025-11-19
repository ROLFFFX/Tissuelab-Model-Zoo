#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InstanSeg Segmentation Main Script
Created for TissueLab integration
"""

import argparse
import numpy as np
import time
import os
import platform
import zarr
import json

from instanseg_nuc_seg import SlideSegmentation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', default='', type=str, help='Path to slide image')
    parser.add_argument('--read_image_method', default='tiffslide', type=str, 
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--instanseg_model', default='brightfield_nuclei', type=str,
                        help='InstanSeg model name (e.g., brightfield_nuclei, fluorescence_nuclei)')
    parser.add_argument('--image_reader', default='tiffslide', type=str,
                        help='Image reader for InstanSeg')
    parser.add_argument('--target_mpp', default=None, type=float, 
                        help='Target microns per pixel for processing')
    parser.add_argument('--bbox', default=None, type=str, 
                        help='Bounding box for segmentation in format "x,y,width,height"')
    parser.add_argument('--polygon_points', default=None, type=json.loads, 
                        help='Polygon points for segmentation in JSON string format "[[x1,y1],[x2,y2],...]".')
    parser.add_argument('--debug', default=False, action='store_true', 
                        help='Enable debug mode to save mask images')
    return parser.parse_args()


def main(args):
    try:
        result = {
            "status": "success",
            "message": "",
            "nuclei_count": 0
        }
        
        start_time = time.time()
        
        # Zarr store is typically a directory, change extension from .h5 to .zarr
        zarr_path = args.slidepath + ".zarr"
        
        # Check if zarr store exists and has segmentation
        ALREADY_HAVE_NUCLEI_SEGMENTATION = False
        centroids = None
        contours = None
        probability = None
        
        if os.path.exists(zarr_path):
            zf = zarr.open_group(zarr_path, mode='r')
            if 'InstanSegNode' in zf:
                try:
                    centroids = zf['InstanSegNode']['centroids'][()].copy()
                    if 'contours' in zf['InstanSegNode']:
                        contours = zf['InstanSegNode']['contours'][()].copy()
                    if 'probability' in zf['InstanSegNode']:
                        probability = zf['InstanSegNode']['probability'][()].copy()
                    ALREADY_HAVE_NUCLEI_SEGMENTATION = True
                except:
                    print("Error: InstanSegNode group is corrupted.")
                    centroids = None
                    contours = None
                    probability = None
                
                if ALREADY_HAVE_NUCLEI_SEGMENTATION and len(centroids) > 0:
                    result["nuclei_count"] = len(centroids)
                    result["message"] = "Using existing nuclei segmentation."
                    print(f"Using existing segmentation with {len(centroids)} nuclei")
                    return result
        
        print('Working on %s ...' % args.slidepath)
        
        if not ALREADY_HAVE_NUCLEI_SEGMENTATION:
            # Initialize InstanSeg segmentation
            ss = SlideSegmentation(
                args,
                tile_size=4096,
                overlap=256,
                model_name=args.instanseg_model,
                image_reader=args.image_reader,
                verbosity=1
            )
            
            ss.run_WSI_segmentation()
            
            contours = ss.final_coord.astype(np.int32) if ss.final_coord is not None else None
            centroids = ss.final_points.astype(np.int32) if ss.final_points is not None else np.array([]).reshape(0, 2).astype(np.int32)
            probability = ss.prob_all.astype(np.float32) if ss.prob_all is not None else np.array([]).astype(np.float32)
            
            # Save segmentation results
            zf = zarr.open_group(zarr_path, mode='a')
            print("Number of nuclei: %d" % len(centroids))
            
            # Create a group for nuclei segmentation
            nuclei_seg = zf.require_group('InstanSegNode')
            if 'contours' in nuclei_seg:
                del nuclei_seg['contours']
            if 'centroids' in nuclei_seg:
                del nuclei_seg['centroids']
            if 'probability' in nuclei_seg:
                del nuclei_seg['probability']
            
            if contours is not None:
                nuclei_seg.create_dataset('contours', data=contours)
            nuclei_seg.create_dataset('centroids', data=centroids)
            if probability is not None and len(probability) > 0:
                nuclei_seg.create_dataset('probability', data=probability)
            
            # Generate embeddings if available
            try:
                from nuc_embedding import NucleiEmbedding
                print("Generating nuclei embeddings...")
                ne = NucleiEmbedding(args, centroids, contours=contours)
                ne.generate_embeddings(zarr_path=zarr_path, dataset_path='InstanSegNode/embedding')
            except ImportError:
                print("Warning: nuc_embedding not available, skipping embedding generation")
            except Exception as e:
                print(f"Warning: Error generating embeddings: {e}")
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        result["message"] = "Segmentation completed successfully"
        result["nuclei_count"] = len(centroids) if centroids is not None else 0
        return result
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "nuclei_count": 0
        }


if __name__ == '__main__':
    args = parse_args()
    print("Currently working on " + list(platform.uname())[1] + " Machine")
    result = main(args)
    print(f"Result: {result}")

