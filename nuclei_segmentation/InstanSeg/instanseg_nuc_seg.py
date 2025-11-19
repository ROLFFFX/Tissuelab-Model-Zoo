# -*- coding: utf-8 -*-
"""
InstanSeg WSI Segmentation Module
Handles whole slide image segmentation using InstanSeg
"""

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from PIL import Image, ImageOps
import cv2
from skimage import morphology
from skimage.measure import regionprops
from tqdm import tqdm
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import tiffslide
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from collections import defaultdict
import json

opj = os.path.join


class SlideSegmentation:
    """InstanSeg-based WSI segmentation class"""
    
    def __init__(self,
                 args,
                 tile_size=2048,
                 overlap=224,
                 model_name="brightfield_nuclei",
                 image_reader="tiffslide",
                 verbosity=1,
                 progress_callback=None):
        """
        Initialize InstanSeg segmentation
        
        Args:
            args: Arguments namespace containing slidepath, read_image_method, etc.
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles
            model_name: InstanSeg model name (e.g., "brightfield_nuclei", "fluorescence_nuclei")
            image_reader: Image reader method ("tiffslide", "openslide", etc.)
            verbosity: Verbosity level for InstanSeg
            progress_callback: Callback function for progress updates
        """
        super(SlideSegmentation, self).__init__()
        
        self.args = args
        self.tile_size = tile_size
        self.overlap = overlap
        self.model_name = model_name
        self.image_reader = image_reader
        self.verbosity = verbosity
        self.progress_callback = progress_callback
        
        # Initialize InstanSeg model with device selection
        try:
            from instanseg import InstanSeg
            import torch
            import os
            
            # Check device availability and set environment for InstanSeg
            # InstanSeg should automatically detect and use the best available device
            if torch.cuda.is_available():
                device_str = "cuda"
                print(f"InstanSeg will use CUDA device")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_str = "mps"
                print(f"InstanSeg will use MPS device (Apple Silicon GPU)")
                # Set environment variable to prefer MPS (if InstanSeg supports it)
                # Also ensure PyTorch uses MPS by setting default device if available
                try:
                    # PyTorch 2.0+ supports set_default_device
                    if hasattr(torch, 'set_default_device'):
                        torch.set_default_device("mps")
                        print("Set PyTorch default device to MPS")
                except Exception as e:
                    print(f"Note: Could not set default device to MPS: {e}")
                    print("InstanSeg should still detect MPS automatically")
            else:
                device_str = "cpu"
                print(f"InstanSeg will use CPU device")
            
            # Initialize InstanSeg - it should automatically detect and use MPS if available
            self.instanseg = InstanSeg(model_name, image_reader=image_reader, verbosity=verbosity, device=device_str)
            print(f"InstanSeg model '{model_name}' initialized successfully")
            
            # Verify model is on correct device
            if hasattr(self.instanseg, 'instanseg') and hasattr(self.instanseg.instanseg, 'parameters'):
                model_params = list(self.instanseg.instanseg.parameters())
                if len(model_params) > 0:
                    actual_model_device = str(model_params[0].device)
                    print(f"InstanSeg model device: {actual_model_device}")
                    if device_str == "mps" and actual_model_device != "mps":
                        print(f"[WARNING] Model is on {actual_model_device} but should be on mps!")
                
                # CRITICAL: Check if model is TorchScript (ScriptModule)
                import torch
                if isinstance(self.instanseg.instanseg, torch.jit.ScriptModule):
                    print(f"[CRITICAL] Model is TorchScript (ScriptModule)!")
                    print(f"[CRITICAL] TorchScript has poor MPS support - this is why inference is slow!")
                    print(f"[CRITICAL] TorchScript may fallback to CPU during execution even if parameters are on MPS")
                    print(f"[CRITICAL] Recommendation: Use smaller tiles (512x512) or multi-process parallel processing")
                    print(f"[CRITICAL] See TORCHSCRIPT_MPS_ISSUE.md for details and solutions")
            
            # Force model to use MPS if available (InstanSeg might not automatically use it)
            import torch
            self.device = None
            if torch.backends.mps.is_available():
                try:
                    # Try to move model to MPS device
                    if hasattr(self.instanseg, 'model') and self.instanseg.model is not None:
                        # Check current device
                        try:
                            model_params = list(self.instanseg.model.parameters())
                            if len(model_params) > 0:
                                current_device = model_params[0].device
                                print(f"InstanSeg model current device: {current_device}")
                                
                                # Move to MPS if not already there
                                if str(current_device) != 'mps':
                                    print(f"Moving InstanSeg model to MPS device...")
                                    self.instanseg.model = self.instanseg.model.to('mps')
                                    # Verify
                                    new_params = list(self.instanseg.model.parameters())
                                    if len(new_params) > 0:
                                        new_device = new_params[0].device
                                        print(f"InstanSeg model moved to device: {new_device}")
                                        self.device = torch.device('mps')
                        except Exception as e:
                            print(f"Warning: Could not check/move model device: {e}")
                    
                    # Also try to set device in InstanSeg's internal state if possible
                    if hasattr(self.instanseg, 'device'):
                        self.instanseg.device = torch.device('mps')
                        self.device = torch.device('mps')
                        print(f"Set InstanSeg.device to MPS")
                    elif hasattr(self.instanseg, '_device'):
                        self.instanseg._device = torch.device('mps')
                        self.device = torch.device('mps')
                        print(f"Set InstanSeg._device to MPS")
                    
                    # Try to set device via eval_small_image if it has device parameter
                    # Check InstanSeg's eval_small_image signature
                    import inspect
                    if hasattr(self.instanseg, 'eval_small_image'):
                        sig = inspect.signature(self.instanseg.eval_small_image)
                        if 'device' in sig.parameters:
                            print(f"InstanSeg.eval_small_image supports device parameter")
                            self.device = torch.device('mps')
                except Exception as e:
                    print(f"Warning: Could not force MPS device: {e}")
                    import traceback
                    import sys
                    try:
                        print(traceback.format_exc())
                    except UnicodeEncodeError:
                        # Fallback for Windows GBK encoding issues
                        exc_info = sys.exc_info()
                        tb_str = traceback.format_exception(*exc_info)
                        safe_tb = ''.join(tb_str).encode('ascii', 'replace').decode('ascii')
                        print(safe_tb)
            
            # Verify the device being used
            if hasattr(self.instanseg, 'model') and self.instanseg.model is not None:
                try:
                    model_params = list(self.instanseg.model.parameters())
                    if len(model_params) > 0:
                        actual_device = str(model_params[0].device)
                        print(f"InstanSeg model is using device: {actual_device}")
                        if actual_device != 'mps' and torch.backends.mps.is_available():
                            print(f"WARNING: Model is not on MPS device despite MPS being available!")
                            print(f"WARNING: This may cause slow inference. InstanSeg may not support MPS.")
                            print(f"WARNING: Consider checking InstanSeg version and MPS support.")
                        else:
                            self.device = torch.device(actual_device)
                except Exception as e:
                    print(f"Could not verify model device: {e}")
            
            # Final device assignment
            if self.device is None:
                if torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                elif torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
                print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error initializing InstanSeg: {e}")
            import traceback
            import sys
            try:
                print(traceback.format_exc())
            except UnicodeEncodeError:
                # Fallback for Windows GBK encoding issues
                exc_info = sys.exc_info()
                tb_str = traceback.format_exception(*exc_info)
                # Replace problematic Unicode characters
                safe_tb = ''.join(tb_str).encode('ascii', 'replace').decode('ascii')
                print(safe_tb)
            raise
        
        # Initialize z-stack related attributes
        self.is_zstack = False
        self.num_z_layers = 1
        self.z_layer_for_segmentation = 0
        
        self.read_data()
        self.wsi_mask = self.simple_get_mask()
        
        # Results storage
        self.final_points = None
        self.final_coord = None
        self.prob_all = None
        
        # ROI handling
        self.roi_bbox = None
        self.roi_polygon = None
        self._parse_roi()
    
    def _parse_roi(self):
        """Parse ROI from args (bbox or polygon)"""
        if hasattr(self.args, 'bbox') and self.args.bbox:
            if isinstance(self.args.bbox, str):
                parts = self.args.bbox.split(',')
                if len(parts) == 4:
                    self.roi_bbox = [int(float(p)) for p in parts]
            elif isinstance(self.args.bbox, (list, tuple)) and len(self.args.bbox) == 4:
                self.roi_bbox = [int(float(p)) for p in self.args.bbox]
        
        if hasattr(self.args, 'polygon_points') and self.args.polygon_points:
            if isinstance(self.args.polygon_points, str):
                try:
                    self.roi_polygon = json.loads(self.args.polygon_points)
                except:
                    self.roi_polygon = None
            elif isinstance(self.args.polygon_points, list):
                self.roi_polygon = self.args.polygon_points
    
    def point_in_polygon(self, x, y, polygon):
        """Check if point is inside polygon"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def read_data(self):
        """Read slide data and get dimensions"""
        print("Reading data ...", datetime.now().strftime("%H:%M:%S"))
        
        try:
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
            mpp = float(self.slide.properties.get('tiffslide.mpp-x', 0.25))
            print("Successfully read file using TiffSlide")
        except Exception as e:
            print(f"TiffSlide failed: {str(e)}")
            
            file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.slide = SimpleImageWrapper(self.args.slidepath)
                mpp = 0.25
            elif file_extension in ['dcm']:
                self.slide = DicomImageWrapper(self.args.slidepath)
                mpp = 0.25
            else:
                self.slide = TiffFileWrapper(self.args.slidepath)
                mpp = 0.25
        
        self.original_mpp = mpp
        self.args.original_mpp = mpp
        
        # Handle target_mpp if provided
        if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None:
            self.mpp_resize_factor = self.original_mpp / self.args.target_mpp
            mpp = self.args.target_mpp
        else:
            self.mpp_resize_factor = None
        
        # Calculate magnification
        reference_mpp_1x = 10
        self.args.magnification = reference_mpp_1x / mpp
        print(f"Magnification: {self.args.magnification}")
        
        # Set tile size based on magnification
        # Optimized for MPS: use 1024 for better performance (4x faster than 2048)
        # Tile size is set in __init__, but we can override here if needed
        # Keep the tile_size from __init__ (1024) for optimal MPS performance
        if hasattr(self, 'tile_size') and self.tile_size is not None:
            # Use the tile_size from __init__
            pass
        else:
            # Fallback: set based on magnification
            if self.args.magnification > 80 + 1:
                self.tile_size = 1024
            elif self.args.magnification > 40 + 1:
                self.tile_size = 1024
            elif self.args.magnification > 20 + 1:
                self.tile_size = 1024
            else:
                self.tile_size = 1024
        
        print("-" * 100)
        print(f"Tile size: {self.tile_size}")
        print("-" * 100)
        
        # Get dimensions
        self.dim = self.slide.dimensions
    
    def simple_get_mask(self):
        """Generate tissue mask for WSI"""
        try:
            level = np.min([5, len(self.slide.level_dimensions) - 1])
            dim = list(self.slide.level_dimensions)[level]
            print(f"Using level {level} with dimensions {dim}")
            
            if (dim[0] > 10000) or (dim[1] > 10000):
                print('Thumbnail too large, using higher level')
                level = min(level + 1, len(self.slide.level_dimensions) - 1)
                dim = list(self.slide.level_dimensions)[level]
                print(f"Adjusted to level {level} with dimensions {dim}")
            
            temp_thumb = self.slide.read_region((0, 0), level, dim).convert('RGB')
            gray = np.array(ImageOps.grayscale(temp_thumb))
            
            block_size = 51
            C = 2
            binary_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, C
            )
            
            mask = (binary_mask > 0).astype(np.uint8) * 255
            mask = morphology.remove_small_objects(mask > 0, min_size=16 * 16, connectivity=2)
            mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
            
            struct_element = morphology.disk(16)
            mask = morphology.binary_dilation(mask, struct_element)
            mask = mask.astype(np.uint8) * 255
            
            self.wsi_mask = (mask > 0).astype(np.uint8)
            return self.wsi_mask
        
        except Exception as e:
            print(f"Error in mask generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return a default mask if error occurs
            try:
                dim = list(self.slide.level_dimensions)[0]
                return np.ones(dim[::-1], dtype=np.uint8)
            except:
                # Fallback to a reasonable default size
                return np.ones((1000, 1000), dtype=np.uint8)
    
    def labeled_to_contours(self, labeled_image):
        """Convert labeled image to contours and centroids"""
        perf = {}
        total_start = time.time()
        
        centroids = []
        contours = []
        probabilities = []
        
        # Convert to numpy array if needed (handle torch tensors and other types)
        convert_start = time.time()
        original_type = type(labeled_image)
        original_shape_before = getattr(labeled_image, 'shape', None)
        
        # Check if it's on GPU/MPS and needs transfer (this can be slow!)
        if hasattr(labeled_image, 'device'):
            perf['input_device'] = str(labeled_image.device)
            if str(labeled_image.device) != 'cpu':
                # GPU/MPS to CPU transfer - this can be a bottleneck!
                transfer_start = time.time()
                if hasattr(labeled_image, 'cpu'):
                    labeled_image = labeled_image.cpu()
                perf['gpu_to_cpu_transfer'] = time.time() - transfer_start
                if self.verbosity > 0:
                    print(f"[PERF] GPU->CPU transfer took {perf['gpu_to_cpu_transfer']:.3f}s")
        
        if hasattr(labeled_image, 'cpu'):
            # It's a torch tensor (might already be on CPU)
            numpy_start = time.time()
            labeled_image = labeled_image.numpy()
            perf['tensor_to_numpy'] = time.time() - numpy_start
            if self.verbosity > 0:
                print(f"[DEBUG] Converted torch tensor to numpy: {original_shape_before} -> {labeled_image.shape}")
        elif hasattr(labeled_image, 'numpy'):
            # It's a tensorflow tensor or similar
            labeled_image = labeled_image.numpy()
        elif not isinstance(labeled_image, np.ndarray):
            labeled_image = np.array(labeled_image)
        perf['convert_to_numpy'] = time.time() - convert_start
        
        original_shape = labeled_image.shape
        if self.verbosity > 0:
            print(f"[DEBUG] After conversion, shape: {original_shape}, dtype: {labeled_image.dtype}")
        
        # Ensure it's a 2D array for regionprops
        # Handle different possible shapes:
        # - (H, W) - correct format
        # - (1, H, W) - batch dimension, squeeze it
        # - (H, W, 1) - channel dimension, squeeze it
        # - (1, H, W, 1) - both batch and channel, squeeze both
        # - (H, W, C) where C > 1 - take first channel or convert to grayscale
        if len(labeled_image.shape) == 2:
            # Already 2D, use as is
            pass
        elif len(labeled_image.shape) == 3:
            if labeled_image.shape[0] == 1:
                # (1, H, W) - remove batch dimension
                labeled_image = labeled_image.squeeze(0)
            elif labeled_image.shape[2] == 1:
                # (H, W, 1) - remove channel dimension
                labeled_image = labeled_image.squeeze(2)
            elif labeled_image.shape[0] == 1 and labeled_image.shape[2] == 1:
                # (1, H, W, 1) - remove both
                labeled_image = labeled_image.squeeze((0, 2))
            else:
                # (H, W, C) with C > 1 - take first channel or convert to grayscale
                # For labeled images, we typically want the first channel or max across channels
                if labeled_image.shape[2] <= 3:
                    # RGB-like, take first channel
                    labeled_image = labeled_image[:, :, 0]
                else:
                    # Multiple channels, take max (for multi-class labels)
                    labeled_image = np.argmax(labeled_image, axis=2)
        elif len(labeled_image.shape) == 4:
            # Handle both formats:
            # - (B, C, H, W) - PyTorch format (batch, channel, height, width) - e.g., [1, 1, 2048, 2048]
            # - (B, H, W, C) - TensorFlow format (batch, height, width, channel)
            
            # Check if it's PyTorch format (B, C, H, W) - channel dimension is at index 1
            # In PyTorch format, typically C is small (1-3) and H, W are large
            # In TensorFlow format, C is at the end
            if labeled_image.shape[1] <= 3 and labeled_image.shape[1] < labeled_image.shape[2]:
                # Likely (B, C, H, W) format - PyTorch style
                if self.verbosity > 0:
                    print(f"[DEBUG] Detected PyTorch format (B, C, H, W): {labeled_image.shape}")
                labeled_image = labeled_image[0]  # Remove batch dimension -> (C, H, W)
                if self.verbosity > 0:
                    print(f"[DEBUG] After removing batch: {labeled_image.shape}")
                if len(labeled_image.shape) == 3:
                    if labeled_image.shape[0] == 1:
                        # (1, H, W) - remove channel dimension
                        labeled_image = labeled_image.squeeze(0)
                        if self.verbosity > 0:
                            print(f"[DEBUG] After removing channel: {labeled_image.shape}")
                    else:
                        # (C, H, W) with C > 1 - take first channel
                        labeled_image = labeled_image[0]
                        if self.verbosity > 0:
                            print(f"[DEBUG] After taking first channel: {labeled_image.shape}")
            else:
                # Likely (B, H, W, C) format - TensorFlow style
                labeled_image = labeled_image[0]  # Remove batch dimension -> (H, W, C)
                if len(labeled_image.shape) == 3:
                    if labeled_image.shape[2] == 1:
                        # (H, W, 1) - remove channel dimension
                        labeled_image = labeled_image.squeeze(2)
                    else:
                        # (H, W, C) with C > 1 - take first channel
                        labeled_image = labeled_image[:, :, 0]
        else:
            raise ValueError(f"Unsupported labeled_image shape: {original_shape} (type: {original_type}). Expected 2D, 3D, or 4D array. Please check InstanSeg output format.")
        
        # Ensure it's integer type for regionprops
        dtype_convert_start = time.time()
        # InstanSeg returns labeled images where each pixel value is the instance ID
        # However, it might return float values that need to be converted to integers
        if labeled_image.dtype in [np.float32, np.float64]:
            # Check if it's a probability map (values 0-1) or labeled image (integer-like values)
            min_val = float(labeled_image.min())
            max_val = float(labeled_image.max())
            unique_vals = np.unique(labeled_image)
            num_unique = len(unique_vals)
            
            if self.verbosity > 0:
                print(f"[DEBUG] Value range: [{min_val:.3f}, {max_val:.3f}], unique values: {num_unique}")
            
            # If values are in 0-1 range, it might be a probability map or binary mask
            # We need to use connected components to extract individual instances
            if min_val >= 0 and max_val <= 1.1:  # Allow slight overflow
                # Use connected components to extract instances from probability/binary map
                if self.verbosity > 0:
                    if num_unique <= 2:
                        print(f"[DEBUG] Detected binary mask (0/1), using connected components to extract instances")
                    else:
                        print(f"[DEBUG] Detected probability map with {num_unique} unique values, thresholding and using connected components")
                
                # Threshold and use connected components to separate instances
                from scipy import ndimage
                cc_start = time.time()
                binary_mask = (labeled_image > 0.5).astype(np.uint8)
                labeled_image, num_features = ndimage.label(binary_mask)
                labeled_image = labeled_image.astype(np.int32)
                cc_time = time.time() - cc_start
                if self.verbosity > 0:
                    print(f"[DEBUG] Connected components found {num_features} regions (took {cc_time:.3f}s)")
            else:
                # Values > 1, likely a labeled image with instance IDs
                if self.verbosity > 0:
                    print(f"[DEBUG] Detected labeled image (range [{min_val:.1f}, {max_val:.1f}]), converting to int32")
                labeled_image = labeled_image.astype(np.int32)
        elif labeled_image.dtype not in [np.int32, np.int64, np.uint32, np.uint64]:
            labeled_image = labeled_image.astype(np.int32)
        perf['dtype_conversion'] = time.time() - dtype_convert_start
        
        # Final check: must be 2D
        if len(labeled_image.shape) != 2:
            raise ValueError(f"After processing, labeled_image is still not 2D. Shape: {labeled_image.shape}, original: {original_shape}")
        
        if self.verbosity > 0:
            print(f"[DEBUG] Final 2D shape: {labeled_image.shape}, dtype: {labeled_image.dtype}, min={labeled_image.min()}, max={labeled_image.max()}")
        
        # Check if image is empty or all zeros
        if labeled_image.size == 0 or np.all(labeled_image == 0):
            if self.verbosity > 0:
                print("[DEBUG] Labeled image is empty or all zeros, returning empty results")
            return np.array([]).reshape(0, 2), None, np.array([])
        
        rp_start = time.time()
        try:
            props = regionprops(labeled_image)
            perf['regionprops'] = time.time() - rp_start
            if self.verbosity > 0:
                print(f"[DEBUG] Found {len(props)} regions in labeled image (regionprops took {perf['regionprops']:.3f}s)")
        except Exception as e:
            print(f"[ERROR] regionprops failed: {e}")
            print(f"[ERROR] labeled_image shape: {labeled_image.shape}, dtype: {labeled_image.dtype}")
            print(f"[ERROR] labeled_image min: {labeled_image.min()}, max: {labeled_image.max()}")
            raise
        
        extract_start = time.time()
        for prop in props:
            # Get centroid (y, x) -> convert to (x, y)
            centroid_y, centroid_x = prop.centroid
            centroids.append([int(centroid_x), int(centroid_y)])
            
            # Get contour
            coords = prop.coords  # (y, x) format
            # Convert to (x, y) format
            contour = coords[:, [1, 0]].astype(np.int32)
            contours.append(contour)
            
            # Use area as proxy for probability (normalized)
            # InstanSeg doesn't provide explicit probability, so we use a default value
            probabilities.append(1.0)
        perf['extract_props'] = time.time() - extract_start
        
        if len(centroids) == 0:
            return np.array([]).reshape(0, 2), None, np.array([])
        
        array_start = time.time()
        centroids = np.array(centroids, dtype=np.int32)
        probabilities = np.array(probabilities, dtype=np.float32)
        perf['array_conversion'] = time.time() - array_start
        
        # Pad contours to same length (required format)
        pad_start = time.time()
        max_contour_len = max(len(c) for c in contours) if contours else 0
        if max_contour_len > 0:
            padded_contours = []
            for c in contours:
                if len(c) < max_contour_len:
                    # Pad by repeating last point
                    padded = np.vstack([c, np.tile(c[-1:], (max_contour_len - len(c), 1))])
                else:
                    padded = c
                padded_contours.append(padded)
            contours = np.array(padded_contours, dtype=np.int32)
        else:
            contours = None
        perf['pad_contours'] = time.time() - pad_start
        
        perf['total_postprocess'] = time.time() - total_start
        
        # Store performance data for reporting
        if not hasattr(self, '_postprocess_perf'):
            self._postprocess_perf = []
        self._postprocess_perf.append(perf)
        
        return centroids, contours, probabilities
    
    def process_tile(self, x_0, y_0, tile_w, tile_h):
        """Process a single tile"""
        perf_times = {}
        total_start = time.time()
        
        try:
            # Read tile from slide
            read_start = time.time()
            region = self.slide.read_region((x_0, y_0), 0, (tile_w, tile_h))
            img_np = np.array(region)[:, :, :3]  # RGB
            perf_times['read'] = time.time() - read_start
            
            # Resize if needed
            resize_start = time.time()
            if self.mpp_resize_factor is not None:
                new_width = int(np.round(img_np.shape[1] * self.mpp_resize_factor))
                new_height = int(np.round(img_np.shape[0] * self.mpp_resize_factor))
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_np = np.array(img_pil)
                resize_factor = self.mpp_resize_factor
            else:
                resize_factor = 1.0
            perf_times['resize'] = time.time() - resize_start
            
            # Use InstanSeg to segment with detailed profiling
            seg_start = time.time()
            pixel_size = self.args.target_mpp if hasattr(self.args, 'target_mpp') and self.args.target_mpp else self.original_mpp
            
            # Detailed profiling of InstanSeg inference
            inference_breakdown = {}
            import torch
            
            # Check device and model status before inference
            if hasattr(self.instanseg, 'model') and self.instanseg.model is not None:
                try:
                    model_params = list(self.instanseg.model.parameters())
                    if len(model_params) > 0:
                        model_device = model_params[0].device
                        inference_breakdown['model_device'] = str(model_device)
                        # Check if model is actually on MPS
                        if str(model_device) != 'mps' and torch.backends.mps.is_available():
                            inference_breakdown['model_not_on_mps'] = True
                            inference_breakdown['model_should_be_mps'] = 'WARNING: Model not on MPS!'
                        else:
                            inference_breakdown['model_not_on_mps'] = False
                except Exception as e:
                    inference_breakdown['model_device_error'] = str(e)
            
            # Check MPS/CUDA availability
            inference_breakdown['mps_available'] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            inference_breakdown['cuda_available'] = torch.cuda.is_available()
            inference_breakdown['input_shape'] = str(img_np.shape)
            inference_breakdown['input_dtype'] = str(img_np.dtype)
            inference_breakdown['input_size_mb'] = img_np.nbytes / (1024 * 1024)
            
            # Time the actual inference call with synchronization
            # Synchronize if using GPU/MPS before timing to ensure clean measurement
            if inference_breakdown.get('mps_available', False):
                torch.mps.synchronize()
            elif inference_breakdown.get('cuda_available', False):
                torch.cuda.synchronize()
            
            # Try to pass device to eval_small_image if supported
            inference_start = time.time()
            try:
                # Check if eval_small_image accepts device parameter
                import inspect
                sig = inspect.signature(self.instanseg.eval_small_image)
                if 'device' in sig.parameters and self.device is not None:
                    labeled_output, image_tensor = self.instanseg.eval_small_image(img_np, pixel_size, device=self.device)
                    inference_breakdown['device_param_used'] = True
                else:
                    labeled_output, image_tensor = self.instanseg.eval_small_image(img_np, pixel_size)
                    inference_breakdown['device_param_used'] = False
            except TypeError:
                # Device parameter not supported, use default
                labeled_output, image_tensor = self.instanseg.eval_small_image(img_np, pixel_size)
                inference_breakdown['device_param_used'] = False
            inference_mid = time.time()
            
            # Synchronize after inference to ensure all work is done
            if inference_breakdown.get('mps_available', False):
                torch.mps.synchronize()
            elif inference_breakdown.get('cuda_available', False):
                torch.cuda.synchronize()
            
            inference_end = time.time()
            inference_breakdown['total_inference'] = inference_end - inference_start
            inference_breakdown['pure_inference'] = inference_mid - inference_start
            inference_breakdown['sync_time'] = inference_end - inference_mid
            
            # Check output device and type
            if hasattr(labeled_output, 'device'):
                inference_breakdown['output_device'] = str(labeled_output.device)
            if hasattr(labeled_output, 'dtype'):
                inference_breakdown['output_dtype'] = str(labeled_output.dtype)
            if hasattr(labeled_output, 'shape'):
                inference_breakdown['output_shape'] = str(labeled_output.shape)
                if hasattr(labeled_output, 'element_size'):
                    inference_breakdown['output_size_mb'] = labeled_output.numel() * labeled_output.element_size() / (1024 * 1024)
            
            # Check if output needs CPU transfer (this can be a major bottleneck!)
            if hasattr(labeled_output, 'cpu'):
                inference_breakdown['needs_cpu_transfer'] = True
                # Measure actual transfer time
                transfer_start = time.time()
                if inference_breakdown.get('mps_available', False):
                    torch.mps.synchronize()
                elif inference_breakdown.get('cuda_available', False):
                    torch.cuda.synchronize()
                _ = labeled_output.cpu()  # Time the transfer
                inference_breakdown['cpu_transfer_time'] = time.time() - transfer_start
            else:
                inference_breakdown['needs_cpu_transfer'] = False
            
            perf_times['instanseg_inference'] = time.time() - seg_start
            
            # Store breakdown for reporting
            perf_times['inference_breakdown'] = inference_breakdown
            
            # Convert labeled image to centroids and contours
            postprocess_start = time.time()
            centroids, contours, probabilities = self.labeled_to_contours(labeled_output)
            perf_times['postprocess'] = time.time() - postprocess_start
            
            # Scale back to original coordinates if resized
            if resize_factor != 1.0:
                centroids = (centroids / resize_factor).astype(np.int32)
                if contours is not None:
                    contours = (contours / resize_factor).astype(np.int32)
            
            # Add tile offset
            offset_start = time.time()
            if len(centroids) > 0:
                centroids[:, 0] += x_0
                centroids[:, 1] += y_0
                if contours is not None:
                    contours[:, :, 0] += x_0
                    contours[:, :, 1] += y_0
            perf_times['offset'] = time.time() - offset_start
            
            perf_times['total'] = time.time() - total_start
            
            # Always print performance breakdown for first 10 tiles, then every 50 tiles
            if not hasattr(self, '_tile_count'):
                self._tile_count = 0
            self._tile_count += 1
            
            should_print = (self._tile_count <= 10) or (self._tile_count % 50 == 0)
            
            if should_print:
                print(f"\n[TILE PERF #{self._tile_count}] Tile at ({x_0}, {y_0}):")
                print("  Main stages:")
                for stage, t in sorted(perf_times.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
                    if stage != 'total' and stage != 'inference_breakdown' and isinstance(t, (int, float)):
                        pct = (t / perf_times['total']) * 100
                        print(f"    {stage:25s}: {t:6.3f}s ({pct:5.1f}%)")
                
                # Add detailed inference breakdown
                if 'inference_breakdown' in perf_times:
                    breakdown = perf_times['inference_breakdown']
                    print(f"  Inference detailed breakdown:")
                    if 'model_device' in breakdown:
                        print(f"      {'model_device':25s}: {breakdown['model_device']}")
                        if breakdown.get('model_not_on_mps', False):
                            print(f"      {'!!! WARNING':25s}: Model NOT on MPS despite MPS being available!")
                            print(f"      {'':25s}: This is likely why inference is slow!")
                    if 'model_device_error' in breakdown:
                        print(f"      {'model_device_error':25s}: {breakdown['model_device_error']}")
                    if 'mps_available' in breakdown:
                        print(f"      {'mps_available':25s}: {breakdown['mps_available']}")
                    if 'cuda_available' in breakdown:
                        print(f"      {'cuda_available':25s}: {breakdown['cuda_available']}")
                    if 'input_shape' in breakdown:
                        print(f"      {'input_shape':25s}: {breakdown['input_shape']}")
                    if 'input_size_mb' in breakdown:
                        print(f"      {'input_size_mb':25s}: {breakdown['input_size_mb']:.2f} MB")
                    if 'total_inference' in breakdown:
                        print(f"      {'total_inference':25s}: {breakdown['total_inference']:6.3f}s")
                    if 'pure_inference' in breakdown:
                        print(f"      {'pure_inference':25s}: {breakdown['pure_inference']:6.3f}s")
                    if 'sync_time' in breakdown:
                        print(f"      {'sync_time':25s}: {breakdown['sync_time']:6.3f}s")
                    if 'output_device' in breakdown:
                        output_dev = breakdown['output_device']
                        print(f"      {'output_device':25s}: {output_dev}")
                        if output_dev == 'cpu' and breakdown.get('mps_available', False):
                            print(f"      {'!!! WARNING':25s}: Output on CPU - InstanSeg may not be using MPS!")
                    if 'output_shape' in breakdown:
                        print(f"      {'output_shape':25s}: {breakdown['output_shape']}")
                    if 'output_size_mb' in breakdown:
                        print(f"      {'output_size_mb':25s}: {breakdown['output_size_mb']:.2f} MB")
                    if 'cpu_transfer_time' in breakdown:
                        transfer_pct = (breakdown['cpu_transfer_time'] / breakdown['total_inference']) * 100 if breakdown['total_inference'] > 0 else 0
                        print(f"      {'cpu_transfer_time':25s}: {breakdown['cpu_transfer_time']:6.3f}s ({transfer_pct:.1f}% of inference) {'<-- BOTTLENECK!' if transfer_pct > 10 else ''}")
                    if 'device_param_used' in breakdown:
                        print(f"      {'device_param_used':25s}: {breakdown['device_param_used']}")
                    
                    # Performance recommendations
                    if breakdown.get('output_device') == 'cpu' and breakdown.get('mps_available', False):
                        print(f"  Performance recommendations:")
                        print(f"      - InstanSeg output is on CPU despite MPS being available")
                        print(f"      - This suggests InstanSeg may not support MPS acceleration")
                        print(f"      - Check InstanSeg version: pip show instanseg-torch")
                        print(f"      - Consider updating InstanSeg if MPS support was added recently")
                        print(f"      - Alternative: Use smaller tile sizes or batch processing")
                
                # Add postprocess breakdown if available
                if hasattr(self, '_postprocess_perf') and len(self._postprocess_perf) > 0:
                    last_post = self._postprocess_perf[-1]
                    if 'total_postprocess' in last_post:
                        print(f"  Postprocess breakdown (total: {last_post['total_postprocess']:.3f}s):")
                        stages = ['gpu_to_cpu_transfer', 'convert_to_numpy', 'tensor_to_numpy', 'dtype_conversion', 'regionprops', 'extract_props', 'array_conversion', 'pad_contours']
                        for stage in stages:
                            if stage in last_post:
                                pct = (last_post[stage] / last_post['total_postprocess']) * 100 if last_post['total_postprocess'] > 0 else 0
                                print(f"      {stage:25s}: {last_post[stage]:6.3f}s ({pct:5.1f}%)")
                        if 'input_device' in last_post:
                            print(f"      {'input_device':25s}: {last_post['input_device']}")
                
                print(f"  {'TOTAL':25s}: {perf_times['total']:6.3f}s (100.0%)")
                print(f"  {'nuclei_detected':25s}: {len(centroids)}")
                
                # Device diagnostics
                import torch
                print(f"  Device diagnostics:")
                print(f"      {'MPS available':25s}: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
                print(f"      {'CUDA available':25s}: {torch.cuda.is_available()}")
                print()
            
            return centroids, contours, probabilities
        
        except Exception as e:
            print(f"Error processing tile at ({x_0}, {y_0}): {e}")
            import traceback
            print(traceback.format_exc())
            return np.array([]).reshape(0, 2), None, np.array([])
    
    def _process_large_image_split(self, img_np, pixel_size, max_dim=1024):
        """Split large image into smaller chunks for faster InstanSeg inference"""
        from scipy import ndimage
        
        img_h, img_w = img_np.shape[:2]
        overlap = 64  # Small overlap to avoid edge artifacts
        
        # Calculate number of splits
        n_h = int(np.ceil(img_h / max_dim))
        n_w = int(np.ceil(img_w / max_dim))
        
        # Initialize output label image
        labeled_output = np.zeros((img_h, img_w), dtype=np.int32)
        current_label = 1
        
        for ih in range(n_h):
            for iw in range(n_w):
                # Calculate tile coordinates
                y0 = max(0, ih * max_dim - overlap if ih > 0 else 0)
                y1 = min(img_h, (ih + 1) * max_dim + overlap if ih < n_h - 1 else img_h)
                x0 = max(0, iw * max_dim - overlap if iw > 0 else 0)
                x1 = min(img_w, (iw + 1) * max_dim + overlap if iw < n_w - 1 else img_w)
                
                # Extract tile
                tile = img_np[y0:y1, x0:x1]
                
                # Process tile
                tile_labeled, _ = self.instanseg.eval_small_image(tile, pixel_size)
                
                # Convert to numpy if needed
                if hasattr(tile_labeled, 'cpu'):
                    tile_labeled = tile_labeled.cpu().numpy()
                elif not isinstance(tile_labeled, np.ndarray):
                    tile_labeled = np.array(tile_labeled)
                
                # Handle dimensions
                if len(tile_labeled.shape) == 4:
                    if tile_labeled.shape[1] <= 3:
                        tile_labeled = tile_labeled[0, 0] if tile_labeled.shape[1] == 1 else tile_labeled[0, 0]
                    else:
                        tile_labeled = tile_labeled[0, :, :, 0]
                elif len(tile_labeled.shape) == 3:
                    if tile_labeled.shape[0] == 1:
                        tile_labeled = tile_labeled[0]
                    else:
                        tile_labeled = tile_labeled[:, :, 0]
                
                # Convert to int32
                if tile_labeled.dtype in [np.float32, np.float64]:
                    tile_labeled = tile_labeled.astype(np.int32)
                
                # Remove overlap region labels (keep only core region)
                core_y0 = overlap if ih > 0 else 0
                core_y1 = tile_labeled.shape[0] - (overlap if ih < n_h - 1 else 0)
                core_x0 = overlap if iw > 0 else 0
                core_x1 = tile_labeled.shape[1] - (overlap if iw < n_w - 1 else 0)
                
                # Get core region and remap labels
                core_labels = tile_labeled[core_y0:core_y1, core_x0:core_x1]
                unique_labels = np.unique(core_labels)
                unique_labels = unique_labels[unique_labels > 0]  # Remove background
                
                # Remap labels to global label space
                label_map = {old: current_label + i for i, old in enumerate(unique_labels)}
                remapped = np.zeros_like(core_labels)
                for old_label, new_label in label_map.items():
                    remapped[core_labels == old_label] = new_label
                
                # Place in output
                output_y0 = y0 + core_y0
                output_y1 = y0 + core_y1
                output_x0 = x0 + core_x0
                output_x1 = x0 + core_x1
                labeled_output[output_y0:output_y1, output_x0:output_x1] = remapped
                
                current_label += len(unique_labels)
        
        # Convert back to torch tensor format for compatibility
        import torch
        labeled_output = torch.from_numpy(labeled_output).unsqueeze(0).unsqueeze(0).float()
        
        return labeled_output
    
    def run_WSI_segmentation(self):
        """Run segmentation on whole slide image"""
        overall_start_time = time.time()
        
        self.n_col = int(np.ceil(self.dim[0] / (self.tile_size - self.overlap)))
        self.n_row = int(np.ceil(self.dim[1] / (self.tile_size - self.overlap)))
        
        print(f"Processing {self.n_row} rows x {self.n_col} cols tiles")
        
        all_centroids = []
        all_contours = []
        all_probabilities = []
        
        # Count total tiles that will be processed (considering ROI)
        total_tiles = 0
        tiles_to_process = []
        for ir in range(self.n_row):
            for ic in range(self.n_col):
                x_0 = ic * (self.tile_size - self.overlap)
                y_0 = ir * (self.tile_size - self.overlap)
                if self._tile_in_roi(x_0, y_0, self.tile_size, self.tile_size):
                    total_tiles += 1
                    tiles_to_process.append((ir, ic, x_0, y_0))
        
        print(f"Total tiles to process: {total_tiles}")
        
        # Create progress bar
        pbar = tqdm(total=total_tiles, desc="Processing tiles", unit="tile")
        processed_tiles = 0
        
        # Process tiles
        for ir, ic, x_0, y_0 in tiles_to_process:
            # Process tile
            centroids, contours, probabilities = self.process_tile(
                x_0, y_0, self.tile_size, self.tile_size
            )
            
            if len(centroids) > 0:
                all_centroids.append(centroids)
                if contours is not None:
                    all_contours.append(contours)
                all_probabilities.append(probabilities)
            
            processed_tiles += 1
            pbar.update(1)
            
            # Update progress callback
            if self.progress_callback:
                progress = int((processed_tiles / total_tiles) * 100)
                self.progress_callback(progress)
        
        pbar.close()
        
        # Combine results
        if len(all_centroids) > 0:
            self.final_points = np.vstack(all_centroids).astype(np.int32)
            self.prob_all = np.concatenate(all_probabilities).astype(np.float32)
            
            if len(all_contours) > 0:
                # Combine contours
                max_contour_len = max(c.shape[1] for c in all_contours)
                padded_contours = []
                for c in all_contours:
                    if c.shape[1] < max_contour_len:
                        padded = np.pad(c, ((0, 0), (0, max_contour_len - c.shape[1]), (0, 0)), mode='edge')
                    else:
                        padded = c
                    padded_contours.append(padded)
                self.final_coord = np.vstack(padded_contours).astype(np.int32)
            else:
                self.final_coord = None
        else:
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = None
            self.prob_all = np.array([]).astype(np.float32)
        
        # Remove duplicates in overlap regions
        self.post_process_remove_duplicates()
        
        # Filter by ROI if specified
        if self.roi_polygon is not None:
            self._filter_by_polygon()
        elif self.roi_bbox is not None:
            self._filter_by_bbox()
        
        overall_end_time = time.time()
        print(f"\nTotal processing time: {overall_end_time - overall_start_time:.2f}s")
        print(f"Detected {len(self.final_points)} nuclei")
        
        print("---- Segmentation completed successfully ----")
    
    def _tile_in_roi(self, tile_x0, tile_y0, tile_w, tile_h):
        """Check if tile intersects with ROI"""
        tile_x1 = tile_x0 + tile_w
        tile_y1 = tile_y0 + tile_h
        
        if self.roi_polygon is None and self.roi_bbox is None:
            return True
        
        if self.roi_polygon:
            poly_xs = [p[0] for p in self.roi_polygon]
            poly_ys = [p[1] for p in self.roi_polygon]
            poly_min_x, poly_max_x = min(poly_xs), max(poly_xs)
            poly_min_y, poly_max_y = min(poly_ys), max(poly_ys)
            
            if tile_x1 < poly_min_x or tile_x0 > poly_max_x or tile_y1 < poly_min_y or tile_y0 > poly_max_y:
                return False
            return True
        
        if self.roi_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
            bbox_x1 = bbox_x + bbox_w
            bbox_y1 = bbox_y + bbox_h
            
            if tile_x1 < bbox_x or tile_x0 > bbox_x1 or tile_y1 < bbox_y or tile_y0 > bbox_y1:
                return False
        
        return True
    
    def _filter_by_polygon(self):
        """Filter centroids by polygon"""
        if self.final_points is None or len(self.final_points) == 0:
            return
        
        x_coords = self.final_points[:, 0]
        y_coords = self.final_points[:, 1]
        
        within_roi = np.array([self.point_in_polygon(x, y, self.roi_polygon)
                              for x, y in zip(x_coords, y_coords)])
        
        n_before = len(self.final_points)
        self.final_points = self.final_points[within_roi]
        if self.final_coord is not None:
            self.final_coord = self.final_coord[within_roi]
        self.prob_all = self.prob_all[within_roi]
        n_after = len(self.final_points)
        
        print(f"ROI polygon filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
    
    def _filter_by_bbox(self):
        """Filter centroids by bounding box"""
        if self.final_points is None or len(self.final_points) == 0:
            return
        
        bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
        roi_x0, roi_y0 = bbox_x, bbox_y
        roi_x1, roi_y1 = bbox_x + bbox_w, bbox_y + bbox_h
        
        x_coords = self.final_points[:, 0]
        y_coords = self.final_points[:, 1]
        within_roi = (x_coords >= roi_x0) & (x_coords <= roi_x1) & (y_coords >= roi_y0) & (y_coords <= roi_y1)
        
        n_before = len(self.final_points)
        self.final_points = self.final_points[within_roi]
        if self.final_coord is not None:
            self.final_coord = self.final_coord[within_roi]
        self.prob_all = self.prob_all[within_roi]
        n_after = len(self.final_points)
        
        print(f"ROI bbox filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
    
    def post_process_remove_duplicates(self, distance_threshold=10):
        """Remove duplicate detections in overlap regions"""
        if self.final_points is None or len(self.final_points) == 0:
            return
        
        # Calculate pairwise distances
        distances = cdist(self.final_points, self.final_points)
        
        # Find duplicates (points within threshold distance)
        to_remove = set()
        for i in range(len(self.final_points)):
            if i in to_remove:
                continue
            duplicates = np.where((distances[i] < distance_threshold) & (distances[i] > 0))[0]
            # Keep the one with higher probability, remove others
            if len(duplicates) > 0:
                candidates = [i] + list(duplicates)
                probs = self.prob_all[candidates]
                keep_idx = candidates[np.argmax(probs)]
                for idx in candidates:
                    if idx != keep_idx:
                        to_remove.add(idx)
        
        if len(to_remove) > 0:
            keep_mask = np.array([i not in to_remove for i in range(len(self.final_points))])
            n_before = len(self.final_points)
            self.final_points = self.final_points[keep_mask]
            if self.final_coord is not None:
                self.final_coord = self.final_coord[keep_mask]
            self.prob_all = self.prob_all[keep_mask]
            print(f"Removed {len(to_remove)} duplicate nuclei ({n_before} -> {len(self.final_points)})")

