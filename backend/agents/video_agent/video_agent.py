"""
Video Agent Implementation

This module implements a specialized agent for analyzing visual elements
of content to extract engagement-related features, based on Media Richness
Theory (Daft & Lengel, 1986) and visual attention research.
"""

import logging
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import requests
import os
import time
from sklearn.cluster import KMeans
from collections import Counter

from ..base_agent import AgentMessage, AgentStatus, BaseAgent


class VideoAgent(BaseAgent):
    """
    Agent responsible for video content analysis using computer vision techniques.
    
    This agent extracts engagement indicators from the visual components of content,
    implementing research findings on visual attention and engagement factors.
    """
    
    def __init__(self, agent_id: str = "video_agent_default"):
        """Initialize the video agent with default capabilities."""
        super().__init__(
            agent_id=agent_id,
            agent_type="video_agent",
            description="Analyzes visual elements of content to extract engagement features",
            version="0.1.0"
        )
        
        # Define agent capabilities
        self.update_capabilities([
            "scene_transition_detection",
            "visual_complexity_analysis",
            "motion_intensity_measurement",
            "color_scheme_analysis",
            "production_quality_assessment",
            "thumbnail_effectiveness_analysis"
        ])
        
        self.logger = logging.getLogger(f"agent.video.{agent_id}")
        self.update_status(AgentStatus.READY)
        
        # Constants for analysis
        self._SCENE_THRESHOLD = 30.0  # Threshold for scene transition detection
        self._SAMPLE_RATE = 1.0  # Sample every second by default
        self._THUMBNAIL_SIZE = (640, 360)  # Default thumbnail size
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize computer vision models for video analysis.
        
        In a production environment, this would load pre-trained models.
        For this implementation, we'll use OpenCV built-in algorithms.
        """
        # Initialize scene transition detector
        self._scene_transition_model = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Load face detector for thumbnail analysis
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Other models would be initialized here in a full implementation
        self._visual_complexity_model = None
        self._motion_analysis_model = None
        self._color_analysis_model = None
        self._production_quality_model = None
        self._thumbnail_analysis_model = None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video content to extract engagement-related features.
        
        Based on Media Richness Theory, this analyzes visual elements
        that contribute to the richness of communication.
        
        Args:
            input_data: Dict containing video data and processing parameters
                Required keys:
                - video_path: Path to video file or URL
                - content_id: Unique identifier for the content
                Optional keys:
                - analyze_thumbnail: Whether to analyze thumbnail (default: False)
                - thumbnail_path: Path to thumbnail image (if applicable)
                - temporal_resolution: How many frames per second to analyze
        
        Returns:
            Dict containing extracted video features
        """
        self.update_status(AgentStatus.PROCESSING)
        self.logger.info(f"Processing video content for {input_data.get('content_id', 'unknown')}")
        
        try:
            # Get video path
            video_path = input_data.get("video_path")
            if not video_path:
                raise ValueError("video_path is required")
            
            # Load video using OpenCV
            cap, video_metadata = self._load_video(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Set sample rate from input data or use default
            sample_rate = input_data.get("temporal_resolution", self._SAMPLE_RATE)
            
            # Extract frames at regular intervals for analysis
            frames, timestamps = self._extract_frames(cap, sample_rate)
            
            # Analyze video features
            scene_transitions = self._detect_scene_transitions({"frames": frames, "timestamps": timestamps})
            visual_complexity = self._analyze_visual_complexity({"frames": frames})
            motion_intensity = self._analyze_motion({"frames": frames, "timestamps": timestamps})
            color_scheme = self._analyze_color_scheme({"frames": frames})
            production_quality = self._assess_production_quality({"frames": frames, "video_metadata": video_metadata})
            
            # Construct results
            results = {
                "content_id": input_data.get("content_id", "unknown"),
                "video_features": {
                    "resolution": f"{int(video_metadata['width'])}x{int(video_metadata['height'])}",
                    "fps": video_metadata["fps"],
                    "duration": video_metadata["duration"],
                    "scene_transitions": scene_transitions,
                    "visual_complexity": visual_complexity,
                    "motion_intensity": motion_intensity,
                    "color_scheme": color_scheme,
                    "production_quality": production_quality,
                }
            }
            
            # Add thumbnail analysis if requested
            if input_data.get("analyze_thumbnail", False):
                thumbnail_path = input_data.get("thumbnail_path")
                if thumbnail_path:
                    thumbnail_img = self._load_image(thumbnail_path)
                else:
                    # Use first frame as thumbnail if no specific thumbnail provided
                    thumbnail_img = frames[0] if frames else None
                
                if thumbnail_img is not None:
                    results["video_features"]["thumbnail_data"] = self._analyze_thumbnail(thumbnail_img)
            
            # Clean up
            cap.release()
            
            self.update_status(AgentStatus.READY)
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            self.update_status(AgentStatus.ERROR)
            return {
                "error": str(e),
                "content_id": input_data.get("content_id", "unknown")
            }
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages from other agents.
        
        Args:
            message: The message to handle
        """
        if message.message_type == "process_request":
            result = await self.process(message.content)
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="process_response",
                content=result,
                correlation_id=message.correlation_id
            )
        elif message.message_type == "status_request":
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="status_response",
                content={"status": self.get_status().value},
                correlation_id=message.correlation_id
            )
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    def _load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, float]]:
        """
        Load a video file or URL using OpenCV.
        
        Args:
            video_path: Path to video file or URL
            
        Returns:
            Tuple of (VideoCapture object, video metadata)
        """
        # If video is a URL, download it to a temporary file
        if video_path.startswith(('http://', 'https://')):
            temp_dir = tempfile.gettempdir()
            local_filename = os.path.join(temp_dir, f"video_{int(time.time())}.mp4")
            
            response = requests.get(video_path, stream=True)
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            video_path = local_filename
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # Extract video metadata
        metadata = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        return cap, metadata
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image file or URL using OpenCV.
        
        Args:
            image_path: Path to image file or URL
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            # If image is a URL, download it to a temporary file
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                # Load local image
                img = cv2.imread(image_path)
            
            return img
        
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def _extract_frames(self, cap: cv2.VideoCapture, sample_rate: float) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames from a video at regular intervals.
        
        Args:
            cap: OpenCV VideoCapture object
            sample_rate: How many frames to extract per second
            
        Returns:
            Tuple of (list of frames, list of timestamps)
        """
        frames = []
        timestamps = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Calculate frame interval based on sample rate
        frame_interval = int(fps / sample_rate)
        if frame_interval < 1:
            frame_interval = 1
        
        # Extract frames at regular intervals
        current_frame = 0
        
        while True:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame and timestamp to lists
            frames.append(frame)
            timestamps.append(current_frame / fps)
            
            # Move to next frame position
            current_frame += frame_interval
            if current_frame >= frame_count:
                break
        
        return frames, timestamps
    
    def _detect_scene_transitions(self, input_data: Dict[str, Any]) -> List[float]:
        """
        Detect scene transitions in the video.
        
        This implementation uses frame differencing to detect significant
        changes between consecutive frames that indicate scene transitions.
        
        Args:
            input_data: Dictionary containing frames and timestamps
            
        Returns:
            List of timestamps (in seconds) where scene transitions occur
        """
        frames = input_data.get("frames", [])
        timestamps = input_data.get("timestamps", [])
        
        if not frames or len(frames) < 2:
            return []
        
        scene_transitions = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            # Convert current frame to grayscale
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference between consecutive frames
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            
            # Calculate mean change
            mean_diff = np.mean(frame_diff)
            
            # Detect scene transition if mean difference exceeds threshold
            if mean_diff > self._SCENE_THRESHOLD:
                scene_transitions.append(timestamps[i])
            
            # Update previous frame
            prev_gray = curr_gray
        
        return scene_transitions
    
    def _analyze_visual_complexity(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze the visual complexity of video frames.
        
        This analyzes spatial complexity, edge density, and other factors
        that contribute to cognitive load based on Information Processing Theory.
        
        Args:
            input_data: Dictionary containing video frames
            
        Returns:
            Dict containing visual complexity metrics
        """
        frames = input_data.get("frames", [])
        
        if not frames:
            return {
                "spatial_complexity": 0.0,
                "temporal_complexity": 0.0,
                "information_density": 0.0,
                "edge_density": 0.0,
                "object_count_avg": 0.0
            }
        
        # Calculate metrics across all frames
        spatial_complexity_values = []
        edge_density_values = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate spatial complexity using entropy
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            histogram_normalized = histogram / histogram.sum()
            non_zero_vals = histogram_normalized[histogram_normalized > 0]
            entropy = -np.sum(non_zero_vals * np.log2(non_zero_vals))
            max_entropy = np.log2(256)  # Maximum possible entropy
            spatial_complexity = entropy / max_entropy
            spatial_complexity_values.append(spatial_complexity)
            
            # Calculate edge density using Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_density_values.append(edge_density)
        
        # Calculate temporal complexity as standard deviation of spatial complexity
        temporal_complexity = np.std(spatial_complexity_values) * 5.0  # Scale to 0-1 range
        temporal_complexity = min(1.0, temporal_complexity)
        
        # Estimate information density based on spatial complexity and edge density
        information_density = (np.mean(spatial_complexity_values) * 0.6 + 
                               np.mean(edge_density_values) * 0.4)
        
        # Estimate average object count (simplified implementation)
        # In a full implementation, this would use object detection
        object_count_avg = 4.0 + (information_density * 5.0)
        
        return {
            "spatial_complexity": float(np.mean(spatial_complexity_values)),
            "temporal_complexity": float(temporal_complexity),
            "information_density": float(information_density),
            "edge_density": float(np.mean(edge_density_values)),
            "object_count_avg": float(object_count_avg)
        }
    
    def _analyze_motion(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze motion characteristics in the video.
        
        This implementation calculates motion intensity, consistency,
        and camera stability using optical flow techniques.
        
        Args:
            input_data: Dictionary containing video frames and timestamps
            
        Returns:
            Dict containing motion analysis metrics
        """
        frames = input_data.get("frames", [])
        
        if not frames or len(frames) < 2:
            return {
                "motion_intensity_avg": 0.0,
                "motion_consistency": 0.0,
                "camera_stability": 0.0,
                "motion_segments": 0,
                "dynamic_range": 0.0
            }
        
        # Initialize optical flow parameters
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Initialize variables for tracking motion
        motion_values = []
        camera_motion_values = []
        
        for i in range(1, len(frames)):
            # Convert current frame to grayscale
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Lucas-Kanade method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude and direction
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Calculate motion intensity as mean magnitude
            motion_intensity = np.mean(magnitude)
            motion_values.append(motion_intensity)
            
            # Calculate camera motion by analyzing global flow patterns
            # Simplified implementation - in a full version this would be more sophisticated
            h, w = flow.shape[:2]
            step = 16
            y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            
            # Calculate average motion vector
            avg_fx = np.mean(fx)
            avg_fy = np.mean(fy)
            camera_motion = np.sqrt(avg_fx*avg_fx + avg_fy*avg_fy)
            camera_motion_values.append(camera_motion)
            
            # Update previous frame
            prev_gray = curr_gray
        
        # Normalize motion values to 0-1 range
        max_motion = max(motion_values) if motion_values else 1.0
        normalized_motion = [m / max_motion for m in motion_values] if max_motion > 0 else [0.0] * len(motion_values)
        
        # Calculate motion consistency as inverse of std dev
        motion_std = np.std(normalized_motion)
        motion_consistency = 1.0 - min(1.0, motion_std * 2.0)
        
        # Calculate camera stability as inverse of camera motion
        max_camera_motion = max(camera_motion_values) if camera_motion_values else 1.0
        normalized_camera_motion = [m / max_camera_motion for m in camera_motion_values] if max_camera_motion > 0 else [0.0] * len(camera_motion_values)
        camera_stability = 1.0 - min(1.0, np.mean(normalized_camera_motion) * 1.5)
        
        # Identify distinct motion segments
        motion_threshold = 0.2
        motion_segments = 1  # Start with 1 segment
        for i in range(1, len(normalized_motion)):
            if abs(normalized_motion[i] - normalized_motion[i-1]) > motion_threshold:
                motion_segments += 1
        
        # Calculate dynamic range as difference between max and min motion
        dynamic_range = max(normalized_motion) - min(normalized_motion) if normalized_motion else 0.0
        
        return {
            "motion_intensity_avg": float(np.mean(normalized_motion)),
            "motion_consistency": float(motion_consistency),
            "camera_stability": float(camera_stability),
            "motion_segments": motion_segments,
            "dynamic_range": float(dynamic_range)
        }
    
    def _analyze_color_scheme(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze the color scheme of the video.
        
        This implementation extracts color palette, measures color diversity,
        harmony, saturation, brightness, and contrast.
        
        Args:
            input_data: Dictionary containing video frames
            
        Returns:
            Dict containing color analysis metrics
        """
        frames = input_data.get("frames", [])
        
        if not frames:
            return {
                "color_diversity": 0.0,
                "color_harmony": 0.0,
                "saturation_avg": 0.0,
                "brightness_avg": 0.0,
                "contrast_avg": 0.0
            }
        
        # Sample frames at regular intervals (use up to 10 frames)
        sample_size = min(len(frames), 10)
        sampled_frames = frames[::max(1, len(frames) // sample_size)]
        
        # Initialize metrics
        saturation_values = []
        brightness_values = []
        contrast_values = []
        dominant_colors_all = []
        
        for frame in sampled_frames:
            # Convert to HSV for better color analysis
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Extract HSV channels
            h, s, v = cv2.split(hsv_frame)
            
            # Calculate saturation
            saturation = np.mean(s) / 255.0
            saturation_values.append(saturation)
            
            # Calculate brightness
            brightness = np.mean(v) / 255.0
            brightness_values.append(brightness)
            
            # Calculate contrast using standard deviation of intensity
            contrast = np.std(v) / 255.0
            contrast_values.append(contrast)
            
            # Extract dominant colors using K-means clustering
            # Reshape image to be a list of pixels
            pixels = frame.reshape(-1, 3).astype(np.float32)
            
            # Downsample for efficiency
            sample_indices = np.random.choice(len(pixels), min(1000, len(pixels)), replace=False)
            pixels_sample = pixels[sample_indices]
            
            # Apply K-means to find dominant colors
            n_colors = 5
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels_sample)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # Count pixels assigned to each color
            labels_count = Counter(kmeans.labels_)
            
            # Store dominant colors with their proportions
            dominant_colors = []
            for i in range(n_colors):
                color = colors[i].tolist()
                proportion = labels_count[i] / len(kmeans.labels_)
                dominant_colors.append((color, proportion))
            
            dominant_colors_all.extend(dominant_colors)
        
        # Calculate color diversity based on number of distinct dominant colors
        # Simplification: cluster the dominant colors again to find unique color groups
        if dominant_colors_all:
            color_array = np.array([c[0] for c in dominant_colors_all])
            proportion_array = np.array([c[1] for c in dominant_colors_all])
            
            # Cluster dominant colors
            color_kmeans = KMeans(n_clusters=min(10, len(color_array)), random_state=42, n_init=10)
            color_kmeans.fit(color_array)
            
            # Count unique color groups weighted by their proportion
            unique_colors = {}
            for i, label in enumerate(color_kmeans.labels_):
                if label not in unique_colors:
                    unique_colors[label] = 0
                unique_colors[label] += proportion_array[i]
            
            # Calculate effective number of colors
            effective_colors = len(unique_colors)
            
            # Calculate Effective Color Diversity (normalized to 0-1)
            color_diversity = min(1.0, effective_colors / 10.0)
        else:
            color_diversity = 0.0
        
        # Calculate color harmony (simplified implementation)
        # In a full implementation, this would use color theory principles
        harmony_score = 0.75  # Default value for typical videos
        
        return {
            "color_diversity": float(color_diversity),
            "color_harmony": float(harmony_score),
            "saturation_avg": float(np.mean(saturation_values)),
            "brightness_avg": float(np.mean(brightness_values)),
            "contrast_avg": float(np.mean(contrast_values))
        }
    
    def _assess_production_quality(self, input_data: Dict[str, Any]) -> float:
        """
        Assess the overall production quality of the video.
        
        This combines multiple metrics to estimate production value.
        
        Args:
            input_data: Dictionary containing frames and video metadata
            
        Returns:
            Production quality score (0-1)
        """
        frames = input_data.get("frames", [])
        video_metadata = input_data.get("video_metadata", {})
        
        if not frames:
            return 0.0
        
        # Factors that influence production quality:
        
        # 1. Resolution quality
        resolution_factor = 0.0
        width = video_metadata.get("width", 0)
        height = video_metadata.get("height", 0)
        
        if width and height:
            # Normalize based on common resolutions
            if width >= 3840 or height >= 2160:  # 4K
                resolution_factor = 1.0
            elif width >= 1920 or height >= 1080:  # 1080p
                resolution_factor = 0.9
            elif width >= 1280 or height >= 720:  # 720p
                resolution_factor = 0.7
            elif width >= 854 or height >= 480:  # 480p
                resolution_factor = 0.5
            else:  # Lower quality
                resolution_factor = 0.3
        
        # 2. Frame stability assessment
        if len(frames) >= 2:
            stability_measures = []
            prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            
            for i in range(1, min(len(frames), 5)):  # Check first few frames
                curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                frame_diff = cv2.absdiff(curr_frame, prev_frame)
                stability = 1.0 - (np.mean(frame_diff) / 255.0)
                stability_measures.append(stability)
                
                prev_frame = curr_frame
            
            stability_factor = np.mean(stability_measures) if stability_measures else 0.5
        else:
            stability_factor = 0.5
        
        # 3. Noise assessment
        noise_measures = []
        for frame in frames[:5]:  # Check first few frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Estimate noise using Laplacian filter
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = np.std(laplacian) / 10.0  # Normalize
            
            # Inverse noise level (higher is better)
            noise_quality = 1.0 - min(1.0, noise_level)
            noise_measures.append(noise_quality)
        
        noise_factor = np.mean(noise_measures) if noise_measures else 0.5
        
        # 4. Lighting quality
        lighting_measures = []
        for frame in frames[:5]:  # Check first few frames
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2]  # Value channel represents brightness
            
            # Calculate mean and std of brightness
            mean_brightness = np.mean(v) / 255.0
            std_brightness = np.std(v) / 255.0
            
            # Ideal brightness around 0.5-0.6 with some variation
            brightness_quality = 1.0 - min(1.0, abs(mean_brightness - 0.55) * 2.0)
            
            # Some variation is good, but not too much
            variation_quality = 1.0 - min(1.0, abs(std_brightness - 0.15) * 5.0)
            
            lighting_quality = (brightness_quality * 0.7) + (variation_quality * 0.3)
            lighting_measures.append(lighting_quality)
        
        lighting_factor = np.mean(lighting_measures) if lighting_measures else 0.5
        
        # Combine factors with appropriate weights
        production_quality = (
            resolution_factor * 0.25 +
            stability_factor * 0.25 +
            noise_factor * 0.25 +
            lighting_factor * 0.25
        )
        
        return float(production_quality)
    
    def _analyze_thumbnail(self, thumbnail_img: np.ndarray) -> Dict[str, float]:
        """
        Analyze the effectiveness of a video thumbnail.
        
        Based on visual attention research, this assesses factors that
        predict click-through rate, including visual salience, face presence,
        text presence, emotion intensity, and color contrast.
        
        Args:
            thumbnail_img: Thumbnail image as numpy array
            
        Returns:
            Dict containing thumbnail effectiveness metrics
        """
        if thumbnail_img is None:
            return {
                "visual_salience": 0.0,
                "text_presence": 0.0,
                "face_presence": 0.0,
                "emotion_intensity": 0.0,
                "color_contrast": 0.0,
                "click_prediction": 0.0
            }
        
        # Resize thumbnail for analysis
        thumbnail = cv2.resize(thumbnail_img, self._THUMBNAIL_SIZE)
        
        # 1. Detect faces
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Calculate face presence and prominence
        face_presence = 0.0
        emotion_intensity = 0.0
        
        if len(faces) > 0:
            face_presence = 1.0  # Face detected
            
            # Calculate total face area relative to image size
            total_face_area = sum(w * h for (x, y, w, h) in faces)
            image_area = thumbnail.shape[0] * thumbnail.shape[1]
            face_area_ratio = min(1.0, total_face_area / (image_area * 0.5))
            
            # Estimate emotion intensity based on face size and position
            # Central faces with larger area are typically more emotional
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Calculate distance from center
            center_x, center_y = thumbnail.shape[1] / 2, thumbnail.shape[0] / 2
            face_center_x, face_center_y = x + w/2, y + h/2
            distance_from_center = np.sqrt((center_x - face_center_x)**2 + (center_y - face_center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            center_factor = 1.0 - min(1.0, distance_from_center / max_distance)
            
            # Combine size and position for emotion intensity
            emotion_intensity = face_area_ratio * 0.7 + center_factor * 0.3
        
        # 2. Detect text presence (simplified using edge detection)
        # In a full implementation, this would use OCR
        edges = cv2.Canny(gray, 100, 200)
        text_like_edges = cv2.dilate(edges, None, iterations=1)
        text_like_ratio = np.sum(text_like_edges > 0) / (edges.shape[0] * edges.shape[1])
        text_presence = min(1.0, text_like_ratio * 5.0)  # Scale up, as text usually occupies a small area
        
        # 3. Calculate color contrast
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Measure contrast using standard deviation of value channel
        contrast = np.std(value) / 128.0  # Normalize
        color_contrast = min(1.0, contrast * 2.0)  # Scale up for better differentiation
        
        # 4. Calculate visual salience using a simplified model
        # In a full implementation, this would use a more sophisticated saliency model
        
        # Compute lab image for better perceptual accuracy
        lab = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate mean values for each channel
        l_mean, a_mean, b_mean = np.mean(l), np.mean(a), np.mean(b)
        
        # Calculate color uniqueness for each pixel
        l_unique = np.abs(l - l_mean)
        a_unique = np.abs(a - a_mean)
        b_unique = np.abs(b - b_mean)
        
        # Combine uniqueness maps
        uniqueness = (l_unique + a_unique + b_unique) / 3.0
        
        # Apply Gaussian filter for smoothing
        uniqueness = cv2.GaussianBlur(uniqueness, (5, 5), 0)
        
        # Normalize to 0-1
        visual_salience = min(1.0, np.mean(uniqueness) / 30.0)
        
        # 5. Combine metrics to predict click-through potential
        # Weights based on research on thumbnail effectiveness
        click_prediction = (
            visual_salience * 0.25 +
            face_presence * 0.25 +
            emotion_intensity * 0.2 +
            text_presence * 0.15 +
            color_contrast * 0.15
        )
        
        return {
            "visual_salience": float(visual_salience),
            "text_presence": float(text_presence),
            "face_presence": float(face_presence),
            "emotion_intensity": float(emotion_intensity),
            "color_contrast": float(color_contrast),
            "click_prediction": float(click_prediction)
        } 