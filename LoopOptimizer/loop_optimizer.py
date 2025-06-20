#!/usr/bin/env python3
"""
LoopOptimizer - A tool for optimizing animation loops for performance and smoothness.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LoopOptimizer")

class FrameAnalyzer:
    """Analyzes frames to identify optimization opportunities."""
    
    def __init__(self):
        """Initialize the frame analyzer."""
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize the Gemini API client."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            self.gemini_model = None
            return
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {e}")
            self.gemini_model = None
    
    def analyze_frame_sequence(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """
        Analyze a sequence of frames to identify optimization opportunities.
        
        Args:
            frames: List of PIL Image objects representing the animation frames
            
        Returns:
            Dictionary with analysis results
        """
        if len(frames) < 2:
            return {"error": "Need at least 2 frames for analysis"}
        
        results = {
            "frame_count": len(frames),
            "transitions": [],
            "redundant_frames": [],
            "optimization_suggestions": []
        }
        
        # Analyze transitions between consecutive frames
        for i in range(len(frames) - 1):
            transition = self._analyze_transition(frames[i], frames[i+1])
            results["transitions"].append({
                "from_frame": i,
                "to_frame": i+1,
                "similarity": transition[0],
                "motion_areas": transition[1]
            })
        
        # Identify potentially redundant frames
        redundant = self._identify_redundant_frames(results["transitions"])
        results["redundant_frames"] = redundant
        
        # Generate optimization suggestions
        results["optimization_suggestions"] = self._generate_suggestions(frames, results)
        
        # Use Gemini for advanced analysis if available
        if self.gemini_model:
            gemini_suggestions = self._analyze_with_gemini(frames, results)
            results["gemini_suggestions"] = gemini_suggestions
        
        return results
    
    def _analyze_transition(self, frame1: Image.Image, frame2: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyze the transition between two consecutive frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Tuple of (similarity_score, motion_areas)
        """
        # Convert frames to numpy arrays for analysis
        arr1 = np.array(frame1.convert("RGB"))
        arr2 = np.array(frame2.convert("RGB"))
        
        # Calculate difference between frames
        diff = np.abs(arr1.astype(int) - arr2.astype(int))
        mean_diff = np.mean(diff)
        
        # Normalize to get similarity score (0 = completely different, 1 = identical)
        similarity = 1.0 - (mean_diff / 255.0)
        
        # Identify areas with significant motion
        motion_areas = []
        if arr1.shape[0] > 10 and arr1.shape[1] > 10:  # Ensure frame is large enough to divide
            # Divide the image into a grid and check each cell for motion
            cell_height = arr1.shape[0] // 5
            cell_width = arr1.shape[1] // 5
            
            for y in range(0, arr1.shape[0], cell_height):
                for x in range(0, arr1.shape[1], cell_width):
                    cell_diff = diff[y:y+cell_height, x:x+cell_width]
                    cell_mean_diff = np.mean(cell_diff)
                    
                    if cell_mean_diff > 10:  # Threshold for considering motion significant
                        motion_areas.append({
                            "x": x,
                            "y": y,
                            "width": cell_width,
                            "height": cell_height,
                            "intensity": float(cell_mean_diff / 255.0)
                        })
        
        return similarity, motion_areas
    
    def _identify_redundant_frames(self, transitions: List[Dict[str, Any]]) -> List[int]:
        """
        Identify potentially redundant frames based on transition analysis.
        
        Args:
            transitions: List of transition data
            
        Returns:
            List of frame indices that might be redundant
        """
        redundant = []
        
        for i, transition in enumerate(transitions):
            # If frames are very similar (> 0.98 similarity) and there's little motion
            if transition["similarity"] > 0.98 and len(transition["motion_areas"]) < 3:
                # Mark the second frame as potentially redundant
                redundant.append(transition["to_frame"])
        
        return redundant
    
    def _generate_suggestions(self, frames: List[Image.Image], analysis: Dict[str, Any]) -> List[str]:
        """
        Generate optimization suggestions based on the analysis.
        
        Args:
            frames: List of animation frames
            analysis: Analysis results
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check for redundant frames
        if analysis["redundant_frames"]:
            redundant_str = ", ".join(map(str, analysis["redundant_frames"]))
            suggestions.append(f"Consider removing redundant frames: {redundant_str}")
        
        # Check for smooth transitions
        low_similarity_transitions = []
        for t in analysis["transitions"]:
            if t["similarity"] < 0.7:  # Threshold for identifying abrupt transitions
                low_similarity_transitions.append((t["from_frame"], t["to_frame"]))
        
        if low_similarity_transitions:
            for from_frame, to_frame in low_similarity_transitions:
                suggestions.append(f"Add intermediate frames between {from_frame} and {to_frame} for smoother transition")
        
        # Check for overall loop smoothness
        if len(frames) >= 3:
            # Check similarity between last and first frame (loop closure)
            last_to_first = self._analyze_transition(frames[-1], frames[0])
            if last_to_first[0] < 0.7:
                suggestions.append("Improve loop closure: last and first frames have abrupt transition")
        
        return suggestions
    
    def _analyze_with_gemini(self, frames: List[Image.Image], analysis: Dict[str, Any]) -> List[str]:
        """
        Use Gemini AI to perform advanced analysis on the animation frames.
        
        Args:
            frames: List of animation frames
            analysis: Existing analysis results
            
        Returns:
            List of additional suggestions from Gemini
        """
        if not self.gemini_model:
            return ["Gemini API not available for advanced analysis"]
        
        try:
            # Select representative frames to send to Gemini (first, middle, last)
            sample_frames = [frames[0]]
            if len(frames) > 2:
                sample_frames.append(frames[len(frames) // 2])
            sample_frames.append(frames[-1])
            
            # Create a prompt for Gemini
            prompt = f"""
            Analyze these animation frames and provide suggestions for optimization.
            
            Animation details:
            - Total frames: {len(frames)}
            - Potentially redundant frames: {analysis["redundant_frames"]}
            
            Focus on:
            1. Visual consistency across frames
            2. Smoothness of motion
            3. Color and lighting consistency
            4. Efficient use of keyframes
            
            Provide specific, actionable optimization suggestions.
            """
            
            response = self.gemini_model.generate_content([prompt] + sample_frames)
            
            if hasattr(response, 'text'):
                # Extract suggestions from Gemini's response
                suggestions = []
                for line in response.text.strip().split('\n'):
                    line = line.strip()
                    if line and (':' in line or '-' in line or line.startswith('•')):
                        # This looks like a suggestion point
                        suggestions.append(line)
                
                return suggestions if suggestions else ["Gemini analysis completed but no specific suggestions were provided"]
            
            return ["Error: Gemini response did not contain text"]
            
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return [f"Error during Gemini analysis: {str(e)}"]


class LoopOptimizer:
    """Main class for optimizing animation loops."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the loop optimizer.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config = self._load_config(config_file) if config_file else {}
        self.frame_analyzer = FrameAnalyzer()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return {}
    
    def optimize(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize an animation loop.
        
        Args:
            input_path: Path to the input animation (directory of frames or GIF)
            output_path: Optional path for the optimized output
            
        Returns:
            Dictionary with optimization results
        """
        # Load frames from input
        frames = self._load_frames(input_path)
        if not frames:
            return {"error": f"Failed to load frames from {input_path}"}
        
        # Analyze frames
        analysis = self.frame_analyzer.analyze_frame_sequence(frames)
        
        # Perform optimization
        optimized_frames = self._optimize_frames(frames, analysis)
        
        # Save optimized animation if output path provided
        if output_path:
            self._save_optimized_animation(optimized_frames, output_path)
        
        # Return analysis and optimization results
        return {
            "analysis": analysis,
            "original_frame_count": len(frames),
            "optimized_frame_count": len(optimized_frames),
            "optimization_ratio": len(optimized_frames) / len(frames) if frames else 0
        }
    
    def _load_frames(self, input_path: str) -> List[Image.Image]:
        """
        Load animation frames from a file or directory.
        
        Args:
            input_path: Path to the input animation (directory of frames or GIF)
            
        Returns:
            List of PIL Image objects representing the animation frames
        """
        frames = []
        
        try:
            if os.path.isdir(input_path):
                # Load frames from directory of image files
                image_files = sorted([f for f in os.listdir(input_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                
                for file_name in image_files:
                    file_path = os.path.join(input_path, file_name)
                    try:
                        img = Image.open(file_path)
                        frames.append(img.copy())
                    except Exception as e:
                        logger.warning(f"Failed to load frame {file_path}: {e}")
            
            elif input_path.lower().endswith('.gif'):
                # Load frames from GIF file
                gif = Image.open(input_path)
                for i in range(0, gif.n_frames):
                    gif.seek(i)
                    frames.append(gif.convert('RGBA').copy())
            
            else:
                logger.error(f"Unsupported input format: {input_path}")
        
        except Exception as e:
            logger.error(f"Error loading frames: {e}")
        
        return frames
    
    def _optimize_frames(self, frames: List[Image.Image], analysis: Dict[str, Any]) -> List[Image.Image]:
        """
        Optimize animation frames based on analysis.
        
        Args:
            frames: Original animation frames
            analysis: Analysis results
            
        Returns:
            Optimized animation frames
        """
        if not frames:
            return []
        
        # Start with all frames
        optimized = frames.copy()
        
        # Remove redundant frames if suggested and configured
        if self.config.get("remove_redundant_frames", True) and analysis.get("redundant_frames"):
            # Sort in reverse order to avoid index shifting when removing frames
            redundant = sorted(analysis["redundant_frames"], reverse=True)
            
            for idx in redundant:
                if 0 <= idx < len(optimized):
                    del optimized[idx]
        
        # Apply other optimizations as configured
        if self.config.get("equalize_frames", False):
            optimized = self._equalize_frames(optimized)
        
        return optimized
    
    def _equalize_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        Equalize brightness and contrast across frames for more consistent animation.
        
        Args:
            frames: Animation frames
            
        Returns:
            Equalized animation frames
        """
        if not frames:
            return []
        
        # This is a simplified implementation - a real system would use more 
        # sophisticated image processing techniques
        
        # Calculate average brightness across all frames
        avg_brightness = 0
        for frame in frames:
            # Convert to grayscale and calculate mean brightness
            gray = frame.convert("L")
            brightness = np.mean(np.array(gray))
            avg_brightness += brightness
        
        avg_brightness /= len(frames)
        
        # Adjust each frame to match average brightness
        equalized = []
        for frame in frames:
            gray = frame.convert("L")
            brightness = np.mean(np.array(gray))
            
            # Calculate brightness adjustment factor
            factor = avg_brightness / brightness if brightness > 0 else 1.0
            
            # Apply brightness adjustment
            if 0.9 <= factor <= 1.1:  # Only make small adjustments
                adjusted = Image.fromarray(
                    np.clip(np.array(frame) * factor, 0, 255).astype(np.uint8)
                )
                equalized.append(adjusted)
            else:
                equalized.append(frame.copy())
        
        return equalized
    
    def _save_optimized_animation(self, frames: List[Image.Image], output_path: str):
        """
        Save optimized animation to the specified output path.
        
        Args:
            frames: Optimized animation frames
            output_path: Path where to save the output
        """
        if not frames:
            logger.warning("No frames to save")
            return
        
        try:
            if output_path.lower().endswith('.gif'):
                # Save as GIF
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=self.config.get("frame_duration", 100),  # Frame duration in ms
                    loop=0  # 0 = loop indefinitely
                )
                logger.info(f"Saved optimized GIF to {output_path}")
            
            elif os.path.isdir(output_path):
                # Save as individual frames in directory
                for i, frame in enumerate(frames):
                    file_path = os.path.join(output_path, f"frame_{i:04d}.png")
                    frame.save(file_path)
                logger.info(f"Saved {len(frames)} optimized frames to {output_path}")
            
            else:
                logger.error(f"Unsupported output format: {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving optimized animation: {e}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Optimize animation loops for performance and smoothness")
    parser.add_argument("--input", required=True, help="Input animation (directory of frames or GIF)")
    parser.add_argument("--output", help="Output path for optimized animation")
    parser.add_argument("--config", help="Path to configuration file (JSON)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze without optimization")
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = LoopOptimizer(args.config)
    
    if args.analyze_only:
        # Load frames and perform analysis only
        frames = optimizer._load_frames(args.input)
        if not frames:
            logger.error(f"Failed to load frames from {args.input}")
            return 1
        
        analysis = optimizer.frame_analyzer.analyze_frame_sequence(frames)
        print(json.dumps(analysis, indent=2))
    else:
        # Perform full optimization
        result = optimizer.optimize(args.input, args.output)
        print(json.dumps(result, indent=2))
    
    return 0


if __name__ == "__main__":
    main()