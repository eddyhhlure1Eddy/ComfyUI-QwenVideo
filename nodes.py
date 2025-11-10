"""
ComfyUI Qwen Video Analysis Node
Author: eddy
"""

import os
import uuid
import requests
import base64
import json
import subprocess
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Try to import folder_paths from ComfyUI, fallback to temp directory
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    # Fallback for standalone testing
    class folder_paths:
        @staticmethod
        def get_temp_directory():
            import tempfile
            return tempfile.gettempdir()
        
        @staticmethod
        def get_filename_list(folder_name):
            return []


class QwenVideoPromptReversal:
    """
    A ComfyUI node that analyzes video content using Qwen3-VL model
    and generates prompt descriptions (reverse engineering prompts from video)
    """
    
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "qwen/qwen3-vl-235b-a22b-thinking"
        self.temp_dir = folder_paths.get_temp_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        # Accept a VIDEO input (same behavior as Kling Lip Sync nodes)
        return {
            "required": {
                "video": ("VIDEO", {}),
                "api_key": ("STRING", {
                    "default": "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",
                    "multiline": False,
                    "placeholder": "OpenRouter API Key"
                }),
                "num_frames": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "analysis_mode": (["detailed_prompt", "simple_prompt", "technical_details", "scene_description", "custom"], {
                    "default": "detailed_prompt"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Custom instruction (only for 'custom' mode)\nExample: Focus on color palette and lighting"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "frames")
    FUNCTION = "analyze_video"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True
    
    def extract_frames(self, video_path, num_frames):
        """Extract frames from video using ffmpeg"""
        # Handle relative path from ComfyUI input folder (same as official Load Video)
        if not os.path.isabs(video_path) and FOLDER_PATHS_AVAILABLE:
            try:
                input_dir = folder_paths.get_input_directory()
                full_path = os.path.join(input_dir, video_path)
                if os.path.exists(full_path):
                    video_path = full_path
                    print(f"[QwenVideo] Resolved video path: {video_path}")
            except Exception as e:
                print(f"[QwenVideo] Warning resolving path: {e}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}\nPlease check the path or place video in ComfyUI/input folder")
        
        # Create temp directory for frames
        frames_dir = Path(self.temp_dir) / "qwen_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Clear old frames
        for old_frame in frames_dir.glob("frame_*.jpg"):
            old_frame.unlink()
        
        # Get video duration
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            duration = None
        
        frame_paths = []
        
        if duration:
            interval = duration / (num_frames + 1)
            
            for i in range(1, num_frames + 1):
                timestamp = interval * i
                output_file = frames_dir / f"frame_{i:03d}.jpg"
                
                cmd = [
                    'ffmpeg',
                    '-ss', str(timestamp),
                    '-i', str(video_path),
                    '-frames:v', '1',
                    '-q:v', '2',
                    '-y',
                    str(output_file)
                ]
                
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
                if output_file.exists():
                    frame_paths.append(output_file)
        
        return frame_paths
    
    def encode_image_to_base64(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    
    def get_analysis_prompt(self, mode, custom_instruction=""):
        """Get analysis prompt based on mode"""
        if mode == "custom" and custom_instruction:
            return custom_instruction
        
        prompts = {
            "detailed_prompt": """Analyze these video frames and generate a detailed image generation prompt that could recreate this video's visual style and content.

Focus on:
1. Subject/main focus (person, object, scene)
2. Actions and movements
3. Visual style (cinematography, lighting, color grading)
4. Composition and framing
5. Mood and atmosphere
6. Technical aspects (camera angles, depth of field)

Format the output as a single comprehensive prompt suitable for image/video generation AI models.""",
            
            "simple_prompt": """Analyze these video frames and create a concise prompt (1-2 sentences) that describes the key visual elements and action, suitable for AI image generation.""",
            
            "technical_details": """Analyze these video frames and provide technical details:
1. Camera work (angles, movement, framing)
2. Lighting setup and mood
3. Color grading and palette
4. Composition techniques
5. Visual effects or post-processing
6. Cinematography style""",
            
            "scene_description": """Describe what's happening in these video frames chronologically:
1. Setting and environment
2. Characters/subjects and their actions
3. Key events in sequence
4. Emotional tone and narrative
5. Important visual details"""
        }
        
        return prompts.get(mode, prompts["detailed_prompt"])
    
    def analyze_frames_with_api(self, prompt, frame_paths, system_instruction=""):
        """Call OpenRouter API to analyze frames"""
        content = [{"type": "text", "text": prompt}]
        
        for frame_path in frame_paths:
            base64_image = self.encode_image_to_base64(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })
        
        # Build messages with optional system instruction
        messages = []
        # Use default system instruction if not provided
        if not system_instruction:
            system_instruction = "You are an expert video analyst and prompt engineer. Analyze the provided video frames and generate high-quality prompts for AI image/video generation."
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"[QwenVideo] Sending API request to {self.base_url}")
        print(f"[QwenVideo] Model: {self.model}")
        print(f"[QwenVideo] Number of images: {len(frame_paths)}")
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"[QwenVideo] API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[QwenVideo] API Response received successfully")
            return result
        else:
            error_detail = response.text
            print(f"[QwenVideo] API Error Details: {error_detail}")
            raise Exception(f"API request failed: {response.status_code} - {error_detail}")
    
    def load_frames_as_tensor(self, frame_paths):
        """Load frame images as tensor for ComfyUI"""
        images = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
        
        if images:
            # Stack images into batch
            images_tensor = torch.from_numpy(np.array(images))
            return images_tensor
        else:
            # Return empty tensor if no frames
            return torch.zeros((1, 64, 64, 3))
    
    def analyze_video(self, video, api_key, num_frames, analysis_mode, 
                     custom_instruction="", unique_id=None):
        """Main function called by ComfyUI"""
        try:
            # Resolve VIDEO input to a file path (supports both VideoInput and string path)
            resolved_path = None
            if isinstance(video, str) and video.strip():
                candidate = video.strip()
                if not os.path.isabs(candidate) and FOLDER_PATHS_AVAILABLE:
                    try:
                        input_dir = folder_paths.get_input_directory()
                        candidate2 = os.path.join(input_dir, candidate)
                        if os.path.exists(candidate2):
                            candidate = candidate2
                    except Exception:
                        pass
                resolved_path = candidate
            elif hasattr(video, "save_to"):
                # Comfy API VideoInput: save to temp mp4 and use existing ffmpeg extractor
                temp_dir = os.path.join(folder_paths.get_temp_directory(), "qwen_video_cache")
                os.makedirs(temp_dir, exist_ok=True)
                temp_name = f"qwen_video_{uuid.uuid4().hex}.mp4"
                temp_path = os.path.join(temp_dir, temp_name)
                try:
                    video.save_to(temp_path)
                    resolved_path = temp_path
                except Exception as e:
                    raise Exception(f"Failed to save VIDEO input to file: {e}")
            else:
                raise Exception("Unsupported video input type: expected VIDEO or file path string")
            
            # Update API key if provided
            if api_key and api_key.strip():
                self.api_key = api_key.strip()
                print(f"[QwenVideo] Using API key: {self.api_key[:20]}...")
            else:
                print(f"[QwenVideo] WARNING: No API key provided!")
            
            print(f"[QwenVideo] " + "="*60)
            print(f"[QwenVideo] Analyzing video: {resolved_path}")
            print(f"[QwenVideo] Frames to extract: {num_frames}")
            print(f"[QwenVideo] Analysis mode: {analysis_mode}")
            print(f"[QwenVideo] " + "="*60)
            
            # Extract frames
            frame_paths = self.extract_frames(resolved_path, num_frames)
            
            if not frame_paths:
                error_msg = "Error: No frames extracted from video"
                return {
                "ui": {"text": [error_msg]},
                "result": (error_msg, torch.zeros((1, 64, 64, 3)))
            }
            
            print(f"[QwenVideo] ✓ Extracted {len(frame_paths)} frames successfully")
            
            # Get analysis prompt
            prompt = self.get_analysis_prompt(analysis_mode, custom_instruction)
            
            # Show what instruction will be sent
            if analysis_mode == "custom" and custom_instruction:
                print(f"[QwenVideo] Using custom instruction: {custom_instruction[:100]}...")
            else:
                print(f"[QwenVideo] Using preset mode: {analysis_mode}")
            
            # Use default system prompt
            system_prompt = "You are an expert video analyst and prompt engineer. Analyze the provided video frames and generate high-quality prompts for AI image/video generation."
            
            # Analyze with API
            result = self.analyze_frames_with_api(prompt, frame_paths, system_prompt)
            
            # Extract response with error handling
            try:
                generated_prompt = result['choices'][0]['message']['content']
                print(f"[QwenVideo] ✓ Successfully extracted analysis from API response")
            except (KeyError, IndexError) as e:
                print(f"[QwenVideo] ✗ Error extracting response: {e}")
                print(f"[QwenVideo] Full API response: {result}")
                raise Exception(f"Invalid API response format: {e}")
            
            # Load frames as tensor
            frames_tensor = self.load_frames_as_tensor(frame_paths)
            
            print(f"[QwenVideo] " + "="*60)
            print(f"[QwenVideo] ✓ Analysis complete")
            print(f"[QwenVideo] Generated prompt length: {len(generated_prompt)} chars")
            print(f"[QwenVideo] Frames tensor shape: {frames_tensor.shape}")
            print(f"[QwenVideo] " + "="*60)
            print(f"[QwenVideo] PROMPT PREVIEW:")
            print(f"[QwenVideo] {generated_prompt[:300]}...")
            print(f"[QwenVideo] " + "="*60)
            
            # Return prompt and frames, and also render UI text on this node
            return {
                "ui": {"text": [generated_prompt]},
                "result": (generated_prompt, frames_tensor)
            }
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[QwenVideo] " + "="*60)
            print(f"[QwenVideo] ✗ ERROR OCCURRED")
            print(f"[QwenVideo] {error_msg}")
            print(f"[QwenVideo] " + "="*60)
            import traceback
            traceback.print_exc()
            print(f"[QwenVideo] " + "="*60)
            print(f"[QwenVideo] TROUBLESHOOTING TIPS:")
            print(f"[QwenVideo] 1. Check if ffmpeg is installed: run 'ffmpeg -version'")
            print(f"[QwenVideo] 2. Verify video file exists at specified path")
            print(f"[QwenVideo] 3. Check API key is valid (get one at https://openrouter.ai/)")
            print(f"[QwenVideo] 4. Ensure internet connection is working")
            print(f"[QwenVideo] 5. Place videos in ComfyUI/input folder for dropdown selection")
            print(f"[QwenVideo] " + "="*60)
            return (error_msg, torch.zeros((1, 64, 64, 3)))


class GetVideoPath:
    """Extract video path from Load Video node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get video files from ComfyUI's input folder
        video_extensions = ['mp4', 'mov', 'avi', 'webm', 'mkv', 'flv', 'wmv', 'gif']
        video_list = []
        
        if FOLDER_PATHS_AVAILABLE:
            try:
                input_dir = folder_paths.get_input_directory()
                for f in os.listdir(input_dir):
                    if os.path.isfile(os.path.join(input_dir, f)):
                        file_parts = f.split('.')
                        if len(file_parts) > 1 and file_parts[-1].lower() in video_extensions:
                            video_list.append(f)
            except Exception as e:
                print(f"[QwenVideo] Error loading video list: {e}")
        
        return {
            "required": {
                "video": (sorted(video_list) if video_list else ["no_videos_found"],),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "get_path"
    CATEGORY = "video/analysis"
    
    def get_path(self, video):
        """Get full video path"""
        if FOLDER_PATHS_AVAILABLE:
            try:
                input_dir = folder_paths.get_input_directory()
                full_path = os.path.join(input_dir, video)
                if os.path.exists(full_path):
                    print(f"[QwenVideo] Video path: {full_path}")
                    return (full_path,)
                else:
                    print(f"[QwenVideo] Video not found: {full_path}")
                    return (video,)
            except:
                return (video,)
        return (video,)


class ShowQwenPrompt:
    """Simple text display node for Qwen prompts"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "video/analysis"
    
    def show_text(self, text):
        """Display text in UI and forward it"""
        # Normalize to list of strings as UI expects an array
        if isinstance(text, (list, tuple)):
            out = [str(t) for t in text if t is not None]
        else:
            out = [str(text)]
        return {"ui": {"text": out}, "result": (out,)}


NODE_CLASS_MAPPINGS = {
    "QwenVideoPromptReversal": QwenVideoPromptReversal,
    "GetVideoPath": GetVideoPath,
    "ShowQwenPrompt": ShowQwenPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVideoPromptReversal": "Qwen Video Prompt Reversal",
    "GetVideoPath": "Get Video Path",
    "ShowQwenPrompt": "Show Qwen Prompt",
}
