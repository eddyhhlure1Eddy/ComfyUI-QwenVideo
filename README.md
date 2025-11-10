<img width="1515" height="639" alt="image" src="https://github.com/user-attachments/assets/96fbad64-23a6-4054-996f-de81b352b016" />

<img width="1480" height="593" alt="image" src="https://github.com/user-attachments/assets/b291087a-b016-4be7-a240-06b05e12532b" />


# ComfyUI-QwenVideo

A ComfyUI custom node for video analysis and prompt generation using Qwen3-VL-235B vision-language model via OpenRouter API.

## Overview

This node enables reverse engineering of video prompts by analyzing video frames and generating detailed descriptions suitable for AI image/video generation models. It accepts video input from other ComfyUI nodes (like Load Video) and outputs both text prompts and extracted frames.

## Features

- **Direct Video Input**: Accepts VIDEO type input from ComfyUI nodes (compatible with Load Video, Kling nodes, etc.)
- **Intelligent Frame Extraction**: Uses FFmpeg to extract evenly-distributed frames from videos
- **Multiple Analysis Modes**:
  - Detailed Prompt: Comprehensive descriptions for AI generation
  - Simple Prompt: Concise 1-2 sentence summaries
  - Technical Details: Camera work, lighting, and cinematography analysis
  - Scene Description: Chronological narrative of video content
  - Custom: User-defined analysis instructions
- **Dual Output**: Generates both text prompts and frame tensors for downstream processing
- **UI Text Display**: Shows generated prompts directly on the node
- **Companion Nodes**: Includes helper nodes for video path extraction and text display

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "QwenVideo"
3. Click Install

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eddyhhlure1Eddy/ComfyUI-QwenVideo.git
cd ComfyUI-QwenVideo
pip install -r requirements.txt
```

### Requirements

- ComfyUI
- Python 3.8+
- FFmpeg (must be installed and accessible in PATH)
- Dependencies (auto-installed):
  - requests
  - pillow
  - numpy
  - torch

#### Installing FFmpeg

**Windows:**
```bash
winget install FFmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## API Key Setup

This node uses OpenRouter API to access Qwen3-VL-235B model.

### Getting an API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy your key (format: `sk-or-v1-...`)

### Using Your API Key

- Paste your API key into the `api_key` field in the node
- Or use the default embedded key (limited usage, not recommended for production)

## Usage

### Node Overview

**Qwen Video Prompt Reversal**

Inputs:
- `video` (VIDEO): Connect from Load Video or other video nodes
- `api_key` (STRING): Your OpenRouter API key
- `num_frames` (INT): Number of frames to extract (2-30)
- `analysis_mode` (COMBO): Analysis preset mode
- `custom_instruction` (STRING, optional): Custom analysis prompt

Outputs:
- `prompt` (STRING): Generated text description
- `frames` (IMAGE): Extracted video frames as tensor

### Basic Workflow

1. Add "Load Video" node and select your video
2. Add "Qwen Video Prompt Reversal" node
3. Connect Load Video's `VIDEO` output to Qwen's `video` input
4. Configure analysis settings
5. Add "Show Qwen Prompt" node to display results
6. Connect Qwen's `prompt` output to Show Qwen Prompt's `text` input
7. Run the workflow

### Example Workflows

Pre-configured workflows are available in the Hugging Face gift repository:

[Download Example Workflows](https://huggingface.co/eddy1111111/gift)

Workflow examples include:
- Basic video analysis
- Multi-mode comparison
- Video-to-prompt-to-video pipeline
- Batch processing setup

## Nodes Reference

### Qwen Video Prompt Reversal

Main analysis node that processes videos and generates prompts.

**Parameters:**
- `video`: Video input (accepts VIDEO type from other nodes)
- `api_key`: OpenRouter API key for Qwen3-VL access
- `num_frames`: Number of frames to extract (default: 6)
- `analysis_mode`: Preset analysis modes
- `custom_instruction`: Custom prompt for analysis mode

**Analysis Modes:**
- **detailed_prompt**: Comprehensive prompt with style, composition, and technical details
- **simple_prompt**: Concise description in 1-2 sentences
- **technical_details**: Focus on cinematography and production aspects
- **scene_description**: Chronological narrative of events
- **custom**: User-defined analysis instructions

### Get Video Path

Helper node to select videos from ComfyUI input folder and output file paths.

**Inputs:**
- `video`: Dropdown selection of videos in input folder

**Outputs:**
- `video_path`: Full path to selected video file

### Show Qwen Prompt

Display node for visualizing generated prompts in the UI.

**Inputs:**
- `text`: Text input (connect from Qwen's prompt output)

**Outputs:**
- `text`: Pass-through text output

## Technical Details

### Frame Extraction

The node uses FFmpeg to extract frames at evenly-distributed intervals:
```
interval = video_duration / (num_frames + 1)
```

Frames are saved as JPEG with quality level 2 for optimal balance between quality and file size.

### API Integration

- Model: `qwen/qwen3-vl-235b-a22b-thinking`
- Endpoint: OpenRouter API (https://openrouter.ai/api/v1/chat/completions)
- Image encoding: Base64 JPEG
- Timeout: 120 seconds
- Max frames: 30 (API limitation)

### Video Input Handling

The node accepts two types of video input:
1. **VideoInput objects**: From Load Video or similar nodes (automatically saved to temporary MP4)
2. **String paths**: Direct file paths (legacy support)

Temporary files are stored in: `ComfyUI/temp/qwen_video_cache/`

## Troubleshooting

### No frames extracted

**Cause**: FFmpeg not installed or not in PATH

**Solution**: Install FFmpeg and ensure it's accessible from command line
```bash
ffmpeg -version
```

### API request failed

**Cause**: Invalid API key or network issues

**Solution**: 
- Verify your OpenRouter API key is correct
- Check internet connection
- Ensure you have API credits

### "Unsupported video input type" error

**Cause**: Incorrect node connection

**Solution**: Connect a VIDEO output from Load Video or similar nodes

### Video file not found

**Cause**: Invalid file path

**Solution**: 
- Use Get Video Path node to ensure correct path
- Or place videos in `ComfyUI/input` folder
- Use Load Video node for automatic path handling

### No text displayed in UI

**Cause**: Node not set as output node or UI rendering issue

**Solution**:
- Ensure Show Qwen Prompt node is connected
- Refresh ComfyUI after workflow execution
- Check console for generated prompt text

## Performance Considerations

- Frame extraction time scales with video length
- API processing time: ~15-25 seconds for 6 frames
- Recommended frame count: 4-8 for optimal quality/speed balance
- Large videos (>1GB) may take longer to process

## Limitations

- Requires active internet connection
- API usage costs (check OpenRouter pricing)
- Maximum 30 frames per request
- Video must be readable by FFmpeg
- Supports common formats: MP4, MOV, AVI, WEBM, MKV, FLV, WMV, GIF

## Project Structure

```
ComfyUI-QwenVideo/
├── __init__.py           # Node registration
├── nodes.py              # Main node implementations
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── config.example.json  # Example configuration
└── install.py           # Auto-installer script
```

## Credits

- Author: eddy
- Model: Qwen3-VL-235B (Alibaba Cloud)
- API Provider: OpenRouter

## License

MIT License - see repository for full license text

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Changelog

### Version 1.1.0
- Changed video input to VIDEO type for direct node connections
- Added UI text display on main node
- Improved error handling and messages
- Added support for VideoInput objects
- Enhanced frame extraction reliability

### Version 1.0.0
- Initial release
- Basic video analysis functionality
- Multiple analysis modes
- OpenRouter API integration

## Support

For issues, questions, or feature requests:
- GitHub Issues: [Create an issue](https://github.com/eddyhhlure1Eddy/ComfyUI-QwenVideo/issues)
- Discussions: [Join the discussion](https://github.com/eddyhhlure1Eddy/ComfyUI-QwenVideo/discussions)

## Links

- [OpenRouter Platform](https://openrouter.ai/)
- [Qwen Model Documentation](https://qwen.readthedocs.io/)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Example Workflows (Hugging Face)](https://huggingface.co/eddy1111111/gift)
