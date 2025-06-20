# LoopOptimizer

A tool for optimizing animation loops for performance and smoothness.

## Overview

LoopOptimizer analyzes animation frames to identify optimization opportunities, such as redundant frames, abrupt transitions, and inconsistencies in brightness and color. It can then apply optimizations to improve performance and visual quality.

## Features

- Identifies and removes redundant frames
- Detects abrupt transitions between frames
- Analyzes loop closure quality (transition from last to first frame)
- Equalizes brightness and contrast across frames
- Uses Google Gemini AI for advanced analysis
- Supports various input and output formats (image sequences, GIFs)

## Trigger Mechanism

- Command-line interface
- API call with JSON configuration
- Integration with animation pipelines
- Watch folder for automated processing

## Input Schema

For the command-line interface:

```
--input        Path to input animation (directory of frames or GIF)
--output       Path for optimized output (optional)
--config       Path to configuration file (optional)
--analyze-only Only perform analysis without optimization
```

Configuration file (JSON):

```json
{
  "remove_redundant_frames": true,
  "equalize_frames": true,
  "frame_duration": 100,
  "similarity_threshold": 0.98,
  "motion_threshold": 10,
  "optimization_level": "medium"
}
```

## Output Schema

Analysis results:

```json
{
  "analysis": {
    "frame_count": 24,
    "transitions": [
      {
        "from_frame": 0,
        "to_frame": 1,
        "similarity": 0.92,
        "motion_areas": [
          {
            "x": 120,
            "y": 80,
            "width": 60,
            "height": 40,
            "intensity": 0.35
          }
        ]
      }
    ],
    "redundant_frames": [5, 12, 18],
    "optimization_suggestions": [
      "Consider removing redundant frames: 5, 12, 18",
      "Add intermediate frames between 8 and 9 for smoother transition",
      "Improve loop closure: last and first frames have abrupt transition"
    ],
    "gemini_suggestions": [
      "Inconsistent lighting between frames 3-7",
      "Consider reducing color palette for better performance",
      "Motion blur would improve perceived smoothness"
    ]
  },
  "original_frame_count": 24,
  "optimized_frame_count": 21,
  "optimization_ratio": 0.875
}
```

## Implementation

Built with Python and Gemini API for advanced analysis of animation characteristics.

## Dependencies

- Python 3.9+
- NumPy
- Pillow (PIL)
- Google Generative AI Python SDK

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line

```bash
# Basic optimization
python loop_optimizer.py --input frames_directory/ --output optimized.gif

# Analysis only
python loop_optimizer.py --input animation.gif --analyze-only

# With custom configuration
python loop_optimizer.py --input frames_directory/ --output optimized_frames/ --config config.json
```

### Python API

```python
from loop_optimizer import LoopOptimizer

# Initialize optimizer with configuration
optimizer = LoopOptimizer('config.json')

# Optimize animation
result = optimizer.optimize('input_frames/', 'output.gif')
print(f"Optimization ratio: {result['optimization_ratio']}")

# Analyze only
frames = optimizer._load_frames('animation.gif')
analysis = optimizer.frame_analyzer.analyze_frame_sequence(frames)
print(analysis['optimization_suggestions'])
```

## Integration

LoopOptimizer is designed to integrate with other tools in the media production workflow:

- **Input from**: 
  - StoryboardGen (for animating storyboards)
  - TimelineAssembler (for sequence previews)

- **Output to**:
  - FormatNormalizer (for standardizing optimized animations)
  - VeoPromptExporter (for generating optimized video prompts)

## Documentation

For detailed documentation, see the [Google Doc](https://docs.google.com/document/d/...).