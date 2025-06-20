# SceneValidator

A tool that validates scene composition and elements against predefined rules.

## Overview

SceneValidator helps ensure scene compositions adhere to defined guidelines and standards. It checks spatial relationships, color schemes, lighting conditions, and other elements against a set of validation rules.

## Features

- Validates scene composition against predefined rules
- Checks spatial relationships between elements
- Verifies color schemes and lighting conditions
- Ensures scene elements meet project specifications
- Generates detailed validation reports

## Trigger Mechanism

- File upload through web interface
- API call with JSON payload
- Integration with post-production pipeline

## Input Schema

Input is provided as a JSON scene description:

```json
{
  "scene_id": "scene_123",
  "elements": [
    {
      "id": "element_1",
      "type": "character",
      "position": {"x": 120, "y": 340},
      "dimensions": {"width": 50, "height": 180},
      "attributes": {
        "color_scheme": "warm",
        "lighting": "front"
      }
    },
    ...
  ],
  "composition": {
    "rule_of_thirds": true,
    "focal_point": {"x": 320, "y": 240},
    "color_palette": ["#FF5733", "#33FF57", "#3357FF"]
  }
}
```

## Output Schema

Validation report in JSON format:

```json
{
  "scene_id": "scene_123",
  "timestamp": "2025-06-20T15:30:00Z",
  "valid": false,
  "issues": [
    {
      "element_id": "element_1",
      "rule": "position.out_of_bounds",
      "severity": "error",
      "message": "Element is positioned outside visible area"
    }
  ],
  "warnings": [
    {
      "rule": "composition.color_contrast",
      "severity": "warning",
      "message": "Insufficient contrast between background and foreground elements"
    }
  ],
  "suggestions": [
    {
      "rule": "composition.balance",
      "message": "Consider repositioning elements to improve visual balance"
    }
  ]
}
```

## Implementation

Built with Python and Gemini API for advanced validation logic.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure validation rules in `config.yaml`

3. Set up environment variables:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line
```
python validator.py --input scene.json --rules standard_rules.yaml
```

### API
```python
from scene_validator import SceneValidator

validator = SceneValidator(rules_file="standard_rules.yaml")
result = validator.validate("scene.json")
print(result.is_valid)
print(result.issues)
```

## Integration

- Google Cloud Function implementation available
- REST API endpoints for service integration
- Firebase integration for real-time validation

## Documentation

For detailed documentation, see the [Google Doc](https://docs.google.com/document/d/...