#!/usr/bin/env python3
"""
SceneValidator - A tool to validate scene composition and elements against predefined rules.
"""

import os
import json
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SceneValidator")

class ValidationResult:
    """Class representing the result of a scene validation."""
    
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.valid = True
        self.issues: List[Dict[str, str]] = []
        self.warnings: List[Dict[str, str]] = []
        self.suggestions: List[Dict[str, str]] = []
    
    def add_issue(self, element_id: Optional[str], rule: str, message: str, severity: str = "error"):
        """Add an issue to the validation result."""
        issue = {
            "rule": rule,
            "severity": severity,
            "message": message
        }
        if element_id:
            issue["element_id"] = element_id
        
        self.issues.append(issue)
        self.valid = False
    
    def add_warning(self, rule: str, message: str):
        """Add a warning to the validation result."""
        self.warnings.append({
            "rule": rule,
            "severity": "warning",
            "message": message
        })
    
    def add_suggestion(self, rule: str, message: str):
        """Add a suggestion to the validation result."""
        self.suggestions.append({
            "rule": rule,
            "message": message
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "valid": self.valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }
    
    def to_json(self) -> str:
        """Convert the validation result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SceneValidator:
    """Main validator class for checking scene composition and elements."""
    
    def __init__(self, rules_file: str):
        """
        Initialize the validator with rules from a YAML file.
        
        Args:
            rules_file: Path to the YAML file containing validation rules
        """
        self.rules = self._load_rules(rules_file)
        self._init_gemini()
    
    def _load_rules(self, rules_file: str) -> Dict[str, Any]:
        """Load validation rules from a YAML file."""
        try:
            with open(rules_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading rules file: {e}")
            return {}
    
    def _init_gemini(self):
        """Initialize the Gemini API client."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {e}")
            self.gemini_model = None
    
    def _load_scene(self, scene_file: str) -> Dict[str, Any]:
        """Load scene data from a JSON file."""
        try:
            with open(scene_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading scene file: {e}")
            return {}
    
    def validate(self, scene_input: Union[str, Dict[str, Any]]) -> ValidationResult:
        """
        Validate a scene against the loaded rules.
        
        Args:
            scene_input: Either a path to a JSON file or a scene data dictionary
            
        Returns:
            ValidationResult object containing validation results
        """
        # Load scene data if a file path is provided
        scene_data = scene_input
        if isinstance(scene_input, str):
            scene_data = self._load_scene(scene_input)
        
        # Check if we have valid scene data
        if not scene_data:
            result = ValidationResult("unknown")
            result.add_issue(None, "input.invalid", "Invalid or empty scene data", "error")
            return result
        
        # Create validation result object
        scene_id = scene_data.get("scene_id", "unknown")
        result = ValidationResult(scene_id)
        
        # Perform basic validation checks
        self._validate_scene_structure(scene_data, result)
        
        # Validate elements
        if "elements" in scene_data:
            for element in scene_data["elements"]:
                self._validate_element(element, scene_data, result)
        
        # Validate composition
        if "composition" in scene_data:
            self._validate_composition(scene_data["composition"], scene_data, result)
        
        # Use Gemini for advanced validation if available
        if self.gemini_model:
            self._advanced_validation_with_gemini(scene_data, result)
        
        return result
    
    def _validate_scene_structure(self, scene_data: Dict[str, Any], result: ValidationResult):
        """Validate the basic structure of the scene data."""
        required_fields = self.rules.get("required_fields", ["scene_id"])
        
        for field in required_fields:
            if field not in scene_data:
                result.add_issue(None, "structure.missing_field", f"Required field '{field}' is missing", "error")
    
    def _validate_element(self, element: Dict[str, Any], scene_data: Dict[str, Any], result: ValidationResult):
        """Validate a single element in the scene."""
        element_id = element.get("id", "unknown")
        
        # Check required element fields
        required_element_fields = self.rules.get("element_required_fields", ["id", "type"])
        for field in required_element_fields:
            if field not in element:
                result.add_issue(element_id, f"element.missing_field.{field}", 
                               f"Required element field '{field}' is missing", "error")
        
        # Check position bounds if defined
        if "position" in element and "bounds" in self.rules:
            bounds = self.rules["bounds"]
            pos = element["position"]
            
            if pos.get("x", 0) < bounds.get("min_x", 0) or pos.get("x", 0) > bounds.get("max_x", 1000):
                result.add_issue(element_id, "element.position.out_of_bounds_x", 
                               "Element X position is outside allowed bounds", "error")
            
            if pos.get("y", 0) < bounds.get("min_y", 0) or pos.get("y", 0) > bounds.get("max_y", 1000):
                result.add_issue(element_id, "element.position.out_of_bounds_y", 
                               "Element Y position is outside allowed bounds", "error")
        
        # More specific element validation based on element type
        element_type = element.get("type", "")
        type_rules = self.rules.get("element_types", {}).get(element_type, {})
        
        for rule_name, rule_def in type_rules.items():
            self._apply_rule(rule_name, rule_def, element, scene_data, result, element_id)
    
    def _validate_composition(self, composition: Dict[str, Any], scene_data: Dict[str, Any], result: ValidationResult):
        """Validate the composition of the scene."""
        # Check color palette
        if "color_palette" in composition and self.rules.get("validate_color_palette", False):
            if len(composition["color_palette"]) < self.rules.get("min_colors", 3):
                result.add_warning("composition.color_palette.too_few", 
                                "Color palette has fewer colors than recommended")
        
        # Check focal point
        if "focal_point" in composition and "elements" in scene_data:
            # Check if any element is at or near the focal point
            focal_point = composition["focal_point"]
            found_focal_element = False
            
            for element in scene_data["elements"]:
                if "position" in element:
                    position = element["position"]
                    distance = ((position.get("x", 0) - focal_point.get("x", 0)) ** 2 + 
                               (position.get("y", 0) - focal_point.get("y", 0)) ** 2) ** 0.5
                    
                    if distance <= self.rules.get("focal_point_threshold", 50):
                        found_focal_element = True
                        break
            
            if not found_focal_element:
                result.add_warning("composition.focal_point.unused", 
                                 "Focal point defined but no elements are positioned near it")
    
    def _apply_rule(self, rule_name: str, rule_def: Dict[str, Any], 
                   element: Dict[str, Any], scene_data: Dict[str, Any], 
                   result: ValidationResult, element_id: str):
        """Apply a specific validation rule to an element."""
        # This is a simplified implementation; in a real system, this would be more robust
        rule_type = rule_def.get("type", "")
        
        if rule_type == "attribute_required":
            attribute_path = rule_def.get("attribute", "")
            if not self._check_attribute_exists(element, attribute_path):
                result.add_issue(element_id, f"element.missing_attribute.{rule_name}", 
                               f"Required attribute '{attribute_path}' is missing", "error")
        
        elif rule_type == "attribute_value":
            attribute_path = rule_def.get("attribute", "")
            allowed_values = rule_def.get("allowed_values", [])
            value = self._get_attribute_value(element, attribute_path)
            
            if value is not None and allowed_values and value not in allowed_values:
                result.add_issue(element_id, f"element.invalid_value.{rule_name}", 
                               f"Value '{value}' for '{attribute_path}' is not in allowed values {allowed_values}", "error")
    
    def _check_attribute_exists(self, obj: Dict[str, Any], attribute_path: str) -> bool:
        """Check if an attribute exists in an object, supporting dot notation."""
        parts = attribute_path.split('.')
        current = obj
        
        for part in parts:
            if part not in current:
                return False
            current = current[part]
        
        return True
    
    def _get_attribute_value(self, obj: Dict[str, Any], attribute_path: str) -> Any:
        """Get an attribute value from an object, supporting dot notation."""
        parts = attribute_path.split('.')
        current = obj
        
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        
        return current
    
    def _advanced_validation_with_gemini(self, scene_data: Dict[str, Any], result: ValidationResult):
        """Use Gemini API for advanced scene validation."""
        if not self.gemini_model:
            return
        
        try:
            # Prepare prompt for Gemini
            prompt = f"""
            Analyze this scene description and identify potential issues with:
            1. Composition balance
            2. Color harmony
            3. Spatial arrangement
            4. Visual flow
            
            Provide feedback in these categories only. Be specific and constructive.
            
            Scene data:
            {json.dumps(scene_data, indent=2)}
            """
            
            # Get response from Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                # Parse Gemini's response
                lines = response.text.strip().split('\n')
                current_category = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is a category header
                    lower_line = line.lower()
                    if "composition balance" in lower_line:
                        current_category = "composition.balance"
                    elif "color harmony" in lower_line:
                        current_category = "composition.color_harmony"
                    elif "spatial arrangement" in lower_line:
                        current_category = "composition.spatial_arrangement"
                    elif "visual flow" in lower_line:
                        current_category = "composition.visual_flow"
                    elif current_category and (':' in line or '-' in line):
                        # This is a suggestion within a category
                        message = line.split(':', 1)[-1].strip() if ':' in line else line.split('-', 1)[-1].strip()
                        result.add_suggestion(current_category, message)
        
        except Exception as e:
            logger.error(f"Error during Gemini validation: {e}")
            result.add_warning("system.gemini_error", "Advanced validation with Gemini API failed")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate scene composition and elements")
    parser.add_argument("--input", required=True, help="Input scene JSON file")
    parser.add_argument("--rules", required=True, help="Rules YAML file")
    parser.add_argument("--output", help="Output JSON file for validation results")
    args = parser.parse_args()
    
    # Initialize validator
    validator = SceneValidator(args.rules)
    
    # Validate scene
    result = validator.validate(args.input)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result.to_json())
    else:
        print(result.to_json())
    
    # Exit with appropriate status code
    return 0 if result.valid else 1


if __name__ == "__main__":
    main()