from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for tool input."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against schema.
        
        Args:
            input_data: Tool input parameters
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        schema = self.input_schema
        required = schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
                
        return True
