
from typing import Dict, Any
from .base import BaseTool

class ChartGeneratorTool(BaseTool):
    """Tool for generating data visualizations."""
    
    @property
    def name(self) -> str:
        return "generate_chart"

    @property
    def description(self) -> str:
        return (
            "Generate a chart image that appears INLINE in your Telegram response. "
            "ONLY call this tool when: (1) the user explicitly requests a chart/graph/visualization, "
            "OR (2) the data has 3+ comparison points and a chart clearly adds value. "
            "Do NOT call this tool if the user said 'no chart', 'text only', 'brief', or 'just the number'. "
            "After calling this tool, you MUST write [CHART:N] (where N is the chart_index from the result) "
            "on its own line in your response, at the exact position you want the image to appear. "
            "Chart types: 'bar' for comparisons, 'line' for time trends, "
            "'pie' for distribution, 'horizontal_bar' for ranked lists."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "horizontal_bar"],
                    "description": (
                        "Type of chart. Use 'bar' for comparisons, 'line' for time trends, "
                        "'pie' for proportions, 'horizontal_bar' for ranked lists."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Chart title (e.g., 'Top 5 Products by Revenue', 'Order Trend — Last 7 Days')",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Category labels or time points (e.g., product names, dates)",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric values corresponding to each label (e.g., revenue, count)",
                },
                "y_axis_label": {
                    "type": "string",
                    "description": "Optional axis label (e.g., 'Revenue (₹)', 'Orders')",
                },
            },
            "required": ["chart_type", "title", "labels", "values"],
        }
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        super().validate(input_data)
        
        labels = input_data.get("labels", [])
        values = input_data.get("values", [])
        
        if len(labels) != len(values):
            raise ValueError(
                f"labels ({len(labels)}) and values ({len(values)}) must have the same length"
            )
            
        chart_type = input_data.get("chart_type")
        if chart_type not in ["bar", "line", "pie", "horizontal_bar"]:
            raise ValueError(f"Invalid chart_type: {chart_type}")
            
        return True
