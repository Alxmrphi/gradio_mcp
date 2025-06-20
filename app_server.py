#!/usr/bin/env python3
"""
Basic MCP Server Example
This server provides a simple calculator tool and weather resource.
"""

import json
import sys
from typing import Any, Dict, List, Optional

class MCPServer:
    def __init__(self):
        self.tools = {
            "calculator": {
                "name": "calculator",
                "description": "Perform basic arithmetic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
        
        self.resources = {
            "weather": {
                "uri": "weather://current",
                "name": "Current Weather",
                "description": "Current weather information",
                "mimeType": "application/json"
            }
        }

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP messages"""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        try:
            if method == "initialize":
                return self.handle_initialize(msg_id, params)
            elif method == "tools/list":
                return self.handle_list_tools(msg_id)
            elif method == "tools/call":
                return self.handle_call_tool(msg_id, params)
            elif method == "resources/list":
                return self.handle_list_resources(msg_id)
            elif method == "resources/read":
                return self.handle_read_resource(msg_id, params)
            else:
                return self.error_response(msg_id, f"Unknown method: {method}")
        
        except Exception as e:
            return self.error_response(msg_id, str(e))

    def handle_initialize(self, msg_id: str, params: Dict) -> Dict:
        """Handle initialization request"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "basic-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    def handle_list_tools(self, msg_id: str) -> Dict:
        """Return available tools"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": list(self.tools.values())}
        }

    def handle_call_tool(self, msg_id: str, params: Dict) -> Dict:
        """Execute a tool call"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "calculator":
            return self.execute_calculator(msg_id, arguments)
        else:
            return self.error_response(msg_id, f"Unknown tool: {tool_name}")

    def execute_calculator(self, msg_id: str, arguments: Dict) -> Dict:
        """Execute calculator tool"""
        expression = arguments.get("expression", "")
        
        try:
            # Simple evaluation - in production, use a safer parser
            result = eval(expression)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Result: {result}"
                        }
                    ]
                }
            }
        except Exception as e:
            return self.error_response(msg_id, f"Calculation error: {str(e)}")

    def handle_list_resources(self, msg_id: str) -> Dict:
        """Return available resources"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"resources": list(self.resources.values())}
        }

    def handle_read_resource(self, msg_id: str, params: Dict) -> Dict:
        """Read a resource"""
        uri = params.get("uri")
        
        if uri == "weather://current":
            # Mock weather data
            weather_data = {
                "temperature": 22,
                "condition": "sunny",
                "humidity": 65,
                "location": "San Francisco"
            }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(weather_data, indent=2)
                        }
                    ]
                }
            }
        else:
            return self.error_response(msg_id, f"Resource not found: {uri}")

    def error_response(self, msg_id: str, error_message: str) -> Dict:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -1,
                "message": error_message
            }
        }

    def run(self):
        """Main server loop - no async needed"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                message = json.loads(line.strip())
                response = self.handle_message(message)  # Also no need for async
                
                print(json.dumps(response), flush=True)
            except:
                continue

if __name__ == "__main__":
    server = MCPServer()
    server.run()