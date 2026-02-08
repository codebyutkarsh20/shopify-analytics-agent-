import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPService:
    """Service for managing communication with the Shopify MCP server via JSON-RPC."""

    def __init__(self, settings: Settings):
        """
        Initialize the MCP service.

        Args:
            settings: Settings object containing Shopify credentials and configuration
        """
        self.settings = settings
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id: int = 0
        self._lock = asyncio.Lock()
        # Discovered tools from the MCP server (populated at startup)
        self.available_tools: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """
        Start the MCP server as a subprocess.

        Starts the shopify-mcp-server via npx and initializes the JSON-RPC connection.
        """
        if self.process is not None:
            logger.warning("MCP server is already running")
            return

        try:
            logger.info("Starting MCP server", command="npx -y shopify-mcp-server")

            # Set environment variables for the Shopify MCP server
            import os

            env = os.environ.copy()
            env["SHOPIFY_ACCESS_TOKEN"] = self.settings.shopify.access_token
            # shopify-mcp-server expects MYSHOPIFY_DOMAIN not SHOPIFY_SHOP_DOMAIN
            env["MYSHOPIFY_DOMAIN"] = self.settings.shopify.shop_domain

            # Start the subprocess with stdin/stdout pipes for JSON-RPC communication
            # Use larger buffer limit (10MB) to handle large Shopify API responses
            self.process = await asyncio.create_subprocess_exec(
                "npx",
                "-y",
                "shopify-mcp-server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=10 * 1024 * 1024,  # 10MB buffer limit
            )

            logger.info("MCP server process started", pid=self.process.pid)

            # Initialize the MCP connection
            await self.initialize()
            logger.info("MCP server initialized")

            # Discover all available tools from the MCP server
            await self._discover_tools()

        except Exception as e:
            logger.error("Failed to start MCP server", error=str(e), exc_info=True)
            self.process = None
            raise

    async def stop(self) -> None:
        """Stop the MCP server subprocess and clean up resources."""
        if self.process is None:
            logger.warning("MCP server is not running")
            return

        try:
            logger.info("Stopping MCP server", pid=self.process.pid)

            # Check if process is still running before terminating
            if self.process.returncode is not None:
                logger.info("MCP server already exited", returncode=self.process.returncode)
                return

            # Terminate the process gracefully
            self.process.terminate()

            # Wait for graceful termination with timeout
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info("MCP server stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("MCP server did not stop gracefully, killing it")
                self.process.kill()
                await self.process.wait()
                logger.info("MCP server killed")

        except Exception as e:
            logger.error("Error stopping MCP server", error=str(e), exc_info=True)
        finally:
            self.process = None

    async def initialize(self) -> None:
        """Send the MCP initialize request to set up the connection."""
        if self.process is None:
            raise RuntimeError("MCP server is not running")

        # Send initialize request
        init_request = self._build_jsonrpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "shopify-analytics-agent",
                    "version": "1.0.0",
                },
            },
        )

        await self._send_request(init_request)

        # Send initialized notification
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }

        await self._send_notification(init_notification)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available MCP tools from the Shopify server.

        Returns:
            List of tool definitions with name, description, and input schema
        """
        try:
            logger.info("Listing available MCP tools")
            result = await self.call_tool("tools/list", {}, is_meta=True)
            logger.info("Tools listed successfully", count=len(result.get("tools", [])))
            return result.get("tools", [])
        except Exception as e:
            logger.error("Failed to list tools", error=str(e), exc_info=True)
            raise

    async def _discover_tools(self) -> None:
        """
        Discover all tools from the MCP server and cache them.

        Called automatically during start(). The discovered tools are stored
        in self.available_tools and can be retrieved in Claude's format
        via get_tools_for_claude().
        """
        try:
            tools = await self.list_tools()
            self.available_tools = tools

            tool_names = [t.get("name", "?") for t in tools]
            logger.info(
                "MCP tools discovered",
                count=len(tools),
                tools=tool_names,
            )
        except Exception as e:
            logger.warning(
                "Failed to discover MCP tools — Claude will have no tools available",
                error=str(e),
            )
            self.available_tools = []

    def get_tools_for_claude(self) -> List[Dict[str, Any]]:
        """
        Convert discovered MCP tools into Claude's tool format.

        MCP uses 'inputSchema', Claude expects 'input_schema'.
        This method bridges that difference automatically.

        Returns:
            List of tool definitions in Claude API format
        """
        claude_tools = []
        for tool in self.available_tools:
            claude_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
            }
            claude_tools.append(claude_tool)

        return claude_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 30.0,
        is_meta: bool = False,
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with the given arguments.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Dictionary of arguments to pass to the tool
            timeout: Request timeout in seconds (default: 30)
            is_meta: Whether this is a meta tool call (tools/list)

        Returns:
            Parsed result from the tool

        Raises:
            TimeoutError: If the request times out
            RuntimeError: If the MCP server is not running
            ValueError: If the tool returns an error
        """
        if self.process is None:
            raise RuntimeError("MCP server is not running")

        try:
            logger.info("Calling MCP tool", tool=tool_name, arguments=arguments)

            # Build the request
            if is_meta:
                request = self._build_jsonrpc_request("tools/list", arguments)
            else:
                request = self._build_jsonrpc_request(
                    "tools/call",
                    {"name": tool_name, "arguments": arguments},
                )

            # Send request and get response with timeout
            response = await asyncio.wait_for(
                self._send_request(request),
                timeout=timeout,
            )

            # Check for JSON-RPC errors
            if "error" in response:
                error = response["error"]
                logger.error(
                    "MCP tool returned error",
                    tool=tool_name,
                    error_code=error.get("code"),
                    error_message=error.get("message"),
                )
                raise ValueError(f"MCP error: {error.get('message')}")

            logger.info("Tool call successful", tool=tool_name)
            return response.get("result", {})

        except asyncio.TimeoutError:
            logger.error("Tool call timed out", tool=tool_name, timeout=timeout)
            raise TimeoutError(f"Tool call '{tool_name}' timed out after {timeout}s")
        except Exception as e:
            logger.error(
                "Tool call failed",
                tool=tool_name,
                error=str(e),
                exc_info=True,
            )
            raise

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for the matching response.

        Reads lines from stdout in a loop, skipping empty lines and
        server-initiated notifications until we find the response
        that matches our request ID.

        Args:
            request: JSON-RPC request dictionary

        Returns:
            Parsed JSON-RPC response

        Raises:
            RuntimeError: If unable to communicate with the server
        """
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("MCP process is not properly initialized")

        expected_id = request.get("id")

        async with self._lock:
            try:
                # Send request as JSON line
                request_json = json.dumps(request) + "\n"
                self.process.stdin.write(request_json.encode("utf-8"))
                await self.process.stdin.drain()

                logger.debug("Request sent", request_id=expected_id)

                # Read lines in a loop until we get our matching response
                deadline = asyncio.get_event_loop().time() + 30.0
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    response_line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=remaining,
                    )

                    # Empty bytes means the process closed stdout
                    if not response_line:
                        # Try to read stderr for debugging
                        if self.process.stderr:
                            try:
                                stderr_data = await asyncio.wait_for(
                                    self.process.stderr.read(4096),
                                    timeout=1.0,
                                )
                                if stderr_data:
                                    stderr_msg = stderr_data.decode("utf-8", errors="ignore")
                                    logger.error("MCP server stderr", stderr=stderr_msg)
                            except asyncio.TimeoutError:
                                pass
                        raise RuntimeError("MCP server closed connection")

                    # Strip whitespace and skip empty lines
                    line_str = response_line.decode("utf-8").strip()
                    if not line_str:
                        logger.debug("Skipping empty line from MCP server")
                        continue

                    # Try to parse as JSON
                    try:
                        parsed = json.loads(line_str)
                    except json.JSONDecodeError:
                        # Not JSON — likely a log/debug line from the server, skip it
                        logger.debug("Skipping non-JSON line from MCP server", line=line_str[:200])
                        continue

                    # Check if this is a JSON-RPC notification (no "id" field) — skip it
                    if "id" not in parsed:
                        logger.debug(
                            "Skipping server notification",
                            method=parsed.get("method", "unknown"),
                        )
                        continue

                    # Check if this response matches our request ID
                    if parsed.get("id") != expected_id:
                        logger.debug(
                            "Skipping mismatched response",
                            expected_id=expected_id,
                            got_id=parsed.get("id"),
                        )
                        continue

                    # This is our response
                    logger.debug("Response received", response_id=parsed.get("id"))
                    return parsed

            except asyncio.TimeoutError:
                logger.error("Timeout waiting for MCP response")
                raise RuntimeError("Timeout waiting for MCP server response")

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            notification: JSON-RPC notification dictionary
        """
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP process is not properly initialized")

        try:
            notification_json = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_json.encode("utf-8"))
            await self.process.stdin.drain()

            logger.debug("Notification sent", method=notification.get("method"))

        except Exception as e:
            logger.error(
                "Failed to send notification",
                error=str(e),
                exc_info=True,
            )
            raise

    def _build_jsonrpc_request(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a JSON-RPC 2.0 request object.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            JSON-RPC request dictionary
        """
        request_id = self._next_id()
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

    def _next_id(self) -> int:
        """
        Generate the next request ID.

        Returns:
            Incrementing request ID
        """
        self.request_id += 1
        return self.request_id

    @asynccontextmanager
    async def managed_server(self):
        """
        Context manager for automatic server startup and shutdown.

        Usage:
            async with mcp_service.managed_server():
                await mcp_service.call_tool(...)
        """
        try:
            await self.start()
            yield
        finally:
            await self.stop()
