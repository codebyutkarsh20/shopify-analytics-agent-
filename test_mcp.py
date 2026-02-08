#!/usr/bin/env python3
"""Comprehensive end-to-end test for MCP connection and data retrieval."""

import asyncio
import json
import sys
sys.path.insert(0, '.')

from src.config.settings import settings
from src.services.mcp_service import MCPService


async def test_mcp_end_to_end():
    """Test MCP server connection and all available tools."""
    print("=" * 60)
    print("END-TO-END MCP TEST")
    print("=" * 60)
    
    mcp_service = MCPService(settings)
    
    try:
        # 1. Start MCP
        print("\n1️⃣ Starting MCP service...")
        await mcp_service.start()
        print("   ✅ MCP server started!")
        
        # 2. List tools
        print("\n2️⃣ Listing available tools...")
        tools = await mcp_service.list_tools()
        print(f"   ✅ Found {len(tools)} tools")
        
        # 3. Test get-shop
        print("\n3️⃣ Testing get-shop...")
        try:
            result = await mcp_service.call_tool("get-shop", {})
            print(f"   ✅ get-shop returned:")
            print(f"   {json.dumps(result, indent=2)[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 4. Test get-orders
        print("\n4️⃣ Testing get-orders...")
        try:
            result = await mcp_service.call_tool("get-orders", {"limit": 5})
            print(f"   ✅ get-orders returned:")
            # Check if it's the expected format
            if isinstance(result, dict):
                if 'content' in result:
                    content = result.get('content', [])
                    if content and isinstance(content[0], dict) and 'text' in content[0]:
                        text = content[0]['text']
                        print(f"   Response text preview: {text[:300]}...")
                    else:
                        print(f"   {json.dumps(result, indent=2)[:500]}...")
                else:
                    print(f"   {json.dumps(result, indent=2)[:500]}...")
            else:
                print(f"   Raw: {str(result)[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 5. Test get-products
        print("\n5️⃣ Testing get-products...")
        try:
            result = await mcp_service.call_tool("get-products", {"limit": 3})
            print(f"   ✅ get-products returned:")
            if isinstance(result, dict) and 'content' in result:
                content = result.get('content', [])
                if content and isinstance(content[0], dict) and 'text' in content[0]:
                    text = content[0]['text']
                    print(f"   Response text preview: {text[:300]}...")
                else:
                    print(f"   {json.dumps(result, indent=2)[:500]}...")
            else:
                print(f"   {json.dumps(result, indent=2)[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 6. Test get-customers
        print("\n6️⃣ Testing get-customers...")
        try:
            result = await mcp_service.call_tool("get-customers", {"limit": 3})
            print(f"   ✅ get-customers returned:")
            if isinstance(result, dict) and 'content' in result:
                content = result.get('content', [])
                if content and isinstance(content[0], dict) and 'text' in content[0]:
                    text = content[0]['text']
                    print(f"   Response text preview: {text[:300]}...")
                else:
                    print(f"   {json.dumps(result, indent=2)[:500]}...")
            else:
                print(f"   {json.dumps(result, indent=2)[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("END-TO-END TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await mcp_service.stop()
        print("\nMCP service stopped.")


if __name__ == "__main__":
    asyncio.run(test_mcp_end_to_end())
