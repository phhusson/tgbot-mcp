#!/usr/bin/env python

import telethon
import json
import asyncio
from aioconsole import aprint
import os
import aiohttp
import sys
import base64
import tempfile

from datetime import datetime
import io
import ast
import yaml

import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


backend = 'llamacpp'

async def llamacpp_complete(system, user):
    server = os.environ['LLAMACPP_SERVER']
    # Cut at last / to get the server name
    server = server[:server.rfind('/')]
    url = f"{server}/apply-template"
    print("url is ", url)
    data = {
        'messages': [
            {"role":"system", "content":system},
            {"role":"user", "content":user}
        ]
    }
    headers = {'Content-Type': 'application/json'}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=json.dumps(data), headers=headers) as response:
            prompt = await response.json()
    prompt = prompt['prompt']
    
    data = {
        'stream': False,
        'cache_prompt': True,
        'stop': ['<|im_end|>']
    }

    headers = {'Content-Type': 'application/json'}
    data['prompt'] = prompt
    async with aiohttp.ClientSession() as session:
        async with session.post(os.environ['LLAMACPP_SERVER'], data=json.dumps(data), headers=headers) as response:
            response_text = await response.text()
            return json.loads(response_text)['content']

# Create a function that continues the request and make "prompt" bigger to retain context
async def continue_prompt(system, user, max_tokens=512):
    if backend == 'llamacpp':
        content = await llamacpp_complete(system, user)
    else:
        raise ValueError(f'{backend} is not in the list of supported backends')

    print("Returning ", content)
    return content

functions = {}
module_descriptions = []

async def ast_run(node, out):
    if isinstance(node, ast.Module):
        if len(node.body) == 0:
            return None
        return await ast_run(node.body[0].value, out)
    elif isinstance(node, ast.Name):
        print("Received name", node.id)
        return node.id
    elif isinstance(node, ast.Attribute):
        # this is xxx.yyy, just dumbly convert it to string assuming both operands are strings
        return f"{node.value.id}.{node.attr}"
    elif isinstance(node, ast.Call):
        func = await ast_run(node.func, out)
        obj = {
            "function": func,
            "args": [await ast_run(arg, out) for arg in node.args],
            "keywords": {kw.arg: await ast_run(kw.value, out) for kw in node.keywords},
        }
                
        if func == "say":
            out['say'] = obj['args'][0]

        if not func in functions:
            print(f"Function {func} not found", functions.keys())
            return None
        return {"result": await functions[func](*obj['args'], **obj['keywords'])}
    elif isinstance(node, ast.List):
        return [await ast_run(item, out) for item in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(await ast_run(item, out) for item in node.elts)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = await ast_run(node.left, out)
        right = await ast_run(node.right, out)
        if isinstance(node.op, ast.Add, out):
            return left + right
        elif isinstance(node.op, ast.Sub, out):
            return left - right
        elif isinstance(node.op, ast.Mult, out):
            return left * right
        elif isinstance(node.op, ast.Div, out):
            return left / right
        else:
            print("Received unknown binop", node.op)
    else:
        print("Received unknown node", node)
        return None

async def pseudo_py_run(s, out):
    tree = ast.parse(s.strip())
    res = await ast_run(tree, out)
    return res

async def complete(client: telethon.TelegramClient, event, msg, prompt):
    prompt = prompt + msg
    ret = await continue_prompt(prompt, msg)
    await event.reply(f"{ret}")
    
    reply = ""
    for l in ret.split("\n"):
        print("Parsing", l)
        try:
            out = {}
            py = await pseudo_py_run(l, out)
            if py:
                print("py is ", py)
                if 'type' in py['result']:
                    if py['result']['type'] == 'text':
                        reply += py['result']['text'] + "\n"
                    elif py['result']['type'] == 'image':
                        # Store py['resut']['image'] in a temporary file named .jpg and send it
                        f = tempfile.NamedTemporaryFile(suffix=".jpg")
                        f.write(py['result']['image'].getbuffer())
                        f.seek(0)
                        await client.send_file(event.chat_id, f)
            if 'say' in out:
                reply += out['say'] + "\n"
        except SyntaxError as e:
            reply = "Invalid syntax"
            break

    if reply:
        await event.reply(reply)


async def get_prompt():
    prompt = f"""
You are a telegram bot, whose function is to do what your user tells you to do.
You answer with function calls.
Example:
say("Hello, world!")

Current time is {datetime.now().strftime('%Y-%m-%d %H:%S')}.

Available functions:
{"\n\n".join(module_descriptions)}
"""

    prompt += "\n\n"

    print("Returning prompt", prompt)

    return prompt


async def watch_myself():
    last_modified_time = os.path.getmtime(__file__)

    while True:
        await asyncio.sleep(1)  # Poll every second
        current_modified_time = os.path.getmtime(__file__)

        if current_modified_time != last_modified_time:
            print("File changed. Restarting program...")
            os.execl(sys.executable, sys.executable, *sys.argv)
        last_modified_time = current_modified_time

def create_tool_function(sn: str, session: ClientSession, tool: mcp.types.Tool):
    async def tool_function(*args, **kwargs):
        tool_name = f"{sn}.{tool.name}"
        print(f"Calling tool {tool_name} with args (ignored) {args} and kwargs {kwargs}")
        # We might have to convert integers to strings, check the schema
        for key in kwargs:
            if key in tool.inputSchema['properties']:
                if tool.inputSchema['properties'][key]['type'] == 'string':
                    kwargs[key] = str(kwargs[key])

        result = await session.call_tool(tool.name, kwargs)


        if result.isError:
            print(f"Error calling {tool_name}: {result.content}")
            return f"Error calling {tool_name}: {result.content}"
        content = result.content

        if len(content) != 1:
            print("Unknown content", content)
            return None
        content = content[0]

        if content.type == 'image':
            print("Got image", len(content.data))
        else:
            print(f"Got result {result}")

        if content.type == 'text':
            return {"text":content.text, "type":"text"}
        elif content.type == 'image':
            # debase64 content.data
            data = base64.b64decode(content.data)
            data = io.BytesIO(data)

            return {"image": data, "type":"image"}
        else:
            print("unknown content type", content.type)
            return None
    return tool_function

async def connect_to_mcp_server(serverParameters: StdioServerParameters, server_config: dict):
    async with stdio_client(serverParameters) as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print("Initialized session", init_result)
            sn = init_result.serverInfo.name
            sn = sn.replace(" ", "_")
            sn = sn.replace("-", "_")
            descr = f"Module {sn}:\n"
            if 'additional_prompt' in server_config:
                descr += server_config['additional_prompt'] + "\n"
            print("Connected session to", init_result.serverInfo.name)
            prompts = None
            resources = None
            tools = None
            if init_result.capabilities.prompts:
                prompts = await session.list_prompts()
                print("Got prompts", prompts)
            if init_result.capabilities.resources:
                resources = await session.list_resources()
                print("Got resources", resources)
            if init_result.capabilities.tools:
                tools = await session.list_tools()
                for tool in tools.tools:
                    print("Got tool", tool)
                    if 'allowlist' in server_config:
                        if not tool.name in server_config['allowlist']:
                            print("Skipping")
                            continue

                    toolName = f"{sn}.{tool.name}"
                    functions[toolName] = create_tool_function(sn, session, tool)
                    print(tool.inputSchema)
                    argsDesc = ""
                    args = []
                    props = tool.inputSchema['properties']
                    for arg in props:
                        # I'm lazy so for the moment only list required arguments
                        if 'required' in tool.inputSchema and arg in tool.inputSchema['required']:
                            if not argsDesc:
                                argsDesc = "-- "
                            p = props[arg]
                            if 'description' in p:
                                argsDesc += f"{arg}: {p['type']} ({p['description']}); "
                            else:
                                argsDesc += f"{arg}: {p['type']}; "
                            args.append(arg)
                    args = [f"{arg}=..." for arg in args]
                    args = ", ".join(args)
                    descr += f"{toolName}({args}) {tool.description} {argsDesc}"
                    module_descriptions.append(descr)
            server_sessions[init_result.serverInfo.name] = session
            # Loop forever to keep the connection alive
            while True:
                await asyncio.sleep(1)


server_sessions = {}

async def main():
    asyncio.create_task(watch_myself())

    with open('secret.yaml') as f:
        secret = yaml.safe_load(f)
    api_id = secret['telegram']['api_id']
    api_hash = secret['telegram']['api_hash']
    bot = secret['telegram']['bot']


    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    owner = config['owner']
    mode = config['mode']
    if mode != 'owner_only':
        raise ValueError("Only owner mode is supported")
    mcp_servers = config['mcp_servers']
    for server in mcp_servers:
        server = mcp_servers[server]
        print("Connecting to", server)
        command = server['command']
        env = os.environ.copy()
        if 'env' in server:
            env = server['env']
        server_config = StdioServerParameters(command=command[0], args = command[1:], env = env)
        asyncio.create_task(connect_to_mcp_server(server_config, server))

    client = await telethon.TelegramClient('bot', api_id, api_hash).start(bot_token=bot)
    async with client:
        me = await client.get_me()
        print("I am", me)
        @client.on(telethon.events.NewMessage)
        async def new_msg(event):
            global last_conversation
            await aprint("Received event", event)
            if event.message.peer_id.user_id != owner:
                return
            
            await complete(client, event, event.message.message, await get_prompt())

        await client.run_until_disconnected()

asyncio.run(main())
