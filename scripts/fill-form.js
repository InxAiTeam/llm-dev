/**
 * fill-form.js
 *
 * Uses GitHub Copilot LLM Models (via GitHub Models API) as the AI "brain"
 * and MCP Playwright as the browser automation "hands" to fill a sample web form.
 *
 * Architecture:
 *   GitHub Copilot LLM  ←→  MCP Client  ←→  MCP Playwright Server  →  Browser
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import OpenAI from 'openai';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');

// ─── Configuration ────────────────────────────────────────────────────────────

const PORT = 3000;
const SCREENSHOTS_DIR = path.join(rootDir, 'screenshots');
const MODEL = process.env.COPILOT_MODEL ?? 'gpt-4o';
const MAX_ITERATIONS = 30;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function log(icon, ...args) {
  console.log(icon, ...args);
}

// ─── 1. Start local HTTP server serving the sample form ───────────────────────

mkdirSync(SCREENSHOTS_DIR, { recursive: true });

const htmlContent = readFileSync(path.join(rootDir, 'sample', 'index.html'), 'utf-8');
const httpServer = createServer((_req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(htmlContent);
});

await new Promise((resolve) => httpServer.listen(PORT, '127.0.0.1', resolve));
log('🌐', `Local HTTP server started at http://localhost:${PORT}`);

// ─── 2. Connect to MCP Playwright server via stdio transport ──────────────────

const transport = new StdioClientTransport({
  command: 'npx',
  args: [
    '-y',
    '@playwright/mcp',
    '--browser', 'chromium',
    '--headless',
    '--output-dir', SCREENSHOTS_DIR,
  ],
  env: { ...process.env },
});

const mcpClient = new Client(
  { name: 'copilot-form-filler', version: '1.0.0' },
  { capabilities: {} },
);

log('🔌', 'Connecting to MCP Playwright server…');
await mcpClient.connect(transport);
log('✅', 'Connected to MCP Playwright server');

// ─── 3. Retrieve available MCP tools ──────────────────────────────────────────

const { tools } = await mcpClient.listTools();
log('🛠 ', `Available MCP tools (${tools.length}):`, tools.map((t) => t.name).join(', '));

// ─── 4. Build OpenAI client pointing at GitHub Models API ─────────────────────

if (!process.env.GITHUB_TOKEN) {
  throw new Error('GITHUB_TOKEN environment variable is required');
}

const openaiClient = new OpenAI({
  baseURL: 'https://models.inference.ai.azure.com',
  apiKey: process.env.GITHUB_TOKEN,
});

// Convert MCP tool schemas to OpenAI function-calling format
const openaiTools = tools.map((tool) => ({
  type: /** @type {'function'} */ ('function'),
  function: {
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
  },
}));

// ─── 5. Run the agentic loop ──────────────────────────────────────────────────

/** @type {Array<import('openai/resources/chat/completions').ChatCompletionMessageParam>} */
const messages = [
  {
    role: 'system',
    content: [
      'You are an AI assistant that controls a web browser using Playwright tools.',
      'Complete the task step-by-step. Use the browser tools to navigate, interact with',
      'elements, and verify the result. Always take a screenshot at the very end.',
    ].join(' '),
  },
  {
    role: 'user',
    content: `Please fill in the contact form at http://localhost:${PORT} with the following details and submit it:

- 姓名 (Name): John Doe
- 電子郵件 (Email): john.doe@example.com
- 電話 (Phone): 0912-345-678
- 主旨 (Subject): 技術支援
- 訊息 (Message): Hello! This form was automatically filled by GitHub Copilot LLM and MCP Playwright.

Steps:
1. Navigate to http://localhost:${PORT}
2. Fill in every field listed above
3. Click the submit button
4. Verify the success message is shown
5. Take a final screenshot`,
  },
];

log('\n🤖', 'Starting GitHub Copilot agent loop…\n');

let iteration = 0;

while (iteration < MAX_ITERATIONS) {
  iteration++;
  log('─'.repeat(50));
  log('🔄', `Iteration ${iteration}/${MAX_ITERATIONS}`);

  const response = await openaiClient.chat.completions.create({
    model: MODEL,
    messages,
    tools: openaiTools,
    tool_choice: 'auto',
  });

  const choice = response.choices[0];
  const message = choice.message;
  messages.push(message);

  if (message.content) {
    log('💬', `Agent: ${message.content}`);
  }

  // No more tool calls → the agent has finished
  if (!message.tool_calls || message.tool_calls.length === 0) {
    log('\n✅', 'Agent completed the task!');
    break;
  }

  // Execute each requested tool call via MCP
  for (const toolCall of message.tool_calls) {
    const toolName = toolCall.function.name;
    let toolArgs;
    try {
      toolArgs = JSON.parse(toolCall.function.arguments);
    } catch {
      toolArgs = {};
    }

    log('🔧', `Calling tool: ${toolName}`);
    if (Object.keys(toolArgs).length > 0) {
      log('   Args:', JSON.stringify(toolArgs, null, 2).split('\n').join('\n        '));
    }

    let resultContent;
    try {
      const result = await mcpClient.callTool({ name: toolName, arguments: toolArgs });
      resultContent = result.content;

      // Print a short preview of the result
      const preview = JSON.stringify(resultContent);
      log('   ✓ Result:', preview.length > 200 ? preview.slice(0, 197) + '…' : preview);
    } catch (err) {
      log('   ❌ Tool error:', err.message);
      resultContent = [{ type: 'text', text: `Error: ${err.message}` }];
    }

    messages.push({
      role: 'tool',
      tool_call_id: toolCall.id,
      content: JSON.stringify(resultContent),
    });
  }
}

if (iteration >= MAX_ITERATIONS) {
  log('⚠️ ', 'Reached maximum iterations without completing the task');
}

// ─── 6. Cleanup ───────────────────────────────────────────────────────────────

log('\n🧹', 'Cleaning up…');
await mcpClient.close();
httpServer.close();

log('🎉', `Done! Screenshots saved to: ${SCREENSHOTS_DIR}`);
