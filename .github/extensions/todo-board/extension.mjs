// Extension: todo-board — planning board canvas backed by ToDo.md (single source of truth).
import { createServer } from "node:http";
import { readFileSync, watchFile, unwatchFile } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { joinSession, createCanvas, CanvasError } from "@github/copilot-sdk/extension";
import * as todo from "./todo.mjs";

const here = dirname(fileURLToPath(import.meta.url));
// .github/extensions/todo-board -> repo root
const TODO_PATH = join(here, "..", "..", "..", "ToDo.md");
const HTML = readFileSync(join(here, "board.html"), "utf8");
const servers = new Map(); // instanceId -> { server, url, clients }

/** Board snapshot enriched with per-stream stats and next steps for the UI/agent. */
function snapshot() {
    const board = todo.readBoard(TODO_PATH);
    board.streams = board.streams.filter((s) => !(s.implicit && !s.tasks.length));
    for (const s of board.streams) { s.stats = todo.stats(s); s.next = todo.nextSteps(s); }
    return board;
}

async function readBody(req) {
    let s = "";
    for await (const c of req) s += c;
    return s ? JSON.parse(s) : {};
}

const routes = {
    "/api/status": (b) => todo.setStatus(TODO_PATH, b.line, b.text, b.status),
    "/api/add": (b) => todo.addTask(TODO_PATH, b.stream, b.parentLine, b.text),
    "/api/focus": (b) => todo.setFocus(TODO_PATH, b.stream),
    "/api/stream": (b) => todo.addStream(TODO_PATH, b.name),
};

async function startServer() {
    const clients = new Set();
    const server = createServer(async (req, res) => {
        try {
            if (req.url.startsWith("/api/board")) {
                res.setHeader("Content-Type", "application/json");
                return res.end(JSON.stringify(snapshot()));
            }
            if (req.url.startsWith("/events")) {
                res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
                res.write("retry: 2000\n\n");
                clients.add(res);
                return req.on("close", () => clients.delete(res));
            }
            if (req.method === "POST" && routes[req.url]) {
                res.setHeader("Content-Type", "application/json");
                try { routes[req.url](await readBody(req)); res.end("{}"); }
                catch (e) { res.end(JSON.stringify({ error: String(e.message ?? e) })); }
                return;
            }
            res.setHeader("Content-Type", "text/html; charset=utf-8");
            res.end(HTML);
        } catch (e) {
            res.statusCode = 500; res.end(String(e));
        }
    });
    await new Promise((r) => server.listen(0, "127.0.0.1", r));
    // File watch: edits by the agent or the user in ToDo.md refresh all open boards.
    const onChange = () => clients.forEach((c) => c.write("data: changed\n\n"));
    watchFile(TODO_PATH, { interval: 700 }, onChange);
    return { server, clients, onChange, url: `http://127.0.0.1:${server.address().port}/` };
}

const str = { type: "string" };
const action = (name, description, inputSchema, run) => ({
    name, description, inputSchema,
    handler: async (ctx) => {
        try { return run(ctx.input ?? {}); }
        catch (e) { throw new CanvasError("todo_error", String(e.message ?? e)); }
    },
});

await joinSession({
    canvases: [
        createCanvas({
            id: "todo-board",
            displayName: "Planning board",
            description: "Project planning board over ToDo.md: streams, progress, next steps, task breakdown. ToDo.md is the leading source.",
            actions: [
                action("get_board", "Full board as JSON (streams, task trees with line numbers, stats, next steps, focus).", { type: "object", properties: {} }, () => snapshot()),
                action("summarize", "Compact text summary of progress, blocked items and next steps per stream.", { type: "object", properties: {} }, () => ({ summary: todo.summarize(snapshot()) })),
                action("set_focus", "Mark a stream as the current focus (stored in ToDo.md).", { type: "object", properties: { stream: str }, required: ["stream"] }, (i) => { todo.setFocus(TODO_PATH, i.stream); return { ok: true }; }),
                action("set_status", "Set task status. Use line + exact text from get_board.", { type: "object", properties: { line: { type: "integer" }, text: str, status: { enum: ["todo", "doing", "done", "blocked"] } }, required: ["line", "text", "status"] }, (i) => { todo.setStatus(TODO_PATH, i.line, i.text, i.status); return { ok: true }; }),
                action("add_task", "Add a task to a stream, or a sub-step under parentLine (use for breaking ideas into executable steps).", { type: "object", properties: { stream: str, text: str, parentLine: { type: "integer" } }, required: ["stream", "text"] }, (i) => { todo.addTask(TODO_PATH, i.stream, i.parentLine ?? null, i.text); return { ok: true }; }),
                action("add_stream", "Add a new stream (independent workstream) heading to ToDo.md.", { type: "object", properties: { name: str }, required: ["name"] }, (i) => { todo.addStream(TODO_PATH, i.name); return { ok: true }; }),
            ],
            open: async (ctx) => {
                let entry = servers.get(ctx.instanceId);
                if (!entry) { entry = await startServer(); servers.set(ctx.instanceId, entry); }
                return { title: "Planning board", url: entry.url };
            },
            onClose: async (ctx) => {
                const entry = servers.get(ctx.instanceId);
                if (!entry) return;
                servers.delete(ctx.instanceId);
                unwatchFile(TODO_PATH, entry.onChange);
                entry.clients.forEach((c) => c.end());
                await new Promise((r) => entry.server.close(() => r()));
            },
        }),
    ],
});
