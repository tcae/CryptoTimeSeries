// ToDo.md parsing and line-based editing. ToDo.md is the single source of truth;
// every mutation rewrites only the affected lines so free-form notes survive.
import { readFileSync, writeFileSync } from "node:fs";

export const STATUS_MARK = { todo: " ", doing: "~", done: "x", blocked: "!" };
const MARK_STATUS = { " ": "todo", "~": "doing", x: "done", X: "done", "!": "blocked" };
const HEADING = /^(#{1,6})\s+(.*?)\s*$/;
const BULLET = /^(\s*)[-*]\s+(?:\[([ ~xX!])\]\s+)?(.*)$/;
const FOCUS = /^<!--\s*focus:\s*(.*?)\s*-->$/;

const indentWidth = (s) => [...s].reduce((n, c) => n + (c === "\t" ? 4 : 1), 0);

/** Parse ToDo.md text into streams (headings) with nested task trees. */
export function parse(text) {
    const lines = text.split("\n");
    const streams = [];
    let focus = null;
    let stream = null;
    let stack = []; // [{indent, task}]
    const ensureStream = (line) => {
        if (!stream) {
            stream = { name: "General", line, level: 1, tasks: [], notes: [], implicit: true };
            streams.push(stream);
        }
    };
    lines.forEach((raw, i) => {
        const f = FOCUS.exec(raw.trim());
        if (f) { focus = f[1]; return; }
        const h = HEADING.exec(raw);
        if (h) {
            stream = { name: h[2], line: i, level: h[1].length, tasks: [], notes: [] };
            streams.push(stream);
            stack = [];
            return;
        }
        const b = BULLET.exec(raw);
        if (b) {
            ensureStream(i);
            const indent = indentWidth(b[1]);
            while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
            const task = {
                line: i, end: i, text: b[3], status: MARK_STATUS[b[2] ?? " "],
                marked: b[2] !== undefined, children: [], notes: [],
            };
            (stack.length ? stack[stack.length - 1].task.children : stream.tasks).push(task);
            stack.push({ indent, task });
            return;
        }
        if (raw.trim()) {
            if (stack.length) stack[stack.length - 1].task.notes.push(raw.trim());
            else if (stream) stream.notes.push(raw.trim());
        }
    });
    // Mark each task's last owned line (descendants included) for inserts.
    const setEnd = (t) => { t.children.forEach(setEnd); t.end = Math.max(t.line, ...t.children.map((c) => c.end)); };
    streams.forEach((s) => s.tasks.forEach(setEnd));
    return { streams, focus, lineCount: lines.length };
}

const walk = (tasks, fn) => tasks.forEach((t) => { fn(t); walk(t.children, fn); });

/** Progress counts for a stream (leaf tasks only; parents are derived). */
export function stats(stream) {
    const c = { todo: 0, doing: 0, done: 0, blocked: 0 };
    walk(stream.tasks, (t) => { if (!t.children.length) c[t.status]++; });
    const total = c.todo + c.doing + c.done + c.blocked;
    return { ...c, total, pct: total ? Math.round((100 * c.done) / total) : 0 };
}

/** Next actionable leaf tasks: in-progress first, then open ones, in file order. */
export function nextSteps(stream, limit = 3) {
    const doing = [], open = [];
    walk(stream.tasks, (t) => {
        if (t.children.length) return;
        if (t.status === "doing") doing.push(t);
        else if (t.status === "todo") open.push(t);
    });
    return [...doing, ...open].slice(0, limit);
}

export function summarize(board) {
    const out = [];
    if (board.focus) out.push(`Focus: ${board.focus}`);
    for (const s of board.streams) {
        const st = stats(s);
        if (!st.total) continue;
        out.push(`## ${s.name} — ${st.pct}% (${st.done}/${st.total} done, ${st.doing} doing, ${st.blocked} blocked)`);
        const blocked = []; walk(s.tasks, (t) => { if (t.status === "blocked") blocked.push(t); });
        blocked.forEach((t) => out.push(`  blocked: ${t.text}`));
        nextSteps(s).forEach((t) => out.push(`  next: ${t.text}`));
    }
    return out.join("\n");
}

export const readBoard = (path) => parse(readFileSync(path, "utf8"));

function edit(path, fn) {
    const lines = readFileSync(path, "utf8").split("\n");
    fn(lines);
    writeFileSync(path, lines.join("\n"));
}

function assertLine(lines, line, text) {
    if (!(line >= 0 && line < lines.length) || !lines[line].includes(text)) {
        throw new Error(`ToDo.md changed: line ${line} no longer contains "${text}". Reload the board.`);
    }
}

/** Set a task's status marker, converting an unmarked bullet to a checkbox bullet. */
export function setStatus(path, line, text, status) {
    edit(path, (lines) => {
        assertLine(lines, line, text);
        const m = BULLET.exec(lines[line]);
        if (!m) throw new Error(`Line ${line} is not a bullet`);
        const dash = lines[line].match(/^(\s*[-*]\s+)/)[1];
        lines[line] = `${dash}[${STATUS_MARK[status]}] ${m[3]}`;
    });
}

/** Insert a subtask under parentLine (or at the end of the stream when parentLine is null). */
export function addTask(path, streamName, parentLine, text) {
    const board = readBoard(path);
    const stream = board.streams.find((s) => s.name === streamName);
    if (!stream) throw new Error(`Unknown stream "${streamName}"`);
    edit(path, (lines) => {
        let at, indent = "";
        if (parentLine != null) {
            let parent; walk(stream.tasks, (t) => { if (t.line === parentLine) parent = t; });
            if (!parent) throw new Error(`No task at line ${parentLine} in "${streamName}"`);
            const lead = BULLET.exec(lines[parentLine])[1];
            indent = lead + (lead.includes("\t") ? "\t" : "  ");
            at = parent.end + 1;
        } else {
            const idx = board.streams.indexOf(stream);
            const next = board.streams[idx + 1];
            let end = next ? next.line : lines.length;
            while (end > stream.line + 1 && !lines[end - 1].trim()) end--;
            at = end;
        }
        lines.splice(at, 0, `${indent}- [ ] ${text}`);
    });
}

export function addStream(path, name) {
    edit(path, (lines) => {
        while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
        lines.push("", `## ${name}`, "");
    });
}

/** Persist the focused stream as an HTML comment so it survives sessions and is visible in the file. */
export function setFocus(path, name) {
    edit(path, (lines) => {
        const i = lines.findIndex((l) => FOCUS.test(l.trim()));
        const row = `<!-- focus: ${name} -->`;
        if (i >= 0) lines[i] = row; else lines.unshift(row);
    });
}
