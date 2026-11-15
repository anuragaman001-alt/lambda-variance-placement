// ---------------------------------------------------------------------------
// index.mjs  (Node.js 20 version of the test function)
//
// Same three tasks as the Python version (lambda_function.py), written in
// JavaScript, so Python and Node.js can be compared doing the same work.
// The WORKLOAD environment variable (set by deploy.py) picks the task.
// ---------------------------------------------------------------------------
import fs from "fs";          // file system, for the io task
import crypto from "crypto";  // used to make random bytes

function cpu() {
  // CPU-heavy task: 1 million multiplications and remainders.
  let s = 0;
  for (let i = 0; i < 1000000; i++) s += (i * i) % 7;
  return s;
}

function memory() {
  // Memory-heavy task: reserve 32 MB, then write one byte in every 4 KB
  // page so the system really has to hand over all of the memory.
  const b = Buffer.alloc(32 * 1024 * 1024);
  for (let i = 0; i < b.length; i += 4096) b[i] = 1;
  return b.length;
}

function ioTask() {
  // Disk-heavy task: write 5 MB of random bytes to /tmp, then read it back.
  const data = crypto.randomBytes(5 * 1024 * 1024);
  fs.writeFileSync("/tmp/f.bin", data);
  return fs.readFileSync("/tmp/f.bin").length;
}

// Map the WORKLOAD text to the matching function.
const TASKS = { cpu, memory, io: ioTask };

// Lambda calls "handler" on every invocation.
export const handler = async () => {
  const w = process.env.WORKLOAD || "cpu";   // which task to run
  const t0 = performance.now();               // start timer
  const r = TASKS[w]();                       // do the work
  // Small answer back; the real timings come from Lambda's REPORT log line.
  return { workload: w, result: r, compute_ms: performance.now() - t0 };
};
