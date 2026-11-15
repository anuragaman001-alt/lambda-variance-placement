import fs from "fs";
import crypto from "crypto";

function cpu() {
  let s = 0;
  for (let i = 0; i < 1000000; i++) s += (i * i) % 7;
  return s;
}

function memory() {
  const b = Buffer.alloc(32 * 1024 * 1024);           // 32 MB
  for (let i = 0; i < b.length; i += 4096) b[i] = 1;  // touch every page
  return b.length;
}

function ioTask() {
  const data = crypto.randomBytes(5 * 1024 * 1024);    // 5 MB
  fs.writeFileSync("/tmp/f.bin", data);
  return fs.readFileSync("/tmp/f.bin").length;
}

const TASKS = { cpu, memory, io: ioTask };

export const handler = async () => {
  const w = process.env.WORKLOAD || "cpu";
  const t0 = performance.now();
  const r = TASKS[w]();
  return { workload: w, result: r, compute_ms: performance.now() - t0 };
};
