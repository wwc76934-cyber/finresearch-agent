// Record the end-to-end demo against a running server (real API, no fixtures).
//   node scripts/record_demo.mjs http://127.0.0.1:8000 reports/demo
// Uses the installed Chrome/Edge via Playwright channels (no browser download).
import { mkdirSync, readdirSync, renameSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(join(process.cwd(), "apps/web_react/package.json"));
const { chromium } = require("playwright");
const [, , base = "http://127.0.0.1:8000", out = "reports/demo"] = process.argv;
mkdirSync(out, { recursive: true });

const pause = (ms) => new Promise((r) => setTimeout(r, ms));
let browser;
for (const channel of ["chrome", "msedge"]) {
  try { browser = await chromium.launch({ channel, headless: true }); break; } catch { /* try next */ }
}
if (!browser) throw new Error("no Chrome or Edge found");
const ctx = await browser.newContext({ viewport: { width: 1440, height: 860 }, recordVideo: { dir: out, size: { width: 1440, height: 860 } } });
const page = await ctx.newPage();
const step = (s) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${s}`);

step("open workspace");
await page.goto(base + "/");
await page.getByRole("button", { name: "开始研究" }).waitFor();
await page.locator("select").first().selectOption("990001.SZ");
await page.locator('input[type="number"]').first().fill("1000");   // small budget -> approval card
await pause(800);
step("start research");
await page.getByRole("button", { name: "开始研究" }).click();

step("wait for approval card");
const card = page.getByRole("dialog", { name: "需要你的审批" });
await card.waitFor({ timeout: 60000 });
await pause(1800);
await card.getByRole("button", { name: "批准", exact: true }).click();

step("wait for report");
await page.getByText("关键争议与证伪条件", { exact: true }).waitFor({ timeout: 120000 });
await pause(1500);
const right = page.locator("section").nth(2);
await right.evaluate((el) => el.scrollBy({ top: 900, behavior: "smooth" }));
await pause(1500);

step("open evidence drawer for an investigation claim");
const badge = page.locator("div.my-1", { hasText: "追查[" }).first().locator("button").first();
await badge.scrollIntoViewIfNeeded();
await badge.click();
await page.getByText("血缘链", { exact: true }).waitFor();
await pause(2500);
await page.getByRole("button", { name: "关闭" }).click();
await pause(600);

step("evaluation dashboard");
await page.getByRole("link", { name: "评测中台" }).click();
await page.getByText("B0–B4 基线对比", { exact: true }).waitFor();
await pause(2500);
await page.mouse.wheel(0, 700);
await pause(1500);

step("data health");
await page.getByRole("link", { name: "数据质量" }).click();
await page.getByText("DataSnapshotManifest", { exact: true }).waitFor();
await pause(2000);

await ctx.close();
await browser.close();
const webm = readdirSync(out).filter((f) => f.endsWith(".webm")).map((f) => join(out, f));
const final = join(out, "demo.webm");
renameSync(webm[webm.length - 1], final);
console.log("WEBM " + final);
