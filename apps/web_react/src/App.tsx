import { useEffect, useState } from "react";
import DataHealth from "./DataHealth";
import Evaluation from "./Evaluation";
import Research from "./Research";

const VIEWS = [
  { id: "research", label: "投研工作台" },
  { id: "evaluation", label: "评测中台" },
  { id: "data-health", label: "数据质量" },
] as const;
type View = (typeof VIEWS)[number]["id"];

const fromHash = (): View => (VIEWS.find((v) => "#" + v.id === location.hash)?.id ?? "research");

export default function App() {
  const [view, setView] = useState<View>(fromHash);
  const [openRun, setOpenRun] = useState<string | null>(null);
  useEffect(() => {
    const f = () => setView(fromHash());
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  const go = (v: View) => { location.hash = v; setView(v); };
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="no-print flex items-center gap-6 border-b border-slate-200 bg-white px-4 py-2.5 dark:border-slate-800 dark:bg-slate-900">
        <b>FinResearch Agent</b>
        <nav className="flex gap-4 text-sm">
          {VIEWS.map((v) => (
            <a key={v.id} href={"#" + v.id} onClick={(e) => { e.preventDefault(); go(v.id); }}
               className={view === v.id ? "font-semibold text-blue-600 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"}>
              {v.label}
            </a>
          ))}
        </nav>
        <span className="ml-auto text-xs text-amber-700 dark:text-amber-400">演示数据：6 家虚构公司（离线 Golden Snapshot）</span>
      </header>
      {view === "research" && <Research openRun={openRun} />}
      {view === "evaluation" && <Evaluation onReplay={(r) => { setOpenRun(r); go("research"); }} />}
      {view === "data-health" && <DataHealth />}
    </div>
  );
}
