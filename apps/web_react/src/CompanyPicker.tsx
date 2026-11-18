import { useEffect, useRef, useState } from "react";
import { api } from "./api";

export interface Company { ticker: string; name: string; local?: boolean }
const MAX = 4;

/** Type-ahead over the whole A-share market. The first chip is the primary company; two to four chips make a comparison. */
export default function CompanyPicker({ value, onChange }: { value: Company[]; onChange: (v: Company[]) => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Company[]>([]);
  const [size, setSize] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const timer = useRef<number>(0);

  useEffect(() => {
    window.clearTimeout(timer.current);
    if (!q.trim()) { setHits([]); return; }
    timer.current = window.setTimeout(() => {
      api(`/api/catalog/search?q=${encodeURIComponent(q.trim())}&limit=8`)
        .then((r) => { setHits(r.results); setSize(r.catalog_size); setOpen(true); })
        .catch(() => setHits([]));
    }, 180);
    return () => window.clearTimeout(timer.current);
  }, [q]);

  const add = (c: Company) => {
    if (value.length < MAX && !value.some((v) => v.ticker === c.ticker)) onChange([...value, c]);
    setQ(""); setHits([]); setOpen(false);
  };
  const remove = (t: string) => onChange(value.filter((v) => v.ticker !== t));

  return (
    <div>
      <div className="flex flex-wrap gap-1">
        {value.map((c, i) => (
          <span key={c.ticker} className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-900 dark:bg-blue-950 dark:text-blue-200">
            {i === 0 && value.length > 1 && <b>主</b>}{c.name} {c.ticker}
            <button aria-label={`移除 ${c.name}`} className="text-blue-700 dark:text-blue-300" onClick={() => remove(c.ticker)}>×</button>
          </span>))}
      </div>
      <input className="field mt-1" value={q} placeholder={value.length >= MAX ? "最多 4 家" : "代码 / 简称 / 拼音首字母，如 600519、茅台、gzmt"}
             disabled={value.length >= MAX} onChange={(e) => setQ(e.target.value)} onFocus={() => hits.length && setOpen(true)} />
      {open && hits.length > 0 && (
        <ul className="mt-1 max-h-56 overflow-auto rounded border border-slate-300 bg-white text-sm shadow dark:border-slate-600 dark:bg-slate-900" role="listbox">
          {hits.map((h) => (
            <li key={h.ticker}>
              <button className="flex w-full items-center justify-between px-2 py-1 text-left hover:bg-slate-100 dark:hover:bg-slate-800" onClick={() => add(h)}>
                <span>{h.name} <span className="text-slate-500">{h.ticker}</span></span>
                <span className={`text-[11px] ${h.local ? "text-green-700 dark:text-green-400" : "text-amber-700 dark:text-amber-400"}`}>{h.local ? "已入库" : "在线拉取"}</span>
              </button>
            </li>))}
        </ul>)}
      <p className="mt-1 text-[11px] text-slate-500">
        {size === null ? "" : size ? `全市场 ${size} 家，未入库的公司首次研究时在线拉取。` : "全市场名单尚未缓存：只能搜索已入库的公司。"}
        {value.length > 1 ? " 选了多家即为同业对比。" : ""}
      </p>
    </div>
  );
}
