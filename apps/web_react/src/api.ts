export async function api<T = any>(path: string, opt?: RequestInit): Promise<T> {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const post = <T = any>(path: string, body: unknown) =>
  api<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");

export interface Claim {
  claim_id: string; section: string; claim_type: "disclosed_fact" | "derived_metric" | "model_inference";
  text: string; evidence_ids: string[]; metric_ids: string[]; status: string; produced_by: string;
}
export interface Evidence {
  evidence_id: string; kind: string; source_ref: string; summary: string; excerpt: string;
  available_at: string; metric_ids: string[]; document_id?: string; page?: string;
}
export interface Span {
  node: string; seq: number; latency_ms?: number; tool_calls?: any[] | number; input_tokens?: number;
  output_tokens?: number; tokens?: number; summary?: string; status?: string;
}
export interface Report {
  run_id: string; architecture: string; ticker: string; name: string; as_of: string; snapshot_id: string;
  overview: string; executive_summary: string; sections: { key: string; title: string; claim_ids: string[] }[];
  claims: Record<string, Claim>; evidence: Record<string, Evidence>; metrics: Record<string, any>;
  open_questions: { topic: string; reason: string }[]; uncertainties: { text: string }[];
  conflicts: { metric: string; period: string; chosen: string }[]; debate: any; risk: any; charts: Record<string, any>;
  plan: { subtasks: { agent: string; objective: string }[]; hypotheses?: any[] } | null; trace: Span[]; totals: any;
  comparison?: { tickers: { ticker: string; name: string; template: string }[]; period: string | null;
    rows: { dimension: string; metric: string; values: Record<string, string> }[]; warnings: string[] } | null;
  thesis_review?: { items: any[]; theses?: any[]; notes?: string[] } | null; live_ingestion?: any; data_mode?: string;
}
export interface Approval {
  reason: string; used_tokens: number; token_budget: number; proposed: { code: string; message: string; query: string }[];
}
