import * as echarts from "echarts";
import { useEffect, useRef } from "react";

export function Chart({ option, height = 300 }: { option: any; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || !option) return;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const c = echarts.init(ref.current, dark ? "dark" : undefined);
    c.setOption({ backgroundColor: "transparent", ...option });
    const onResize = () => c.resize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); c.dispose(); };
  }, [option]);
  return <div ref={ref} style={{ height }} />;
}
