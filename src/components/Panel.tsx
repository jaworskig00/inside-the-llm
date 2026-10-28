import type { ReactNode } from 'react';
import { useRunStore } from '../store/useRunStore';

interface PanelProps {
  title: string;
  /** One sentence, shown when Explain mode is on. Written for someone who is not an ML engineer. */
  explain: string;
  /** Small stats or controls rendered on the right of the header. */
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}

/**
 * The shell every view sits in. It owns the one-sentence caption, which is how Explain
 * mode works: a single store flag, no per-component plumbing.
 */
export function Panel({ title, explain, aside, children, className = '', id }: PanelProps) {
  const explainOn = useRunStore((state) => state.explain);

  return (
    <section
      className={`panel flex min-w-0 flex-col ${className}`}
      aria-labelledby={id && `${id}-title`}
      // The guided tour anchors its spotlight to these.
      data-tour={id}
    >
      <header className="flex items-start justify-between gap-3 border-b border-edge px-4 py-3">
        <div className="min-w-0">
          <h2
            id={id && `${id}-title`}
            className="text-[13px] font-semibold tracking-tight text-ink"
          >
            {title}
          </h2>
          {explainOn && (
            <p className="mt-1 max-w-prose text-[12px] leading-relaxed text-ink-dim">{explain}</p>
          )}
        </div>
        {aside && <div className="shrink-0 text-right">{aside}</div>}
      </header>
      <div className="min-w-0 flex-1 p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="inline-flex flex-col items-end leading-tight">
      <span className="label">{label}</span>
      <span className="tabular text-[13px] font-medium text-ink">{value}</span>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-[120px] items-center justify-center text-center text-[12px] text-ink-faint">
      {children}
    </div>
  );
}
