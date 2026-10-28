import { useRunStore } from '../store/useRunStore';

/**
 * Phones get told the truth: the layout wants a wide screen, and nobody should download
 * 89 MB of weights on mobile data. They still get the full interface, driven by the
 * recorded run, so the page is never a dead end.
 */
export function MobileNotice() {
  const demoMode = useRunStore((state) => state.demoMode);

  return (
    <div className="panel mx-4 mt-4 flex flex-col gap-2 border-amber/40 bg-amber/5 p-4 lg:hidden">
      <p className="text-[13px] font-semibold text-amber">Best on a desktop screen</p>
      <p className="text-[12px] leading-relaxed text-ink-dim">
        The attention grid alone is six layers by twelve heads, so this is built for 1280 px and up.{' '}
        {demoMode
          ? 'You are seeing a recorded run of the real model — every view works, but the prompt is fixed and nothing is downloaded.'
          : 'You are running the real model in this tab.'}
      </p>
      {demoMode && (
        <a className="btn btn-primary w-fit" href="?demo=0">
          Run the real model anyway (89 MB)
        </a>
      )}
    </div>
  );
}
