import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useRunStore } from '../store/useRunStore';
import { TOUR_STEPS, markTourSeen, shouldAutoStartTour } from '../lib/tour';

interface TourProps {
  /** Runs a generation, so the first step can show the thing working. */
  onGenerate: () => void;
}

interface Spot {
  top: number;
  left: number;
  width: number;
  height: number;
}

const PADDING = 8;
const CARD_WIDTH = 380;
const GAP = 14;

/**
 * The first-run walkthrough: a spotlight on one element at a time with a short explanation.
 *
 * Hand-rolled rather than pulled from a library — it is one overlay, one measured rectangle
 * and a card, and a dependency would have been larger than the component. The spotlight is
 * a single div with an enormous box-shadow, which dims everything outside it in one paint
 * and needs no SVG mask or four-rectangle gymnastics.
 */
export function Tour({ onGenerate }: TourProps) {
  const open = useRunStore((state) => state.tourOpen);
  const step = useRunStore((state) => state.tourStep);
  const phase = useRunStore((state) => state.phase);
  const hasRun = useRunStore((state) => state.tokens.length > 0);
  const startTour = useRunStore((state) => state.startTour);
  const endTour = useRunStore((state) => state.endTour);
  const goToTourStep = useRunStore((state) => state.goToTourStep);

  const [spot, setSpot] = useState<Spot | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const autoStarted = useRef(false);

  const current = TOUR_STEPS[Math.min(step, TOUR_STEPS.length - 1)];
  const isLast = step >= TOUR_STEPS.length - 1;

  // Opens itself once the app is usable - not over the loading screen.
  useEffect(() => {
    if (autoStarted.current || phase !== 'ready') return;
    autoStarted.current = true;
    if (shouldAutoStartTour(window.location.search)) startTour();
  }, [phase, startTour]);

  const finish = useCallback(() => {
    markTourSeen();
    endTour();
  }, [endTour]);

  const next = useCallback(() => {
    if (isLast) finish();
    else goToTourStep(step + 1);
  }, [isLast, finish, goToTourStep, step]);

  const back = useCallback(() => goToTourStep(step - 1), [goToTourStep, step]);

  // Measure the spotlit element, and keep measuring while the page moves under it.
  useLayoutEffect(() => {
    if (!open || !current) return;

    const element = document.querySelector<HTMLElement>(`[data-tour="${current.target}"]`);

    const measure = () => {
      // Nothing to point at: the card centres itself rather than spotlighting empty space.
      if (!element) {
        setSpot(null);
        return;
      }
      const rect = element.getBoundingClientRect();
      setSpot({
        top: rect.top - PADDING,
        left: rect.left - PADDING,
        width: rect.width + PADDING * 2,
        height: rect.height + PADDING * 2,
      });
    };

    measure();
    if (!element) return;

    element.scrollIntoView({ block: 'center', behavior: 'smooth' });

    // Re-measure after the smooth scroll has settled.
    const timer = window.setTimeout(measure, 320);
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    window.addEventListener('scroll', measure, true);
    window.addEventListener('resize', measure);

    return () => {
      window.clearTimeout(timer);
      observer.disconnect();
      window.removeEventListener('scroll', measure, true);
      window.removeEventListener('resize', measure);
    };
  }, [open, current, hasRun]);

  useEffect(() => {
    if (!open) return;
    cardRef.current?.focus();
  }, [open, step]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        finish();
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        next();
      } else if (event.key === 'ArrowLeft' && step > 0) {
        event.preventDefault();
        back();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, finish, next, back, step]);

  if (!open || !current) return null;

  const card = placeCard(spot);

  return (
    <div className="fixed inset-0 z-[60]" role="presentation">
      {/* One div, one huge shadow: everything outside the rectangle dims in a single paint. */}
      {spot && (
        <div
          className="pointer-events-none absolute rounded-xl ring-2 ring-accent transition-all duration-300"
          style={{
            top: spot.top,
            left: spot.left,
            width: spot.width,
            height: spot.height,
            boxShadow: '0 0 0 9999px rgba(4, 5, 8, 0.74)',
          }}
        />
      )}
      {!spot && <div className="absolute inset-0 bg-void/80" />}

      {/* Clicking the dimmed area leaves the tour, the way a dialog backdrop should. */}
      <button
        type="button"
        className="absolute inset-0 h-full w-full cursor-default"
        aria-label="Close the walkthrough"
        onClick={finish}
      />

      <div
        ref={cardRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-title"
        aria-describedby="tour-body"
        className="panel absolute w-[min(380px,calc(100vw-32px))] p-4 shadow-2xl outline-none"
        style={card}
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <span className="label">
            {step + 1} of {TOUR_STEPS.length}
          </span>
          <div className="flex gap-1" aria-hidden="true">
            {TOUR_STEPS.map((entry, index) => (
              <span
                key={entry.id}
                className={`h-1 w-5 rounded-full ${index <= step ? 'bg-accent' : 'bg-edge-bright'}`}
              />
            ))}
          </div>
        </div>

        <h2 id="tour-title" className="text-[14px] font-semibold tracking-tight">
          {current.title}
        </h2>
        <p id="tour-body" className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">
          {current.body}
        </p>

        <div className="mt-4 flex items-center justify-between gap-2">
          <button type="button" className="btn px-2.5 py-1 text-[11.5px]" onClick={finish}>
            Skip
          </button>
          <div className="flex gap-2">
            {step > 0 && (
              <button type="button" className="btn px-3 py-1.5 text-[12px]" onClick={back}>
                Back
              </button>
            )}
            {current.action === 'generate' && !hasRun ? (
              <button
                type="button"
                className="btn btn-primary px-3 py-1.5 text-[12px]"
                onClick={() => {
                  onGenerate();
                  next();
                }}
              >
                Run it for me →
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-primary px-3 py-1.5 text-[12px]"
                onClick={next}
              >
                {isLast ? 'Start exploring' : 'Next →'}
              </button>
            )}
          </div>
        </div>

        <p className="mt-3 text-[10.5px] text-ink-faint">← → to move · Esc to leave</p>
      </div>
    </div>
  );
}

/**
 * Puts the card under the spotlight, or above it when there is no room below, and keeps it
 * inside the viewport either way.
 */
function placeCard(spot: Spot | null): { top: number; left: number } {
  const viewportWidth = typeof window === 'undefined' ? 1280 : window.innerWidth;
  const viewportHeight = typeof window === 'undefined' ? 800 : window.innerHeight;

  if (!spot) {
    return {
      top: Math.round(viewportHeight / 2 - 120),
      left: Math.round(viewportWidth / 2 - CARD_WIDTH / 2),
    };
  }

  const estimatedHeight = 260;
  const below = spot.top + spot.height + GAP;
  const above = spot.top - estimatedHeight - GAP;

  let top: number;
  if (below + estimatedHeight < viewportHeight) {
    top = below;
  } else if (above > GAP) {
    top = above;
  } else {
    // The target is taller than the viewport - the attention grid usually is. Pinning the
    // card to the bottom edge covers the least of what it is describing; dropping it at the
    // top would sit squarely over the head grid.
    top = viewportHeight - estimatedHeight - GAP;
  }

  const preferred = spot.left + spot.width / 2 - CARD_WIDTH / 2;
  const left = Math.min(Math.max(GAP, preferred), viewportWidth - CARD_WIDTH - GAP);

  return { top: Math.round(Math.max(GAP, top)), left: Math.round(Math.max(GAP, left)) };
}
