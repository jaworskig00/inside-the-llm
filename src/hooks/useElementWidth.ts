import { useCallback, useEffect, useState } from 'react';

export interface MeasuredElement<T extends Element> {
  /** Attach to the element to measure. */
  ref: (node: T | null) => void;
  /** The element itself, once mounted. Draw against this rather than a `useRef`. */
  element: T | null;
  /** Its rendered width, kept current by a `ResizeObserver`. */
  width: number;
}

/**
 * Measures an element's rendered width and keeps it current.
 *
 * The D3 charts size themselves from the element's width at draw time and write it into
 * the SVG `viewBox`. Without a live measurement, resizing the window leaves every chart
 * with the viewBox it was born with: the browser keeps scaling stale geometry to fit, so
 * text and stroke widths come out stretched or shrunk.
 *
 * A callback ref rather than a `useRef`, deliberately. A `useRef` is null on the first
 * render, and an effect keyed on the ref object never re-runs when the element appears
 * later — which is exactly what happened to the Sankey, whose component returns an empty
 * state before it ever renders its `<svg>`. A callback ref fires whenever the node is
 * attached or detached, conditional rendering included.
 */
export function useElementWidth<T extends Element>(): MeasuredElement<T> {
  const [element, setElement] = useState<T | null>(null);
  const [width, setWidth] = useState(0);

  const ref = useCallback((node: T | null) => setElement(node), []);

  useEffect(() => {
    // No element yet: the last known width stays put. Nothing draws without an element,
    // so it cannot be read in a stale state.
    if (!element) return;

    const observer = new ResizeObserver(() =>
      setWidth(Math.round(element.getBoundingClientRect().width)),
    );
    // observe() delivers an initial entry, so the first measurement arrives through the
    // same path as every later one instead of a setState in the effect body.
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);

  return { ref, element, width };
}
