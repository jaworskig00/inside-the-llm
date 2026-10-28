/**
 * The first-run walkthrough.
 *
 * Six stops, in the order someone actually needs them: make the model say something, then
 * explain what appeared. Each step names an element by its `data-tour` attribute; a step
 * whose element is missing is skipped rather than spotlighting empty space.
 */
export interface TourStep {
  id: string;
  /** Value of the `data-tour` attribute to spotlight. */
  target: string;
  title: string;
  body: string;
  /**
   * Primary button on this step. "generate" starts a run so the visitor sees the thing
   * work without having to figure out the controls first.
   */
  action?: 'generate';
}

export const TOUR_STEPS: readonly TourStep[] = [
  {
    id: 'prompt',
    target: 'prompt',
    title: 'The model runs here, in this tab',
    body: 'DistilGPT-2 — 82 million parameters — was downloaded once and is running on your machine. No server, no API key, nothing you type leaves the browser. Write anything, or let it go with the prompt already there.',
    action: 'generate',
  },
  {
    id: 'tokens',
    target: 'tokens',
    title: 'It never sees letters',
    body: 'Text arrives as these chunks. Brighter chips are tokens the model thought were likely; a dim one caught it by surprise. Click any chip and every view below jumps to that moment.',
  },
  {
    id: 'playback',
    target: 'playback',
    title: 'One token at a time',
    body: 'Generation is a loop: predict, pick, repeat. Play it back, or walk through it with the arrow keys — ← and → step, space plays.',
  },
  {
    id: 'sampling',
    target: 'sampling',
    title: 'What the dials actually do',
    body: 'The model scores all 50 257 tokens. The faint bars are its real opinion, the solid bars what survives temperature, top-k and top-p. Drag a slider and watch the distribution get reshaped — it re-filters the real numbers, it does not re-run the model.',
  },
  {
    id: 'attention',
    target: 'attention',
    title: 'Every attention head at once',
    body: 'Six layers by twelve heads. A bright diagonal is a head reading the previous word; a bright first column is a head parking its attention on the start of the text. Click any square to open it in the arc diagram underneath.',
  },
  {
    id: 'sankey',
    target: 'sankey',
    title: 'Watch it change its mind',
    body: 'If the model were forced to answer after each layer, what would it say? Ribbons that slope across are probability moving from one word to another as the text travels up the stack. Below this, the alternatives tree lets you fork the run from any token it did not take.',
  },
] as const;

export const TOUR_SEEN_KEY = 'inside-the-llm:tour-seen';

/** Whether the walkthrough should open by itself. `?tour=1` forces it, `?tour=0` suppresses it. */
export function shouldAutoStartTour(search: string): boolean {
  const params = new URLSearchParams(search);
  const flag = params.get('tour');
  if (flag === '1') return true;
  if (flag === '0') return false;
  try {
    return window.localStorage.getItem(TOUR_SEEN_KEY) === null;
  } catch {
    // Private windows refuse storage; showing the tour once per visit is the better failure.
    return true;
  }
}

export function markTourSeen(): void {
  try {
    window.localStorage.setItem(TOUR_SEEN_KEY, new Date().toISOString());
  } catch {
    // ignore
  }
}
