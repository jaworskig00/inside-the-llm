import { expect, test, type Page } from '@playwright/test';

/**
 * One smoke test that walks the app the way a visitor does: load, generate, step, inspect.
 *
 * It runs against `?demo=1`, which replays a run recorded from the real ONNX graph through
 * the same protocol the real worker uses. That keeps CI offline and fast while still
 * covering the worker boundary, the store, and every view's reaction to a selection change.
 */

const DEMO = '?demo=1';

async function generate(page: Page) {
  await page.goto(DEMO);
  await expect(page.getByRole('button', { name: 'Generate' })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Generate' }).click();
  await expect(page.getByText('step 12 / 12')).toBeVisible({ timeout: 30_000 });
}

test('loads, shows what is running, and keeps the weights out of CI', async ({ page }) => {
  await page.goto(DEMO);

  await expect(page).toHaveTitle(/Inside the LLM/);
  await expect(page.getByRole('heading', { name: 'Inside the LLM', level: 1 })).toBeVisible();
  await expect(page.getByText(/distilgpt2 · 6 layers · 12 heads/)).toBeVisible();
  await expect(page.getByText('recorded run', { exact: true })).toBeVisible();

  // The loading overlay must get out of the way.
  await expect(page.getByRole('button', { name: 'Generate' })).toBeEnabled({ timeout: 30_000 });
});

test('generates, and every view fills in', async ({ page }) => {
  await generate(page);

  // Token strip: prompt plus generated tokens, and the decoded output.
  await expect(page.getByRole('listitem').first()).toBeVisible();
  expect(await page.getByRole('listitem').count()).toBe(21);
  await expect(
    page.getByText('the area before, but the local council voted unanimously to remove'),
  ).toBeVisible();

  // Every panel reports something rather than its empty state.
  await expect(page.getByRole('img', { name: 'Next-token probabilities' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Attention head grid/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /Attention arcs from token/ })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Logit lens flow across layers' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Residual stream magnitude/ })).toBeVisible();

  await expect(page.getByText('Generate something to see the distribution.')).toHaveCount(0);
  await expect(page.getByText('Run a generation to read the layers.')).toHaveCount(0);
});

test('keyboard steps the run and relinks the other views', async ({ page }) => {
  await generate(page);

  const grid = page.getByRole('img', { name: /Attention head grid/ });
  const before = await grid.getAttribute('aria-label');

  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByText('step 10 / 12')).toBeVisible();

  // The arc diagram follows the step, which is the whole point of "linked views".
  const arcs = page.getByRole('img', { name: /Attention arcs from token/ });
  await expect(arcs).toBeVisible();
  expect(await grid.getAttribute('aria-label')).toBe(before);

  await page.keyboard.press('ArrowRight');
  await expect(page.getByText('step 11 / 12')).toBeVisible();
});

test('selecting a layer and a head moves the attention detail', async ({ page }) => {
  await generate(page);

  await page
    .getByRole('group', { name: 'Layer' })
    .getByRole('button', { name: '2', exact: true })
    .click();
  await page
    .getByRole('group', { name: 'Head' })
    .getByRole('button', { name: '7', exact: true })
    .click();

  await expect(
    page.getByRole('img', { name: /Attention arcs from token \d+ in layer 2, head 7/ }),
  ).toBeVisible();
  await expect(page.getByRole('img', { name: /Selected layer 2, head 7/ })).toBeVisible();
});

test('the sampling sliders reshape the distribution without re-running the model', async ({
  page,
}) => {
  await generate(page);

  const kept = page.locator('text=KEPT').locator('xpath=following-sibling::*[1]');
  const temperature = page.getByLabel('temperature');

  await temperature.fill('0');
  // Greedy decoding keeps exactly one candidate.
  await expect(kept).toHaveText('1', { timeout: 10_000 });

  await temperature.fill('1.5');
  await expect(kept).not.toHaveText('1', { timeout: 10_000 });
});

test('the alternatives tree expands lazily', async ({ page }) => {
  await generate(page);

  const panel = page.locator('section', {
    has: page.getByRole('heading', { name: 'Alternatives' }),
  });
  await expect(panel.getByText('prompt')).toBeVisible();

  await panel.locator('g.node').first().click();
  // One forward pass later there are five branches under the root.
  await expect(panel.locator('g.node')).toHaveCount(6, { timeout: 15_000 });
});

// Regression: space already activates a focused control, so taking it for playback as well
// made one key press do two things.
test('space does not hijack a focused button', async ({ page }) => {
  await generate(page);

  const layerTwo = page
    .getByRole('group', { name: 'Layer' })
    .getByRole('button', { name: '2', exact: true });
  await layerTwo.focus();
  await page.keyboard.press(' ');

  await expect(layerTwo).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: /Pause playback/ })).toHaveCount(0);

  // It still works when focus is not on a control.
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press(' ');
  await expect(page.getByRole('button', { name: /Pause playback/ })).toBeVisible();
});

// Regression: the charts wrote their width into the viewBox at draw time and never redrew,
// so after a resize the browser stretched stale geometry instead of re-laying it out.
test('charts redraw at the size they are displayed at after a resize', async ({ page }) => {
  await generate(page);
  await page.setViewportSize({ width: 1100, height: 900 });

  const labels = [
    'Next-token probabilities',
    'Logit lens flow across layers',
    /Attention arcs from token/,
  ];

  for (const label of labels) {
    const svg = page.getByRole('img', { name: label });
    await expect
      .poll(
        async () => {
          const box = await svg.evaluate((node) => ({
            viewBox: Number(node.getAttribute('viewBox')?.split(' ')[2] ?? 0),
            width: node.getBoundingClientRect().width,
          }));
          return Math.abs(box.viewBox - box.width);
        },
        { timeout: 10_000 },
      )
      .toBeLessThan(24);
  }
});

test('explain mode can be turned off', async ({ page }) => {
  await page.goto(DEMO);
  const caption = page.getByText('The model does not see letters');
  await expect(caption).toBeVisible();

  await page.getByRole('button', { name: /Explain/ }).click();
  await expect(caption).toHaveCount(0);
});

test('a narrow viewport is told the truth instead of being left broken', async ({ page }) => {
  await page.setViewportSize({ width: 414, height: 900 });
  await page.goto(DEMO);

  await expect(page.getByText('Best on a desktop screen')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate' })).toBeEnabled({ timeout: 30_000 });
});
