import { describe, expect, it } from 'vitest';
import {
  applyExpansion,
  collapse,
  countNodes,
  createRoot,
  findNode,
  markLoading,
  pathKey,
} from '../src/lib/tree';
import type { TreeExpansion } from '../src/lib/protocol';

const FIRST: TreeExpansion = {
  path: [],
  children: [
    { id: 262, text: ' the', prob: 0.6 },
    { id: 284, text: ' to', prob: 0.25 },
    { id: 257, text: ' a', prob: 0.15 },
  ],
};

const SECOND: TreeExpansion = {
  path: [262],
  children: [
    { id: 1097, text: ' same', prob: 0.5 },
    { id: 717, text: ' first', prob: 0.3 },
  ],
};

describe('pathKey', () => {
  it('names the trunk and joins everything else', () => {
    expect(pathKey([])).toBe('root');
    expect(pathKey([262, 1097])).toBe('262.1097');
  });
});

describe('applyExpansion', () => {
  const chosen = [262, 717];
  const once = applyExpansion(createRoot(5), FIRST, chosen);
  const twice = applyExpansion(once, SECOND, chosen);

  it('attaches children and marks the node expanded', () => {
    expect(once.status).toBe('expanded');
    expect(once.children.map((child) => child.tokenId)).toEqual([262, 284, 257]);
  });

  it('multiplies probabilities along the path', () => {
    const deep = findNode(twice, [262, 717]);
    expect(deep?.prob).toBeCloseTo(0.3, 9);
    expect(deep?.cumulative).toBeCloseTo(0.6 * 0.3, 9);
  });

  it('tracks depth', () => {
    expect(findNode(twice, [262])?.depth).toBe(1);
    expect(findNode(twice, [262, 1097])?.depth).toBe(2);
  });

  it('highlights only the branch the model actually took', () => {
    expect(findNode(twice, [262])?.onChosenPath).toBe(true);
    expect(findNode(twice, [284])?.onChosenPath).toBe(false);
    expect(findNode(twice, [262, 717])?.onChosenPath).toBe(true);
    // " same" was the likeliest continuation but not the sampled one.
    expect(findNode(twice, [262, 1097])?.onChosenPath).toBe(false);
  });

  it('leaves untouched branches alone', () => {
    expect(findNode(twice, [284])?.children).toEqual([]);
    expect(findNode(twice, [284])?.status).toBe('collapsed');
  });

  it('does not mutate the tree it was given', () => {
    const root = applyExpansion(createRoot(3), FIRST);
    const next = applyExpansion(root, SECOND);
    expect(root.children[0]?.children).toEqual([]);
    expect(next.children[0]?.children).toHaveLength(2);
  });

  it('ignores an expansion for a path that is not in the tree', () => {
    const stray = applyExpansion(once, { path: [999], children: FIRST.children });
    expect(countNodes(stray)).toBe(countNodes(once));
  });
});

describe('findNode', () => {
  const tree = applyExpansion(createRoot(2), FIRST);

  it('finds the root for the empty path', () => {
    expect(findNode(tree, [])?.key).toBe('root');
  });

  it('returns null for an unknown path', () => {
    expect(findNode(tree, [1, 2, 3])).toBeNull();
  });
});

describe('markLoading / collapse', () => {
  const tree = applyExpansion(createRoot(2), FIRST);

  it('flips one node to loading without touching its siblings', () => {
    const loading = markLoading(tree, [284]);
    expect(findNode(loading, [284])?.status).toBe('loading');
    expect(findNode(loading, [262])?.status).toBe('collapsed');
  });

  it('drops the children when a node is collapsed', () => {
    const deep = applyExpansion(tree, SECOND);
    expect(countNodes(deep)).toBe(6);
    const folded = collapse(deep, [262]);
    expect(findNode(folded, [262])?.children).toEqual([]);
    expect(countNodes(folded)).toBe(4);
  });
});

describe('countNodes', () => {
  it('counts the root plus every descendant', () => {
    expect(countNodes(createRoot(1))).toBe(1);
    expect(countNodes(applyExpansion(createRoot(1), FIRST))).toBe(4);
  });
});
