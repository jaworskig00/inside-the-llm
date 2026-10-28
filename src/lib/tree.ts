/**
 * The alternatives tree: "what else could the model have said here?"
 *
 * Nodes are keyed by the token path that reaches them, so an expansion request and its
 * response can be matched up without the UI holding onto worker promises. Children arrive
 * lazily - expanding a node costs one forward pass, so nothing is computed until a user
 * actually opens a branch.
 */
import type { TreeChild, TreeExpansion } from './protocol';

export interface TreeNode {
  /** `root` for the trunk, otherwise the joined token path. */
  key: string;
  path: number[];
  tokenId: number | null;
  text: string;
  /** Probability of this token given its parent. 1 for the root. */
  prob: number;
  /** Product of probabilities along the path - how likely this whole continuation is. */
  cumulative: number;
  depth: number;
  children: TreeNode[];
  status: 'collapsed' | 'loading' | 'expanded';
  /** True for the branch the model actually took. */
  onChosenPath: boolean;
}

export function pathKey(path: number[]): string {
  return path.length === 0 ? 'root' : path.join('.');
}

export function createRoot(promptLength: number): TreeNode {
  return {
    key: 'root',
    path: [],
    tokenId: null,
    text: `prompt (${promptLength} tokens)`,
    prob: 1,
    cumulative: 1,
    depth: 0,
    children: [],
    status: 'collapsed',
    onChosenPath: true,
  };
}

export function findNode(root: TreeNode, path: number[]): TreeNode | null {
  if (path.length === 0) return root;
  let node: TreeNode = root;
  for (const tokenId of path) {
    const next = node.children.find((child) => child.tokenId === tokenId);
    if (!next) return null;
    node = next;
  }
  return node;
}

/**
 * Attaches an expansion to the tree, returning a new root (the store keeps tree state
 * immutable so React re-renders predictably).
 */
export function applyExpansion(
  root: TreeNode,
  expansion: TreeExpansion,
  chosenContinuation: number[] = [],
): TreeNode {
  return rebuild(root, [], expansion, chosenContinuation);
}

function rebuild(
  node: TreeNode,
  currentPath: number[],
  expansion: TreeExpansion,
  chosenContinuation: number[],
): TreeNode {
  if (pathKey(currentPath) === pathKey(expansion.path)) {
    return {
      ...node,
      status: 'expanded',
      children: expansion.children.map((child) => makeChild(node, child, chosenContinuation)),
    };
  }
  if (!isPrefix(currentPath, expansion.path)) return node;
  return {
    ...node,
    children: node.children.map((child) =>
      rebuild(child, [...currentPath, child.tokenId as number], expansion, chosenContinuation),
    ),
  };
}

function makeChild(parent: TreeNode, child: TreeChild, chosenContinuation: number[]): TreeNode {
  const path = [...parent.path, child.id];
  return {
    key: pathKey(path),
    path,
    tokenId: child.id,
    text: child.text,
    prob: child.prob,
    cumulative: parent.cumulative * child.prob,
    depth: parent.depth + 1,
    children: [],
    status: 'collapsed',
    onChosenPath: parent.onChosenPath && chosenContinuation[parent.depth] === child.id,
  };
}

export function markLoading(root: TreeNode, path: number[]): TreeNode {
  return mapNode(root, [], (node, nodePath) =>
    pathKey(nodePath) === pathKey(path) ? { ...node, status: 'loading' } : node,
  );
}

export function collapse(root: TreeNode, path: number[]): TreeNode {
  return mapNode(root, [], (node, nodePath) =>
    pathKey(nodePath) === pathKey(path) ? { ...node, status: 'collapsed', children: [] } : node,
  );
}

function mapNode(
  node: TreeNode,
  currentPath: number[],
  fn: (node: TreeNode, path: number[]) => TreeNode,
): TreeNode {
  const mapped = fn(node, currentPath);
  if (mapped !== node) return mapped;
  const children = node.children.map((child) =>
    mapNode(child, [...currentPath, child.tokenId as number], fn),
  );
  return children.some((child, i) => child !== node.children[i]) ? { ...node, children } : node;
}

function isPrefix(prefix: number[], full: number[]): boolean {
  if (prefix.length > full.length) return false;
  return prefix.every((value, i) => full[i] === value);
}

/** Counts nodes, so the UI can warn before an expansion explodes. */
export function countNodes(node: TreeNode): number {
  return 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0);
}
