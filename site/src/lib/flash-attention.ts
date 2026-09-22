import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(process.cwd(), '..');
const directory = 'VeriTile/Examples/FlashAttention1/';
const read = (path: string) => readFileSync(resolve(root, directory, path), 'utf8');
const kernels = read('Common/Kernels.lean');
const references = read('Common/Spec.lean');
const refinement = read('NaiveRefinement.lean');
const views = read('Boundary/Views.lean');

// Exact excerpts from the checked sources. A changed declaration must be reviewed
// instead of silently leaving a stale copy in the website.
function excerpt(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from + start.length);
  if (from < 0 || to < 0) throw new Error(`FlashAttention excerpt changed: ${start}`);
  return source.slice(from, to).trim();
}

const kernel = excerpt(kernels,
  'def fa1ForwardKernelStridedBoundaryD\n',
  '\n\ndef fa1ForwardKernelStridedCausalBoundaryD');
const contract = excerpt(refinement,
  'specification fa1_boundaryD_refines_naive_reference_views\n',
  '\n\n/-- Causal FA-1');
const proofStart = contract.indexOf(' := by\n');
if (proofStart < 0) throw new Error('FlashAttention contract no longer has the expected proof body.');

export const flashAttention = {
  kernel,
  update: excerpt(kernel, '    scores_raw :=', '\n  }').replace(/^ {4}/gm, ''),
  viewBinding: excerpt(references, 'def boundaryKernelD (views : FA1Views4D', '\n\ndef causalBoundaryKernelD'),
  layoutBinding: excerpt(references, 'def boundaryKernelD (layout : FA1Layout4D', '\n\ndef causalBoundaryKernelD'),
  reference: excerpt(refinement, 'noncomputable def fa1NaiveReference4D', '\n\nnoncomputable def fa1NaiveCausalReference4D'),
  attentionReference: excerpt(references, 'noncomputable def attentionReal4D', '\n\n/-- 4D ℝ-valued causal'),
  statement: contract.slice(0, proofStart),
  proof: contract.slice(proofStart + ' := '.length),
  contract,
  executionProof: excerpt(refinement,
    'theorem fa1_boundaryD_refines_naive_reference_exec_views\n',
    '\n\n/-- Compute-facing FA-1'),
  viewProof: excerpt(views,
    'theorem fa1_forward_correct_4D_boundaryD_views\n',
    '\n\n/-- D-tail causal-boundary'),
  source: refinement.trimEnd(),
  sourcePath: directory + 'NaiveRefinement.lean',
};
