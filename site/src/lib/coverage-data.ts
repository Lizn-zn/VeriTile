import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import checkedRun from './coverage-ci.json';

if (checkedRun.schema !== 1 || checkedRun.conclusion !== 'success'
    || !/^[0-9a-f]{40}$/.test(checkedRun.commit)
    || checkedRun.url !== `https://github.com/Lizn-zn/VeriTile/actions/runs/${checkedRun.run_id}`) {
  throw new Error('Invalid recorded corpus-check evidence.');
}
export const corpusCheck = checkedRun;

type Reference = { name: string; line: number };
export type CoverageRow = {
  id: string; port: string; file: string; line: number; declaration: string;
  python_file: string; python_functions: Reference[]; coverage_level: string;
  numeric_model: string; scope: string; statement: string; references: Reference[];
  io: { name: string; kernel: string; definition: string }[]; issue: string;
};

// One validator owns source fingerprints, explicit reviews, and the manifest.
// A source edit without an updated review stops the website build too.
export const coverage: { reviewed_on: string; basis: string; rows: CoverageRow[] } = JSON.parse(
  execFileSync('python3', ['bench/coverage_review.py', '--json'], {
    cwd: resolve(process.cwd(), '..'), encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
  }),
);

export const coverageLabels: Record<string, string> = {
  full_value_candidate: 'Original-kernel candidate',
  specialization: 'Configured model / stage',
  precomputed_input_slice: 'Precomputed-input slice',
  pre_rounding_slice: 'Pre-rounding slice',
  projection_only: 'Projection only',
  blocked_summary: 'Blocked surface',
  public_summary_with_proof_gap: 'Value proof gap',
};

export const numericLabels: Record<string, string> = {
  mathematical: 'Mathematical execution',
  abstract_rounding: 'Abstract cast/store rounding',
};

export const sourceLink = (path: string, line: number) =>
  `https://github.com/Lizn-zn/VeriTile/blob/main/${path}#L${line}`;
