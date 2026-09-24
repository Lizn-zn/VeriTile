import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// The tutorial displays the complete declarations from the executable showcase.
// Fail the build when an anchor changes instead of showing stale copied code.
const source = readFileSync(resolve(process.cwd(), '../bench/examples/VectorAdd.lean'), 'utf8');
function excerpt(pattern: RegExp): string {
  const match = source.match(pattern);
  if (!match) throw new Error(`VectorAdd tutorial source anchor changed: ${pattern}`);
  return match[1].trim();
}

export const tutorial = {
  python: excerpt(/```python\n([\s\S]*?)\n```/),
  setup: excerpt(/(import VeriTile\.Triton\n[\s\S]*?open VeriTile\.Examples\n)/),
  kernel: excerpt(/(def addKernel [\s\S]*?\n\})/),
  io: excerpt(/(def addIO [\s\S]*?write := fun pid => pid \* B)/),
  proof: excerpt(/(specification add_kernel_correctness[\s\S]*?)(?=\n\n\/\-! ## Trust gates)/),
  gates: excerpt(/(#axiomsClean add_kernel_correctness[\s\S]*?)(?=\n\nend VeriTile\.Bench)/),
};
