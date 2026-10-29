# Data-section redesign: research before implementation

26 October 2026. Scope: the local Data & Evaluation section and its direct connection to results. Preserve the six-task audio demos, manuscript claims, private repository, and review-before-push workflow. No deployment.

## Reference review

Read the public project pages for Gestalt, ROMA, SAO-Instruct, SmartDJ, and UNISON. Visually inspected Gestalt's data/training section, ROMA's dataset section, and SAO-Instruct's audio examples.

| Reference | Useful design observation | Decision for this page |
| --- | --- | --- |
| [Gestalt](https://gewu-lab.github.io/Gestalt/) | Clear separation of scale numbers and an explanatory curriculum graphic, with matching typography | Rebuild diagrams for website reading, keep data units explicit |
| [ROMA](https://gewu-lab.github.io/ROMA/) | Named data subsets, statistics and concrete examples form a coherent dataset section | Keep each setting's scale, composition and example attached in a fixed column |
| [SAO-Instruct](https://eth-disco.github.io/sao-instruct/) | Exact editing instructions sit immediately beside their audio examples | Keep the common pitch instruction adjacent to both reference examples |
| [SmartDJ](https://zitonglan.github.io/project/smartdj/smartdj.html) | Constructor stages explain how source sounds become editable scenes | Show source pools before the two construction settings |
| [UNISON](https://lizhaoqing.github.io/UNISON-demo/) | Task labels and task-specific examples make broad capability browsable | Retain six-task coverage and name the four selective operations where their composition rules apply |

These are design observations, not a ranking of project pages. Reuse no scientific claims or example media from other projects.

## Scientific relationships checked in the manuscript

- The two source pools contain 635,019 class-labeled chunks and 380,165 fine-captioned chunks. Both Complex and Selective reuse these pools and their source-level train/validation/test partition (Audio Data Sources; Selective Benchmark Construction).
- The approximately one-million-entry source pool is upstream of both settings, not a training-set size exclusive to Complex.
- Complex training pairs are constructed online, with a nominal length of approximately 600,000 samples. Selective training contains exactly 61,440 fixed samples. Each setting has 6,000 validation and 6,000 test samples.
- Composition counts refer to Replace, Remove, Pitch, and Volume: Complex references have one to four sources (16.8%, 62.1%, 15.5%, 5.5%); Selective references have exactly two temporally non-overlapping sources.
- Generate and Add do not inherit the two-source rule. In Selective, Add uses a one-source reference, while Generate follows the Complex construction strategy.
- The provided road-noise/dog-barking spectrograms are representative input examples, not model outputs. Their instruction is to increase the pitch of dog barking.

## Implementation plan, before edits

1. Present a shared source-pool node feeding two equal setting columns.
2. In each column show training, validation and test scale, then source composition, then the corresponding example. Keep quantities in the same units and distinguish nominal online training length from fixed samples.
3. Redraw source-count and example annotations in native HTML/SVG. Preserve measured spectrogram pixels from the original PDF; do not synthesize empirical audio, results, or timestamps. Keep target/non-target visual encoding consistent across the pair.
4. Make editable vector assets available alongside the live page. Keep diagrams accessible and legible on mobile; retain figure enlargement. Verify values and images against the manuscript, then check desktop/mobile browser rendering.
5. Leave the revised page local for author review. Do not commit, push, publish, or enable deployment.

## Follow-up: clarify the two visual relationships

- Draw explicit construction branches with complete arrowheads and labels; avoid clipped dot endpoints.
- Attach source labels to the measured spectrograms using the original paper's source brackets. Add a separately labeled schematic source layout to explain overlap without fabricating empirical timing.
- Regenerate both reference SVGs and the complete overview. Preserve the compact count units, exact source facts, and local review workflow.
