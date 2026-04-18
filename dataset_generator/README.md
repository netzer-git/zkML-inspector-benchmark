# dataset_generator (planned)

Tools for assembling and validating new ground-truth audit-finding datasets.

## Planned scope

- **Schema validation** — verify that an xlsx conforms to the dataset schema (columns, closed-list values, code-ref format).
- **Curation helpers** — interactive prompts for adding new findings, with closed-list validation and ID auto-assignment.
- **Candidate proposal** — pipelines that scan an existing audit report and propose candidate finding entries (severity, category, code refs) for human review.
- **Export** — write the curated dataset back to xlsx in the canonical column order.

## Status

Not yet implemented.
