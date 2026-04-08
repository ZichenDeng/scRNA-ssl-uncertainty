# MS2 Sections 1-2 Speaker Notes

## Slide 1: Why GSE96583?
- Open with the rescoping decision.
- Emphasize that MS2 rewards a defensible dataset and a clear wrangling story.
- Say explicitly that GSE96583 already gives us both labels and shift.

## Slide 2: Raw Data Structure
- Explain that the GEO download is fragmented rather than analysis-ready.
- Name the batch1 and batch2 source files once so the TF sees that you understand the actual data layout.
- Transition into why preprocessing is necessary.

## Slide 3: Wrangling Pipeline
- Keep this procedural and concrete.
- The important point is that each preprocessing step protects validity of cross-batch comparisons.

## Slide 4: Wrangling Outcomes
- End with concrete cell counts and class imbalance.
- Set up the next presenter by saying these processed objects are now ready for EDA and baseline modeling.
