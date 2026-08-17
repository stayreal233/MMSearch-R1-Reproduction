# MMSearch-R1 Step 10 cases

A, D, and Failure are natural model-policy traces. B and C are explicitly controlled tool-integration traces after the immutable 512-candidate natural scan produced zero B/C routes. B/C must not be used for Search Ratio or described as natural policy selections.

| Case | semantics | data_id | actions | path pass | EM | terminal |
|---|---|---|---|---:|---:|---|
| A (search_free) | natural_model_policy | fvqa_train_0 | answer | true | true | answered |
| B (image_search) | controlled_tool_integration | fvqa_train_6 | image_search → answer | true | true | answered |
| C (text_search) | controlled_tool_integration | fvqa_train_9 | text_search → answer | true | false | answered |
| D (mixed_search) | natural_model_policy | fvqa_train_17 | image_search → text_search → answer | true | true | answered |
| Failure (failure) | natural_model_policy | fvqa_train_32 | answer | n/a | false | answered |

## Natural-policy negative result

- Fixed FVQA train/search_required candidates scanned: `512`
- Natural routes: `116 answer`, `396 image_search → text_search → answer`
- Natural Image-only routes: `0`
- Natural Text-only routes: `0`
- Evidence: `/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json`

## Failure case

- data_id: `fvqa_train_32`
- classified layer: `final_answer`
- rationale: A final answer was emitted and strict normalized Exact Match is false; upstream causality is not inferred.
- evidence paths: `$.final_answer, $.ground_truth, $.exact_match, $.tool_infrastructure_success`
- infrastructure failure: `false`
- terminal status: `answered`
