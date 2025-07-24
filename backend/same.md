**Model Decision**

The personalised auto-encoder concludes that the test page was written by the known writer, expressing moderate assurance (≈62 % confidence). In other words, despite some small stylistic shifts, the statistical fingerprint of the new sample still aligns most closely with the writer whose past material the system studied.

**Key supportive cues (features with the five lowest negative SHAP contributions)**  
The table below lists the elements that pulled the decision toward “known writer.” Their negative SHAP values mean they reduced the odds of the handwriting being attributed to a different writer.

| Feature | Normal Value | Test Value | What to look for |
|---------|--------------|------------|------------------|
| Contour-Hinge Principal Component 1 | -0.298 | 0.967 | They explain shape and curvature of letters by analyzing the angles between curve segments. Look for similarities in curvature and angles between segments in the handwriting. |
| Contour-Hinge Principal Component 8 | 0.461 | 1.542 | They explain shape and curvature of letters by analyzing the angles between curve segments. Look for similarities in curvature and angles between segments in the handwriting. |
| Mean Gap Between Words | 36.10 | 43.69 | Writer usually leaves moderate word gaps; the test page shows wider spacing, but still within their habitual range. |
| Number of Ultra-Fine Strokes | 32.20 | 26.90 | Writer usually scatters many hair-thin strokes; the test page shows slightly fewer, can be a tolerable day-to-day fluctuation. |
| Standard Deviation of Gap Between Words | 17.71 | 17.84 | Consistency of word spacing is almost unchanged, echoing the writer’s typical rhythm. |

**How these cues matter**

Taken as a group, the two contour-hinge components point to familiar letter curvature, while the gap-based metrics confirm the writer’s characteristic pacing. The slight reduction in ultra-fine strokes is offset by consistent spacing variability, suggesting the same motor control but a different pen or paper texture. Their combined stability convinced the model that the handwriting falls within the known writer’s natural range.

**Points that argued the other way (features with the five highest positive SHAP contributions)**  
If you're still uncertain, you can inspect the items below—the main reasons the system still entertained the possibility of a different writer.

| Feature | Normal Value | Test Value | What to look for |
|---------|--------------|------------|------------------|
| Contour-Hinge Principal Component 3 | 1.802 | 3.824 | They explain shape and curvature of letters by analyzing the angles between curve segments. Look for differences in curvature and angles between segments in the handwriting. |
| Contour-Hinge Principal Component 7 | -0.296 | -0.037 | They explain shape and curvature of letters by analyzing the angles between curve segments. Look for differences in curvature and angles between segments in the handwriting. |
| Letter 'e' Shape Descriptor 5 | 1.570 | 1.470 | They capture the shape of the letter 'e' in the handwriting. Look for differences in the shape and curvature of the letter 'e' between the two samples. |

**Final thoughts**

Overall, the balance of evidence favours the known writer, but some curvature and spacing changes kept the confidence below absolute. Readers should visually cross-check the highlighted regions—letter shapes, stroke fineness, and word spacing—using the accompanying images before accepting the model’s judgment.