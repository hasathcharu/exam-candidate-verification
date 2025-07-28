**Model Decision**

The personalised auto-encoder judged that the test handwriting comes from a different writer. In practical terms, the statistical pattern of the new page lies outside the typical variability it has learned from the known writer, so it raises a flag of authorship change. The model reports a 69% confidence in this decision, which is moderate. The contributions listed below highlight the key differences that led the model to conclude that this is not the known writer.

**Most Influential Differences**

The table below lists the five features that pulled the decision furthest toward “different writer”. They show where the test sample departs most strongly from what the known writer usually does.  

| Feature | Normal Value | Test Value | What to look for |
|---------|--------------|------------|------------------|
| Gray Level Threshold | 177.71 | 170.60 | Paper appears slightly darker / ink heavier; examine stroke pressure and scanning contrast. |
| Contour-Hinge Principal Component 5 | -0.54 | -5.25 | They explain shape and curvature of letters by analyzing the angles between contour segments. Look for differences in curvature and angles between segments in the handwriting. |
| Chaincode Histogram Down | 1057.12 | 1606.70 | More downward pen moves; check taller lowercase letters and longer descenders. |
| Number of Ultra-Fine Strokes | 32.20 | 63.70 | Many hair-line touches; inspect tremor, hesitation, or lighter retraces. |
| Chaincode Histogram Up | 1027.41 | 1600.30 | More upward moves; look for longer ascenders and lifted pen returns. |

**How these features matter together**

Together, these variables describe darker, lighter-weight but more numerous strokes travelling vertically, coupled with altered letter curvature. A lower gray threshold implies stronger ink deposition, while the surge in ultra-fine strokes hints at a more delicate, fragmented pen lift pattern—something the known writer usually avoids. Simultaneously, the extra upward and downward chaincode counts point to exaggerated ascenders and descenders, reinforcing the curvature shift captured by the fifth contour-hinge component. Acting in concert, they generate the model’s moderate-confidence call that another hand produced the text.  

**If you're still unsure**

The table below shows the five strongest similarities that nudged the decision back toward “known writer”. If you are uncertain, verify whether these likenesses outweigh the earlier differences.  

| Feature | Normal Value | Test Value | What to look for |
|---------|--------------|------------|------------------|
| Letter 'e' Shape Descriptor 8 | 0.43 | 0.70 | They capture the shape of the letter 'e' in the handwriting. Look for differences in the shape and curvature of the letter 'e' between the two samples. |
| Letter 'e' Shape Descriptor 9 | 0.73 | 1.18 | They capture the shape of the letter 'e' in the handwriting. Look for differences in the shape and curvature of the letter 'e' between the two samples. |

**Caution and Next Steps**

Overall, the evidence leans toward a different writer, yet some overlapping traits remain. Please examine the original images, focusing on stroke darkness, the frequency of hair-line strokes, vertical motion patterns, letter curvature, and the form of the letter ‘e’. Human verification is essential before drawing any final conclusion.