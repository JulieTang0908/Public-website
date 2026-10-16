# N–x–p Composition Probability Viewer

A self-contained interactive website showing every integer field length N from 1 to 100. The website runs entirely in the browser, with embedded plotting code and frozen simulation data. It requires no GPT account, API key, backend, or external script service.

## Website

GitHub Pages address (available after Pages is enabled): https://julietang0908.github.io/Public-website/

- `index.html`: interactive probability surfaces and individual N slices.
- `guide.html`: parameter definitions, formulas, interpretation, and advisor presentation notes.
- `guide.md`: downloadable Markdown explanation.
- `figures.html`: scientific PNG gallery and download links.
- `feature_metrics.csv`: numerical summaries for the feature distributions.

## Enable GitHub Pages

In **Settings → Pages**, select **Deploy from a branch**, choose **main** and **/(root)**, then **Save**. The `.nojekyll` file preserves this static site as uploaded.

## Local use

Download the repository files and open `index.html` directly. No installation or build is required. The 17 MB HTML includes all interactive data and plotting libraries, so its initial download can take time on a slow connection.

## Interpretation

The default plot shows composition posterior probabilities under a uniform prior over legal count pairs. Its reversible composition code is a display label whose distances have no physical meaning. Density heights can exceed one. Separate feature-marginal views address P(F|N). These frozen synthetic examples do not establish real-image recognition accuracy or Gaussian convergence.
