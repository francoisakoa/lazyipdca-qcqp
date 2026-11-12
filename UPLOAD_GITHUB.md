# Upload this release to GitHub

Extract the ZIP and place the **contents of `lazyqcqp/`** at the root of the repository `francoisakoa/lazyqcqp`.

If you already have a local checkout, copy the files into that checkout, inspect `git status`, then commit and push normally:

```bash
git status
git add .
git commit -m "Add LazyQCQP manuscript R24 and reproducibility materials"
git push
```

The root `README.md` links to the two PDFs, their sources, the solver and the analysis commands. `CITATION.cff` supplies the citation metadata. Keep the directory structure when uploading.

For a new, empty remote repository, run these commands from inside `lazyqcqp/`:

```bash
git init -b main
git add .
git commit -m "Add LazyQCQP manuscript R24 and reproducibility materials"
git remote add origin https://github.com/francoisakoa/lazyqcqp.git
git push -u origin main
```

For journal submission, use the main PDF and supplementary PDF in `manuscript/`. The same directory contains the LaTeX sources, figures, document class, bibliography style and `.bib` file.
