# Source and redistribution notes

The model, dataset loader, loss, and training equations were extracted from the
authors' CSDS_Net research workspace. `results/provenance.json` records source
hashes and `results/history_audit.json` records experiment-level provenance.
The packaged model retains the trained semantic variant and its parameter keys.
New command-line utilities remove dependencies on local Windows paths and the
private benchmark orchestration scripts.

The architecture/block illustrations and all five README experiment figures
originate from the author-supplied manuscript LaTeX archive. They are copied
unchanged. The README tables are transcribed from the same manuscript and its
data files, with original numerical precision retained. They describe what
the manuscript reports; see `docs/PAPER_REFERENCE.md` for result provenance
and `docs/IMPLEMENTATION.md` before relating them to the released code.
`results/paper_asset_provenance.json` records source entries and image/data
hashes. Extra `figs/predictions.png` and `figs/ablation_curves.png` are archived
code checks from the earlier packaging pass, not the README paper figures.
Their sample IDs and raw histories remain available in `results/`.

No dataset, third-party repository, pretrained weight, training checkpoint, or
full manuscript PDF/LaTeX project is redistributed in this source package;
the extracted manuscript illustrations and small reported-data CSVs are included.
Dataset access and reuse follow the original providers' terms. Dependencies retain
their own licenses. The seven comparator implementations are not vendored here.
Archived SCSegamba and Swin-UMamba results use pure-PyTorch selective-scan
approximations, not their official CUDA implementations.

A source-code license has not been selected by the authors. No open-source
license grant is implied by this package or the manuscript's journal template.
Before public release, the rights holders should add their chosen LICENSE and
confirm permission to redistribute the illustrations. Citation metadata alone
does not grant reuse rights.
