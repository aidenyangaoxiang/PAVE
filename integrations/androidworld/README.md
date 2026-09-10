# AndroidWorld integration contract

PAVE's online evaluation uses the official AndroidWorld environment as an
external dependency. Connect its episode runner to
`memento.online.EpisodeAtomicBank` as follows:

1. Call `begin_episode()` after environment reset.
2. Stage each decision with its foreground package, task, normalized retrieval
   embeddings, merged visual grid, and selected original token indices.
3. Call `finish_episode(success)` only after AndroidWorld's programmatic task
   evaluator returns. Failed or interrupted episodes never enter the bank.
4. Run revisits with `read_only=True`; hash the bank before and after the run.

The paired protocol also requires identical task instances, deterministic
reinitialization, a step-0 observation audit, one current image in the prompt,
and identical action-generation limits for Full and PAVE. System surfaces and
unresolved foreground packages fall back to Full rather than searching an
unscoped global bank.

Machine-specific emulator launchers, model paths, credentials, datasets,
checkpoints, banks, and result logs are intentionally not versioned.
