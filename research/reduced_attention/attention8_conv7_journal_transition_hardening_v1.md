# Conv7 synthetic journal transition hardening

**CPU-only local simulation; no paid-run authority.** Following merged PR #1387, a static audit found that the journal snapshot loop exited at the first ERROR, STARTED, or NONE. This could conceal forged later-model markers; a journal with invalid post-error history might appear cleanly terminal.

This successor PR validates all three transitions before classifying the journal. A later model may have a start or terminal marker only when its immediate predecessor is COMPLETE. It also rejects symlinks, including dangling symlinks that otherwise appear absent to Path.exists.

New adversarial tests cover forged valid markers after a failure or unfinished attempt, skipped model slots, symlink markers and a valid complete three-model history.

This does not establish atomicity on Modal Volume or reliability of a real distributed reservation. It cannot run H100 jobs, consume seeds, or approve spending. The scientific screen failure and consumed seed 60232 remain immutable.
