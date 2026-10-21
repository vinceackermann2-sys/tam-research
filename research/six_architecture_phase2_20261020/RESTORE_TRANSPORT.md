# Checkpoint transport after the21 October interruption

The first restoration runtime disappeared after checkpoint-piece uploads but
before reassembly or the release marker. Its controller could not start, so
that attempt adds zero training exposures. The prior22,224,896-position parent
reservation remains conservatively charged. Cause of the second runtime's
disappearance is unknown; neither interruption is an architecture failure.

To avoid allocating another GPU during a long desktop upload, exact verified
optimizer/RNG files are cached as assets of an unpublished draft release in the
user's repository. Files remain outside Git. GitHub documents files below2GiB
and no total release-size or bandwidth limit in its
[release storage documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
Draft-release listings require push access, as described by the
[release API](https://docs.github.com/en/rest/releases/releases).
This cache must never be published automatically.

Every asset's server SHA-256 and size must match the original locally verified
checkpoint before allocating one replacement L4 within existing entitlement.
The desktop mints short-lived signed asset redirects using the
[asset download API](https://docs.github.com/en/rest/releases/assets).
The GitHub credential stays on the desktop. Signed URLs are confined to private
work files, with a restricted directory on the runtime, and are deleted there
after successful verification. They never enter source, public metadata or logs.

The replacement downloads whole files, verifies their hashes, reconstructs the
original64MiB chunks and verifies each chunk plus the concatenated checkpoint.
Only all-four success writes RESTORE_COMPLETE. The unchanged preparation/worker
then reproduces the exact corpus hashes and runs the original controller from
the same seed/full optimizer/RNG states. Scientific source files, gates, batch,
context, optimizer and learning schedule are unchanged. An offline corrupt-asset
fixture verifies that an invalid file cannot write the release marker.

Immutable pre-allocation archives retain unique lost-reservation digests. Future
recovery merges those archives even if a local evidence export lags a subsequent
reservation acknowledgment. Replaying an uncommitted training reservation is
charged once; transport retries before controller release are not training.
