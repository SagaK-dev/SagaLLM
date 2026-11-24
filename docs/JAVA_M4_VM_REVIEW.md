# M4 disposable VM readiness — NOT_RUN

Applies to independent Java evaluation only. Does not authenticate reviewers,
prove isolation or waive Issue #36. Never execute generated Java on GitHub
Actions runners or persistent workstations.

1. Obtain the human-reviewed frozen M2 suite; keep it in private storage.
2. Provision a disposable Linux VM with no secrets, sensitive mounts, shared
   folders or persistent credentials. Require local Docker default context
   and Unix socket; review engine and cleanup with another real person.
3. Approve a Java 17 image by immutable repository digest. Verify image
   provenance and Java version with trusted commands. Review the networkless,
   non-root, read-only, capability-drop, no-new-privileges, resource-limited
   OCI configuration and confirm no engine socket is mounted into containers.
4. Record actual operator and separate reviewer, date, traceable review URLs,
   suite SHA and exact image. Keep record private; pin its SHA before inference.
5. Verify the same local context and digest before execution. Opt in only
   after all gates. Destroy the VM afterwards and retain bounded evidence.
6. Independently audit rights, reviewer independence and actual isolation.

## Private review JSON schema

- schema: sagallm-java-m4-vm-review-v1
- suite_sha256: exact frozen suite SHA-256
- image: exact repository@sha256 digest
- operator and reviewer: objects with id, date (YYYY-MM-DD), review_url
  (HTTPS); different identities and links
- controls: object containing exactly the following booleans, each true
  only after actual review:

  - disposable_linux_vm
  - no_host_secrets
  - no_persistent_mounts
  - no_engine_socket_mount
  - network_disabled
  - nonroot
  - resource_limits_reviewed
  - image_reviewed
  - jdk17_verified
  - destroy_after_run
  - local_engine_reviewed

Hash original bytes with sha256sum and pass the file and expected hash to
the evaluator. Structural success means ELIGIBLE_SELF_ATTESTED, not verified
identity or VM isolation. No real review exists yet. M4 remains NOT_RUN.
Common CI environment checks are defense in depth, not proof of execution
location. Only local Docker is supported for this independent path.

The independent evaluator also rejects image-declared volumes. The shared OCI runner sets `--log-driver=none` and an explicit `/bin/sh` entrypoint. Verify these settings on the actual approved engine; mocked tests do not establish that the runtime enforces them.
