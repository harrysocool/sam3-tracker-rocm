# AMD XDNA D-state prevention and unattended recovery stack

**Date:** 2026-07-27

**Status:** deployed and boot-persistence verified

## Canonical project ownership

The recovery stack is maintained independently from SAM3 at:

```text
/home/amd/project/amdxdna-recovery
branch: main
tag: v0.1.0
```

This tracker report records workload-specific validation only. Driver source,
installation/rollback tools, watchdog configuration, monitor policy, generic
validation, and the operations runbook belong to `amdxdna-recovery`.

Local delivery artifacts:

```text
dist/amdxdna-pmfix-validated.ko
  SHA256 a302f61ed45f3bf6a053c967055d5b9d591d0ec96273caf208137a15d8a61124
dist/amdxdna-recovery-v0.1.0.bundle
  SHA256 ea6961a9d980d223f00519a9b84633fd97e59a73b068d931aa0f9118d5645cfb
```

## Problem

The February amdxdna driver could leave `amdxdna_js` in uninterruptible
D-state during normal sequential inference. A normal remote reboot stopped
networking first and then waited indefinitely for the D-state worker, leaving
the machine inaccessible until an on-site power cycle.

The initial scheduler-TDR backport recovered a controlled delayed host response
but did not recover the real frame-26 wedge.

## Preventive driver fix

Upstream commit `2be0d73` documents a runtime-PM deadlock: runtime suspend
drains the scheduler workqueue while `run_job()` on that workqueue calls
`pm_runtime_resume_and_get()`. The backport moves PM get before scheduler
queueing and holds it until final job cleanup.

Upstream `0220d14` was also backported to protect mailbox teardown from a NULL
callback handle when flushing a firmware-wedged channel.

```text
workspace: /home/amd/project/amdxdna_tdr_pmfix_20260726
branch: fix/tdr-pm-deadlock
module SHA256: a302f61ed45f3bf6a053c967055d5b9d591d0ec96273caf208137a15d8a61124
checkpatch: 0 errors, 0 warnings
```

Validation:

- temporary load and parameter verification;
- one-frame, accuracy, and five-frame gates;
- two independent 30-frame runs with different optimized backbones;
- four explicit 6.5-second autosuspend -> resume cycles;
- no TDR reports or D-state after the runs.

The module was installed persistently at:

```text
/lib/modules/6.14.0-1020-oem/updates/dkms/amdxdna.ko.zst
```

Original DKMS backup:

```text
/home/amd/project/9_to_delete/amdxdna_pmfix_persistent_20260727/
amdxdna.original.ko.zst
```

Two controlled reboots confirmed the PM-fix raw SHA and TDR parameters load
from disk. DKMS/package updates may overwrite this file and therefore require
post-update SHA verification.

## Hardware watchdog

The AMD FCH SMBus device supports `sp5100_tco`. Ubuntu's OEM kernel package
blacklists watchdog drivers for automatic modules-load, so a systemd drop-in
explicitly loads it after `systemd-modules-load.service`:

```text
/etc/systemd/system/systemd-modules-load.service.d/90-sp5100-watchdog.conf
ExecStartPost=/sbin/modprobe sp5100_tco
```

System manager configuration:

```text
WatchdogDevice=/dev/watchdog0
RuntimeWatchdogSec=30s
RebootWatchdogSec=2min
KExecWatchdogSec=off
```

Boot verification:

```text
identity=SP5100 TCO timer
state=active
timeout=30
timeleft sample=29 -> 27
RuntimeWatchdogUSec=30s
RebootWatchdogUSec=2min
```

The first initramfs-loading attempt was rejected: although the module was
included, it was not retained/registered early enough. The initramfs modules
file was restored; the post-modules-load drop-in is the validated solution.

## Persistent D-state monitor

Installed service:

```text
amdxdna-dstate-monitor.service
ExecStart=/usr/local/sbin/amdxdna-dstate-monitor
interval=10s
threshold=6 consecutive D-state samples
```

It matches D-state tasks whose wait channel or command contains `amdxdna`. A
single/transient sample does nothing. If the condition persists for about 60
seconds, it logs the task snapshot and requests a normal systemd reboot. If
shutdown then hangs, `RebootWatchdogSec=2min` forces a hardware reset.

Final state:

```text
service enabled and active
watchdog active
PM-fix SHA matches
amdxdna use count 0
D-state none
```

## Proven scope and remaining boundary

Proven:

- the observed runtime-PM path no longer wedged across two 30-frame runs;
- repeated actual autosuspend/resume works;
- PM-fix and watchdog survive reboot;
- PID1 owns and feeds the hardware watchdog;
- monitor policy is installed, enabled, healthy, and detects no false state on
  the healthy system.

Not deliberately fault-injected:

- a new real amdxdna D-state followed by automatic monitor reboot;
- forced expiration of the hardware watchdog during a simulated shutdown hang.

Those tests would intentionally risk a hard reset and filesystem damage. The
deployed stack is therefore a validated layered mitigation, not proof that all
firmware deadlocks are recoverable in place.

## Rollback

Prepared scripts:

```text
scripts/rollback_dstate_monitor.sh
scripts/rollback_watchdog_modules_load_override.sh
scripts/rollback_hardware_watchdog.sh
scripts/rollback_pmfix_persistent.sh
```

Rollback changes the on-disk/boot configuration; it does not unload a module
from a D-state boot.
