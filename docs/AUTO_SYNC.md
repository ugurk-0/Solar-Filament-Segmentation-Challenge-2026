# Local GitHub auto-sync

This Windows checkout has a local watcher that groups saves after 30 seconds
without changes, creates a commit, and pushes to `origin/main`. It restarts at
Windows sign-in. This setup is local to this computer, not a GitHub Action and
not installed automatically by cloning the repository.

The watcher respects `.gitignore`, including datasets, runs, upload bundles,
credentials, and caches. Non-ignored new files and deletions are included.
The local `notebook-clean` Git filter removes notebook outputs from staged
copies while preserving executed notebooks on disk. It uses
`scripts/clean_notebook.py` and the configured Python environment.
It waits while another Git operation or manually staged change is present.
Network/authentication failures are logged and retried. Rejected pushes pause
sync; remote history is never force-pushed or automatically merged.

From a PowerShell terminal at the repository root:

```powershell
# Pause before a longer edit or manual Git operation.
New-Item .git/auto-sync/paused -ItemType File -Force

# Resume after reviewing changes or resolving a rejected push.
Remove-Item -LiteralPath .git/auto-sync/paused -ErrorAction SilentlyContinue

# Inspect recent activity.
Get-Content .git/auto-sync/sync.log -Tail 20
```

The installed script is `%LOCALAPPDATA%\FilamentGitSync\Watch-Repository.ps1`.
The Windows startup entry is `FilamentGitSync` under
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.
To disable startup permanently, remove that value and pause the watcher.

Auto-sync publishes source edits without running training or tests. Kaggle
exports remain an explicit `python scripts/build_kaggle.py` operation so each
upload captures a chosen source snapshot.
