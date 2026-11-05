# Operations

## First-time setup (login node, about 20 minutes)

1. **Repo and push access** (least privilege: a deploy key that can write to this one repo only):
   ```bash
   ssh-keygen -t ed25519 -N "" -C autoresearch-lab-deploy -f ~/.ssh/autoresearch_lab
   cat ~/.ssh/autoresearch_lab.pub
   ```
   On GitHub: repo, then Settings > Deploy keys > Add, paste the public key, and tick "Allow write access".
   ```bash
   printf 'Host github-lab\n  HostName github.com\n  User git\n  IdentityFile ~/.ssh/autoresearch_lab\n  IdentitiesOnly yes\n' >> ~/.ssh/config
   git clone github-lab:tanmayBeginsJourney/autoresearch-lab.git ~/autoresearch-lab
   ```
2. **Bootstrap** (idempotent; rerun any step by name):
   ```bash
   ~/autoresearch-lab/scripts/bootstrap.sh
   ```
   It sets up dirs and env vars on scratch, installs researchctl and Claude Code, syncs the three
   environments, and runs a GPU validation job on a 24 GB slice.
3. **Log in to Claude** (Pro subscription): run `claude` in `~/autoresearch-lab`, trust the folder,
   type `/login`, press `c` to copy the URL, open it in any browser, and paste the code back. Exit.
4. **Optional secrets**, never pasted into chats: add lines like `export S2_API_KEY=...`
   (Semantic Scholar) or `export HF_TOKEN=...` (gated models) to `~/.config/researchctl/secrets.env`
   (already mode 600).
5. **Start the persistent session**:
   ```bash
   ~/autoresearch-lab/scripts/lab-session.sh start
   ```
   In the Claude app, open Code and choose the `plaksha-lab` session.
6. **Restart after reboots**: `crontab -e` and add
   `*/15 * * * * $HOME/autoresearch-lab/scripts/lab-session.sh ensure`.

## Daily use
- Talk to the agent from the app. It answers H200 requests and scientific choices with a question;
  everything else proceeds within the guardrails.
- Approve H200 time: when the agent shows a request id, either approve the `researchctl approve <id>`
  prompt in the app (the hook forces a confirmation) or run it yourself over ssh.
- Check on things yourself: `researchctl digest`, `researchctl ls`, `researchctl budget`.

## Resuming after a session ends
State lives in git and scratch, so nothing needs restoring. The session-start hook injects
`researchctl digest` (running jobs, uncollected experiments, project stages). Start a new session
and say "resume project <p>".

## Recovery
| Symptom | Fix |
|---|---|
| App shows the session offline | `ssh hpc`, then `scripts/lab-session.sh status`. Restart with `stop`, then `start`. |
| "not logged in" in `$LAB_ROOT/session.log` | Run `claude`, then `/login` (logins expire; Claude Code warns 3 days ahead). |
| Job failed | `researchctl status EXP`, `researchctl inspect EXP/TASK`, fix, then `researchctl retry EXP --reason ...`. |
| Scratch purged or lost | Records in git still hold configs, git shas and per-task results. Rerun from records. Back up final artifacts you care about (S3 or GitHub releases). |
| Broken venv | `researchctl env sync <name>` (`--relock` to update versions deliberately). |
| Unapproved H200 job flagged | `researchctl budget` shows it. Cancel it with `scancel <id>` yourself, then investigate how it got through. |

## Costs
- Claude: your Pro subscription. Its 5-hour and weekly usage limits are the binding constraint on
  autonomy; check `/usage`. The CLI's compact JSON, the zero-token `wait`, skills loaded on demand,
  and subagents for paper reading are all there to stretch it.
- GPU: slices by default; H200 tracked against 150 GPU-hours/month by `researchctl budget`.

## Security
- No inbound ports anywhere. The agent's outbound traffic: Anthropic API, GitHub, Hugging Face,
  PyPI, arXiv, Semantic Scholar.
- Secrets live only in `~/.config/researchctl/secrets.env` (600) and `~/.ssh` (600). The hooks
  block reading them, and `settings.json` denies Read on them.
- GitHub access is a single-repo deploy key; revoke it in the repo settings if the cluster account
  is ever compromised.

## Laptop (RTX 4070, 8 GB)
Optional, for quick local iteration on small models: clone the repo, `uv sync --project envs/research`,
and run an entry point with a hand-written `params.json`. Results that matter must still go through
`researchctl` on the cluster so they get a record.
