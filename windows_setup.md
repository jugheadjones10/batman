# WSL2 Mirrored Networking + SSH Setup Notes (What Changed)

This document lists the concrete configuration changes made on the **Windows host** and inside **Ubuntu (WSL2)** to enable inbound SSH (so VS Code Remote-SSH can connect). It’s written so you can later remember what you did and revert it cleanly.

---

## 1) Windows Host Changes

### 1.1 Enabled WSL2 mirrored networking

**File created/edited:** `%UserProfile%\.wslconfig`

**Added:**

```ini
[wsl2]
networkingMode=mirrored
```

**Why:** WSL2 normally sits behind NAT. Mirrored mode makes WSL networking behave more like it is directly reachable on the same network as the Windows host.

**Applied by restarting WSL:**

```powershell
wsl --shutdown
```

**Revert:** remove those lines (or delete the file), then:

```powershell
wsl --shutdown
```

---

### 1.2 Changed Hyper-V / WSL firewall inbound behavior (Admin PowerShell)

**Command run (requires Admin PowerShell):**

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

**Why:** Even if `sshd` is listening inside WSL, Windows can still block inbound traffic at the Hyper-V/WSL boundary. This relaxes that boundary so inbound connections can reach WSL services.

**Revert (strict/safer):**

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Block
```

> Note: if you want to restore the _exact_ prior value, you needed to record it beforehand. `Block` is the safest revert if you didn’t.

---

## 2) Ubuntu (WSL2) Changes

### 2.1 Enabled systemd in WSL

**File created/edited:** `/etc/wsl.conf`

**Added:**

```ini
[boot]
systemd=true
```

**Why:** Without systemd, services don’t behave like normal Ubuntu. With systemd enabled, `systemctl enable --now ssh` makes `sshd` start reliably.

**Applied by restarting WSL (from Windows):**

```powershell
wsl --shutdown
```

**Revert:** remove those lines (or delete `/etc/wsl.conf`), then:

```powershell
wsl --shutdown
```

---

### 2.2 Installed and enabled the SSH server (openssh-server)

**Commands run:**

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

**Why:** VS Code Remote-SSH requires an SSH server on the remote side. `openssh-server` provides `sshd`.

**Verify `sshd` is running:**

```bash
sudo systemctl status ssh
```

**Verify it is listening on port 22:**

```bash
sudo ss -tulpn | grep ':22'
```

**Revert (disable service):**

```bash
sudo systemctl disable --now ssh
```

**Revert (uninstall):**

```bash
sudo apt remove --purge -y openssh-server
sudo apt autoremove -y
```

---

### 2.3 Added SSH public key for key-based login

**File modified (for the WSL user you SSH as):** `~/.ssh/authorized_keys`

**What changed:** your Mac’s full public key line (starts with `ssh-ed25519` or similar) was appended to `~/.ssh/authorized_keys`.

**Why:** Allows passwordless SSH (and VS Code Remote-SSH is much smoother).

**Revert:** remove that specific key line from:

```bash
~/.ssh/authorized_keys
```

---

## 3) Quick Current-State Checks

### 3.1 Check mirrored networking config exists (Windows)

```powershell
Get-Content $env:USERPROFILE\.wslconfig
```

### 3.2 Check ssh server status and listening port (WSL)

```bash
sudo systemctl status ssh
sudo ss -tulpn | grep ':22'
```

### 3.3 Check Hyper-V firewall setting (Windows)

```powershell
Get-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
```

---

## 4) Notes / Common Pitfalls

- **Windows SSH on port 22 can conflict** with WSL SSH on port 22.
  Check on Windows:

  ```powershell
  netstat -ano | findstr ":22"
  ```

  If Windows is listening on 22 and you want WSL to be reachable on 22, disable the Windows `sshd` service.

- **Ping (ICMP) failing doesn’t prove SSH is blocked.** SSH only needs TCP connectivity to port 22.

- If you are connecting over **Tailscale**, prefer testing SSH directly:

  ```bash
  nc -vz <tailscale-ip> 22
  ssh <wsl_user>@<tailscale-ip>
  ```
