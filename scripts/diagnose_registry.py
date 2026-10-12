#!/usr/bin/env python3
"""Compare host HTTPS and Docker registry access; run in a normal Mac terminal."""
import datetime
import os
from pathlib import Path
import re
import subprocess


def redact(text):
    return re.sub(r"(https?://)[^\s/]*@", r"\1[credentials-redacted]@", text)


def run(args, timeout=35):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Timed out"
    except OSError as exc:
        return 127, "", str(exc)


def main():
    lines = ["Registry diagnostics: " + datetime.datetime.now(datetime.timezone.utc).isoformat()]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        lines.append(f"{name}: {'configured' if os.environ.get(name) else 'unset'}")
    code, stdout, stderr = run(["scutil", "--proxy"])
    lines.append(f"\nSystem proxy check, exit={code}")
    # Record enabled modes and ports only, without proxy/PAC URLs or credentials.
    lines.extend(line.strip() for line in stdout.splitlines()
                 if re.search(r"(?:Enable|Port)\s*:", line))
    if code:
        lines.append(redact(stderr[:1000]))

    for path in ("/v2/", "/v2/nvidia/tritonserver/manifests/23.12-py3"):
        code, stdout, stderr = run([
            "curl", "-sS", "--connect-timeout", "10", "--max-time", "25",
            "-D", "-", "-o", "/dev/null", "https://nvcr.io" + path])
        lines.append(f"\nHost HTTPS {path}, exit={code}")
        lines.extend(redact(line.strip()) for line in stdout.splitlines()
                     if re.match(r"^(HTTP/|server:|www-authenticate:|content-type:)", line, re.I))
        if stderr:
            lines.append(redact(stderr[:1500]))

    code, stdout, stderr = run([
        "docker", "info", "--format",
        "HTTP proxy: {{.HTTPProxy}}\nHTTPS proxy: {{.HTTPSProxy}}"])
    lines.append(f"\nDocker daemon/proxy check, exit={code}")
    lines.append(redact((stdout + stderr)[:2000]))

    for tag in ("23.12-py3", "23.12-py3-sdk"):
        image = "nvcr.io/nvidia/tritonserver:" + tag
        code, stdout, stderr = run(["docker", "manifest", "inspect", image], timeout=60)
        lines.append(f"\nDocker CLI manifest {image}, exit={code}")
        lines.append("Manifest accessible" if code == 0 else redact((stderr or stdout)[:2500]))

    project = Path(__file__).resolve().parent.parent
    directory = project / "results"
    directory.mkdir(exist_ok=True)
    output = directory / "registry_diagnostics.txt"
    report = "\n".join(lines) + "\n"
    output.write_text(report)
    print(report, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
