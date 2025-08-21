# Docker Security Policy for Physics Assistant Platform
package trivy

import rego.v1

# Deny running containers as root
deny[res] {
    input.User == "root"
    res := {
        "msg": "Container should not run as root user",
        "severity": "HIGH",
        "policy": "CIS-DI-0001"
    }
}

# Deny containers without health checks
deny[res] {
    not input.Config.Healthcheck
    res := {
        "msg": "Container should have health check configured",
        "severity": "MEDIUM",
        "policy": "CIS-DI-0002"
    }
}

# Deny containers with privileged mode
deny[res] {
    input.HostConfig.Privileged == true
    res := {
        "msg": "Container should not run in privileged mode",
        "severity": "CRITICAL",
        "policy": "CIS-DI-0003"
    }
}

# Deny containers mounting sensitive host paths
deny[res] {
    mount := input.HostConfig.Binds[_]
    sensitive_paths := ["/", "/boot", "/dev", "/etc", "/lib", "/proc", "/sys", "/usr"]
    startswith(mount, sensitive_paths[_])
    res := {
        "msg": sprintf("Container should not mount sensitive host path: %v", [mount]),
        "severity": "HIGH",
        "policy": "CIS-DI-0004"
    }
}

# Deny containers without resource limits
deny[res] {
    not input.HostConfig.Memory
    not input.HostConfig.CpuShares
    res := {
        "msg": "Container should have resource limits configured",
        "severity": "MEDIUM",
        "policy": "CIS-DI-0005"
    }
}

# Deny containers with host networking
deny[res] {
    input.HostConfig.NetworkMode == "host"
    res := {
        "msg": "Container should not use host networking",
        "severity": "HIGH",
        "policy": "CIS-DI-0006"
    }
}

# Deny containers with host PID namespace
deny[res] {
    input.HostConfig.PidMode == "host"
    res := {
        "msg": "Container should not share host PID namespace",
        "severity": "HIGH",
        "policy": "CIS-DI-0007"
    }
}

# Deny containers with host IPC namespace
deny[res] {
    input.HostConfig.IpcMode == "host"
    res := {
        "msg": "Container should not share host IPC namespace",
        "severity": "HIGH",
        "policy": "CIS-DI-0008"
    }
}

# Deny containers without restart policy
deny[res] {
    not input.HostConfig.RestartPolicy.Name
    res := {
        "msg": "Container should have restart policy configured",
        "severity": "LOW",
        "policy": "CIS-DI-0009"
    }
}

# Deny containers with unnecessary capabilities
deny[res] {
    cap := input.HostConfig.CapAdd[_]
    unnecessary_caps := ["SYS_ADMIN", "NET_ADMIN", "SYS_TIME", "SYS_MODULE"]
    cap == unnecessary_caps[_]
    res := {
        "msg": sprintf("Container should not have unnecessary capability: %v", [cap]),
        "severity": "HIGH",
        "policy": "CIS-DI-0010"
    }
}

# Deny containers without security options
deny[res] {
    count(input.HostConfig.SecurityOpt) == 0
    res := {
        "msg": "Container should have security options configured",
        "severity": "MEDIUM",
        "policy": "CIS-DI-0011"
    }
}