$ErrorActionPreference = "Stop"

& "$PSScriptRoot\e1_closed_set.ps1"
& "$PSScriptRoot\e2_open_set.ps1"
& "$PSScriptRoot\e3_federated_noniid.ps1"
& "$PSScriptRoot\e4_combined_open_set_noniid.ps1"
& "$PSScriptRoot\e5_ablation.ps1"
& "$PSScriptRoot\e6_efficiency_scalability.ps1"

