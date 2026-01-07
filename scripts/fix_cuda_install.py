import json
import os

nb_path = 'notebooks/candle_svd_test.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "Installing CUDA 12.4" in source or "Remove existing CUDA" in source or "Removing existing CUDA" in source or "Attempting to remove" in source:
                new_source = [
                    "%%bash\n",
                    "# Don't exit on error immediately during cleanup\n",
                    "set +e\n",
                    "\n",
                    "echo \"🧹 Attempting to remove existing CUDA (errors ignored)...\"\n",
                    "apt-get --purge remove \"*cuda*\" \"*cublas*\" \"*cufft*\" \"*cufile*\" \"*curand*\" \"*cusolver*\" \"*cusparse*\" \"*gds-tools*\" \"*npp*\" \"*nvjpeg*\" \"nsight*\" -y --allow-change-held-packages || true\n",
                    "apt-get autoremove -y || true\n",
                    "rm -rf /usr/local/cuda* || true\n",
                    "\n",
                    "# Re-enable exit on error for installation\n",
                    "set -e\n",
                    "\n",
                    "echo \"📦 Setting up NVIDIA repo...\"\n",
                    "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb\n",
                    "dpkg -i cuda-keyring_1.1-1_all.deb\n",
                    "apt-get update\n",
                    "\n",
                    "echo \"⬇️ Installing cuda-toolkit-12-4...\"\n",
                    "# Force reinstall to ensure files exist even if dpkg implies installed\n",
                    "apt-get install -y --reinstall cuda-toolkit-12-4\n",
                    "\n",
                    "echo \"🔗 Configuring environment...\"\n",
                    "ln -sfn /usr/local/cuda-12.4 /usr/local/cuda\n",
                    "export PATH=/usr/local/cuda-12.4/bin:$PATH\n",
                    "export CUDA_HOME=/usr/local/cuda-12.4\n",
                    "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH\n",
                    "\n",
                    "echo \"✅ Verifying installation...\"\n",
                    "/usr/local/cuda-12.4/bin/nvcc --version"
                ]
                cell['source'] = new_source
                found = True
                print("✅ Fixed CUDA installation cell to force reinstall")
                break

    if not found:
        print("❌ Could not find the CUDA installation cell")
    else:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            
except Exception as e:
    print(f"Error: {e}")
