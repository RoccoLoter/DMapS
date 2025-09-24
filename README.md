# DMapS
**DMapS** is **end-to-end quantum compiler** designed for **near-term distributed quantum computing architectures**, integrating qubit mapping and routing algorithms. These algorithms improve the execution efficiency of compiled quantum programs by optimizing EPR pair overhead and local SWAP gate insertions.

## 📦 Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/RoccoLoter/DMapS.git
cd DMapS
pip install -e .
```

## 🛠️ Usage
More usage tutorials can be found [here](tests/usage.ipynb).​

## 📂 Project Structure
```bash
DMapS/
├── src/                            # Core implementation
│   ├── comparion_algorithms/       # Core compiler logic
│   └── frontend/                   # Config processing implementation​
│   └── mapper/                     # Qubit mapping implementation
│   └── partitioner/                # Circuit partitioning implementation 
│   └── router/                     # Qubit routing implementation
│   └── global_config.py            # Relative path definition
├── tests/                          # Experimental files and usage tutorials​
│   ├── benchmark/                  # OpenQASM benchmark files
│   └── hardware_info/              # Reconfiguration files for quantum chip networks​
│   └── usage.ipynb                 # Usage tutorials​
```

## 📄 License
This project is licensed under the Apache 2.0.

## 📬 Contact 
- The code comments are still insufficient, and we will work on improving them.
- For questions, please contact listenwetnessluo@163.com or open an issue on GitHub.

## ✨ How to Cite the DMapS Paper
If you consider citing the DMapS paper:
```bash
@article{luo2025dmaps,
  title={DMapS: End-to-end Qubit Mapping and Routing for Distributed Quantum Computing Architectures},
  author={Luo, Tingyu and Zheng, Yuzhen and Deng, Yuxin and Fu, Xiang},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2025},
  publisher={IEEE}
}
```
