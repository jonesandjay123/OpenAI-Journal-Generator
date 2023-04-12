## 建立 PyTorch GPU 環境

本專案使用 PyTorch 和 CUDA 進行 GPU 加速。首先，使用 conda 創建一個名為 `pytorch_gpu` 的虛擬環境。

```bash
conda create --name pytorch_gpu python=3.8
```
#### 激活創建的虛擬環境：

```bash
conda activate pytorch_gpu
```
#### 安裝依賴項
將專案中的 requirements.txt 文件中的依賴項安裝到剛剛建立的虛擬環境中。

```bash
pip install -r requirements.txt
```
#### 更新依賴項
如果專案中有新的 imports 需要更新 requirements.txt 文件，可以使用以下命令將新的依賴項添加到文件中：

```bash
pip freeze > requirements.txt
```
#### 使用說明:
安裝虛擬環境（如上所示）。
運行專案中的腳本。
注意事項
確保使用適當的 Python 版本。
在運行專案前，請確保已安裝所有依賴項。
如果遇到依賴項問題，請更新 `requirements.txt` 文件。

請根據實際情況調整說明。希望這對您有所幫助！