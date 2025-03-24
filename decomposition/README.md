# 📌 How to Place the Dataset

### 📂 **Datasets Location**
Datasets are available at:  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z)

---

## 📁 **Dataset Structure**
Each dataset folder consists of the following files:
- Numeric features: `N_train/N_val/N_test.npy` (or None if no numerical features exist)
- Categorical features: `C_train/C_val/C_test.npy` (or None if no categorical features exist)
- Labels: `y_train/y_val/y_test.npy`
- `info.json` which must include the following three contents (task_type can be "regression", "multiclass" or "binclass"):
```json
{
  "task_type": "regression", 
  "n_num_features": 10,
  "n_cat_features": 10
}
