# ML Retail Pipeline TODO

## Phase 1: Data Preprocessing (Per PDF)

- [x] 1. Fix src/utils.py column name ('CustomerTenure' → 'CustomerTenureDays')
- [x] 2. Repair notebooks/preprocessing.ipynb JSON errors + add full pipeline cells (verified valid)
- [x] 3. Enhance src/preprocessing.py (ColumnTransformer, PCA 95%, stratify split) - already complete
- [x] 4. Test: python src/preprocessing.py → SUCCESS (CSVs saved: processed_data.csv, X_train/test/y_train/test; models/preprocessor.pkl/pca.pkl; outliers fixed; PCA 1 comp 100%; plot running)
- [x] 5. Update README.md with usage instructions - already complete & comprehensive
- [ ] 6. Next phase: modeling (train_model.py)

Current status: Phase 1 complete.

## Phase 2: Modeling (Churn Classification)

- [ ] 1. Implement src/train_model.py (load train_test CSVs, models: LogisticRegression/RF/XGBoost, GridSearchCV/Optuna tuning, evaluate ROC-AUC/F1, save best model.pkl)
- [ ] 2. Create notebooks/modeling.ipynb mirroring train_model.py
- [ ] 3. Test: python src/train_model.py
- [ ] 4. Update TODO/README
