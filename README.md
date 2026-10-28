# CSMAR Real-Data Extraction, 102-Feature Cleaning & Train/Validation/Test Split Pipeline (`Bertomeu et al., 2020`)

本项目针对论文 **Bertomeu, J., Cheynel, E., Floyd, E., & Pan, W. (2020), "Using machine learning to detect misstatements", Review of Accounting Studies (RAST), 25, 468–518**，构建了一套 **100% 面向真实 CSMAR 数据（绝无任何随机模拟/伪造数据）** 的特征提取、防泄露缩尾清洗与训练集/验证集/测试集切分管线。

---

## 一、核心验证结论：学校机构账号能否直接调用 CSMAR Python API？

**结论：不能直接将澳科大（MUST）CAS / EZProxy 机构账号传入 `csmarapi` 调用。**

1. **CSMAR Python API 限制 (`py_api.txt`)**：`CSMAR-PYTHON` 仅支持 Windows 系统，且 `csmar.login(account, pwd, lang)` **“仅限个人注册账号使用（用户名 / 已验证电话 / 已验证邮箱）”**，无法接收高校 CAS Ticket 或 EZProxy 浏览器 Cookie。
2. **澳科大认证中转结构 (`translink.txt` & `structure.txt`)**：`login.must.edu.mo` (`body#cas`) 登录并通过 `login.libezproxy.must.edu.mo` 跳转至 `https://global.csmar.com/`，仅授予浏览器网页端机构 IP 访问权限。

---

## 二、获取真实 CSMAR 数据的两种标准方式（已完整实现于 `src/query_csmar_table.py` 与 `src/fetch_csmar_raw_data.py`）

管线**绝不生成任何假数据**。若未配置真实 CSMAR 账号且 `data/raw/` 下无真实原表，程序会直接抛出 `FileNotFoundError` 并提示缺失的具体表名：

### 方式 A：通过 `csmarapi` 自动分页拉取真实数据（Windows 环境）
1. 打开浏览器访问 `translink.txt` 中的澳科大 EZProxy 链接，使用学号/工号登录进入 `https://global.csmar.com/`。
2. 在网页右上角注册/登录一个**个人 CSMAR 账号（手机号或邮箱）**，使其绑定澳科大机构权限。
3. 将 `CSMAR-PYTHON` 压缩包解压至本地 Windows Python 的 `Lib\site-packages\csmarapi`。
4. 在 `config.json` 的 `csmar_api.account` 和 `csmar_api.password` 中填入该个人账号密码。
5. 运行管线时，`src/query_csmar_table.py` 会自动调用 `csmar.queryCount()` 与 `csmar.query_df()`（含超过 200,000 条自动 `limit offset,200000` 分页拉取，严格遵循 `py_api.txt` 第 2.7 节），自动拉取以下 7 张核心表并缓存至 `data/raw/`：
   - `FS_Combas`（资产负债表：`A001000000` 总资产、`A001100000` 流动资产、`A001101000` 货币资金、`A001111000` 应收账款、`A001123000` 存货、`A001212000` 固定资产等）
   - `FS_Comins`（利润表：`B001101000` 营业收入、`B001201000` 营业成本、`B002000000` 净利润、`B002100000` 所得税费用）
   - `FS_Comscfd`（现金流量表：`C001000000` 经营现金流净额、`C002006000` 资本支出、`C003000000` 筹资现金流净额）
   - `TRD_Year`（年个股回报率表：`Ysmvosd` 年末市值、`Yretwd` 考虑现金红利再投资年回报率、`Yclsprc` 年末收盘价）
   - `FIN_Audit`（审计意见与费用表：`Audittyp` 审计意见类型、`Domafee` 境内审计费用、`Ndomafee` 非审计/境外费用、`Auditor` 会计师事务所）
   - `CG_Ybasic`（公司治理基本情况表：`Indcd` 行业代码、`Listdt` 上市日期、`Bdnum` 董事会规模、`Bdmting` 董事会会议次数、`Empnum` 员工人数）
   - `FIN_Restatement`（财务重述表：`Accper` 差错会计期间、`Anndt` 重述公告日期、`RestateType` 重述类型）

### 方式 B：直接读取从澳科大 EZProxy 网页端导出的本地真实 CSV / Excel 原表（跨平台通用，无需 `csmarapi`）
1. 通过 `translink.txt` 登录 CSMAR 网页端，将上述数据表导出为 `.csv` 或 `.xlsx`，放入 `data/raw/` 目录（文件名对应 `config.json` 中的 `local_file`，如 `data/raw/FS_Combas.csv` 等；或者直接放合并好的 `data/raw/merged_raw_panel.csv`）。
2. 运行 `bash run_pipeline.sh`，代码会自动合并原表、计算全部 102 个特征并切分数据集。

---

## 三、纯净输出 CSV 结构与论文 Figure 1 时间切分

每个输出 CSV 仅包含模型训练所需的 **105 列（2 列主键 `Stkcd, fyear` + 1 列因变量 `misstatement` + 102 列论文预测变量 `X`）**。在您尚未拉取真实数据前，`data/` 目录下的 CSV 仅包含这 **105 列表头（0 行假数据）**：

- `data/train_sample_2001_2009.csv`：**2001–2009 训练集**（用于训练候选超参数模型，且连续变量 1%/99% 缩尾边界严格仅在该训练集上拟合以防数据泄露）
- `data/validation_sample_2010_2011.csv`：**2010–2011 验证集**（用于验证集超参数优选）
- `data/train_val_combined_2001_2011.csv`：**2001–2011 训练+验证合并集**（用于最优超参数下重估最终预测模型）
- `data/test_sample_2012_2014.csv`：**2012–2014 测试集**（用于样本外测试评估）
- `data/cleaned_misstatement_features.csv`：**2001–2014 全样本清洗汇总表**

---

## 四、单函数模块化清单（共 13 个文件，每个 `.py` 严格仅含 1 个函数）

1. `src/load_config.py` -> `load_config(config_path)`
2. `src/verify_csmar_api.py` -> `verify_csmar_api(config)`
3. `src/verify_school_cas_login.py` -> `verify_school_cas_login(config)`
4. `src/query_csmar_table.py` -> `query_csmar_table(table_spec, csmar_service, start_date, end_date, page_size)`
5. `src/fetch_csmar_raw_data.py` -> `fetch_csmar_raw_data(config, api_verification)`
6. `src/extract_accounting_features.py` -> `extract_accounting_features(df)`
7. `src/extract_audit_governance_features.py` -> `extract_audit_governance_features(df)`
8. `src/extract_market_business_features.py` -> `extract_market_business_features(df)`
9. `src/clean_sample_selection.py` -> `clean_sample_selection(raw_df, start_year, end_year)`
10. `src/clean_and_winsorize_features.py` -> `clean_and_winsorize_features(df, config)`
11. `src/export_cleaned_csv.py` -> `export_cleaned_csv(df, output_path)`
12. `src/split_train_val_test.py` -> `split_train_val_test(cleaned_df, config)`
13. `src/run_pipeline.py` -> `run_pipeline(config_path)`
