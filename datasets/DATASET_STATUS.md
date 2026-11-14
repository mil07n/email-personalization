# Dataset Verification and Schema Report

All five datasets have been downloaded, verified, and placed in separate directories.

| Dataset | Status | Source | Records | File Size | Key Columns |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Enron Email Dataset (AESLC Corpus)** | ✅ SUCCESS | https://www.cs.cmu.edu/~enron/ / AESLC | **16,746** | 28.09 MB | `email_id`, `body`, `subject_line`, `split`... |
| **Customer Care Emails** | ✅ SUCCESS | https://www.kaggle.com/datasets/rtweera/customer-care-emails / Hugging Face rtweera/customer_care_emails | **2,259** | 1.44 MB | `subject`, `sender`, `receiver`, `timestamp`... |
| **Email Thread Summary** | ✅ SUCCESS | https://huggingface.co/datasets/sidhq/email-thread-summary | **4,167** | 45.31 MB | `thread`, `summary` |
| **EPA Dataset (Enron People Assignment for Task Intent)** | ✅ SUCCESS | https://github.com/RevRameshkumar/EPADataset (EMNLP 2018 W-NUT / Microsoft Research) | **6,734** | 16.33 MB | `Key`, `Json` |
| **EmailSum Dataset** | ✅ SUCCESS | https://github.com/ZhangShiyue/EmailSum (ACL 2021) | **2,549** | 1.43 MB | `thread_id`, `num_of_emails`, `short_summary`, `long_summary`... |
