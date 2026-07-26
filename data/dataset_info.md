# Dataset Information

## Source details
- **Dataset Name:** Bank Customer Segmentation (1 Million+ Transactions)
- **Author:** Shivam Bansal (Kaggle)
- **URL:** [https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation)
- **Local Path:** `data/customers.csv`

## Dataset Schema & Field Definitions

The dataset contains transaction-level information from an Indian bank. Below are the definitions of the columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **TransactionID** | Object / String | A unique alphanumeric identifier for each transaction. |
| **CustomerID** | Object / String | A unique alphanumeric identifier for each customer. A customer can have multiple transactions. |
| **CustomerDOB** | Object / String | The customer's Date of Birth (originally in DD/MM/YY format). |
| **CustGender** | Object / String | Gender of the customer (typically 'M' or 'F'). |
| **CustLocation** | Object / String | City or location where the customer's account or transaction is registered. |
| **CustAccountBalance** | Float | The account balance at the time of the transaction. |
| **TransactionDate** | Object / String | The date of the transaction (originally in DD/MM/YY format). |
| **TransactionTime** | Integer / Float | The time of the transaction (originally formatted as HHMMSS). |
| **TransactionAmount (INR)** | Float | The value/amount of the transaction in Indian Rupees (INR). |

## Download Instructions
Since Kaggle API credentials are not configured on this machine, follow these steps to download the dataset manually:
1. Log in to Kaggle and visit [https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation).
2. Click the **Download** button to download the `archive.zip` file (approx. 33.6 MB).
3. Extract `archive.zip` to obtain `bank_transactions.csv` (approx. 125 MB).
4. Create the `data/` directory in the project root if it doesn't exist, move `bank_transactions.csv` into it, and rename it to `customers.csv` (so it resides at `data/customers.csv`).
