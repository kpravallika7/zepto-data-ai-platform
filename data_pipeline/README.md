# Zepto Data & AI Platform

## Module 1 — Data Pipeline

This project implements Module 1 of the Zepto Data & AI Platform capstone project.

The pipeline follows the required flow:

**Scrape → Clean → Convert → Store → Query**

---

## 1. Project Objective

The objective of Module 1 is to automatically collect book data from BooksToScrape, clean and transform the data, store it in a relational SQLite database, and perform SQL and pandas analysis.

The pipeline collects the following fields:

- Book title
- Price in GBP
- Star rating
- Availability
- Category

The cleaned dataset also contains a converted INR price.

---

## 2. Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL
- NumPy

---

## 3. Data Collection

Data is collected automatically using:

- `requests`
- `BeautifulSoup`

No manual copy/paste is used.

The pipeline collects books from three categories:

1. Travel
2. Mystery
3. Historical Fiction

A total of **61 books** are collected.

This satisfies the requirement of:

- At least 60 books
- At least 3 categories

---

## 4. Data Cleaning

The raw scraped data is transformed into correctly typed fields.

### Price

Example:

```text
£45.17 → 45.17
```

The currency symbol (and any stray encoding artifacts from the site's HTML) are stripped, and the remaining text is converted to a numeric `price_gbp` column with `pd.to_numeric(..., errors="coerce")`.

### Star Rating

The site encodes each book's rating as a CSS class on the product card (e.g. `star-rating Three`) rather than a number. This text is mapped to an integer:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

The result is stored as a nullable integer column, `rating`.

### Availability

Availability text such as `"In stock (22 available)"` is converted into a boolean `in_stock` column by checking whether the string contains `"In stock"` (case-insensitive).

### Handling Rows That Fail to Parse

Price and rating are parsed with `errors="coerce"`, so a row with unexpected/malformed text degrades to a null value in that column instead of crashing the pipeline. On the run that produced the committed dataset, **every scraped row parsed successfully** — `books_clean.csv` contains 61 rows and 6 columns with zero nulls in any column. If a future scrape does produce unparseable rows, the intended handling is: drop the affected row when the null rate is small, since these are individual product listings rather than a systematic feature, and the dataset comfortably exceeds the 60-book minimum with a small number of rows removed.

### Currency Conversion

`price_gbp` is converted to `price_inr` using the project's required fixed baseline rate:

```text
1 GBP = 105.50 INR
```

This is a fixed, project-defined constant — not a live or historical market rate — so it requires no external lookup and no network access beyond the initial scrape.

---

## 5. Database Schema

The cleaned data is loaded into a normalized SQLite database (`data/books.db`) with two tables sharing a primary/foreign key relationship:

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL,
    price_inr   REAL,
    rating      INTEGER,
    in_stock    INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```

`books.category_id` is a foreign key referencing `categories.category_id`. The database is dropped and recreated from `books_clean.csv` on every run of `run_pipeline.py`, so it is always reproducible from scratch.

---

## 6. SQL Queries

Six queries are executed against the database, collectively covering every required clause. Each query's full output is saved to `sql_outputs/` as a CSV; a sample of each is shown below.

### 1. SELECT + WHERE — `01_select_where.csv`
```sql
SELECT title, price_inr FROM books WHERE price_inr > 5000
```
| title | price_inr |
|---|---|
| Full Moon over Noah's Ark: An Odyssey to Mount Ararat and Beyond | 5214.86 |
| See America: A Celebration of Our National Parks & Treasured Sites | 5155.78 |
| A Year in Provence (Provence #1) | 6000.84 |

### 2. ORDER BY + LIMIT — `02_order_by_limit.csv`
```sql
SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10
```
| title | price_inr |
|---|---|
| Boar Island (Anna Pigeon #19) | 6275.14 |
| A Year in Provence (Provence #1) | 6000.84 |
| The Past Never Ends | 5960.75 |

### 3. DISTINCT — `03_distinct.csv`
```sql
SELECT DISTINCT rating FROM books ORDER BY rating
```
```text
1
2
3
4
5
```

### 4. BETWEEN — `04_between.csv`
```sql
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40
```
| title | price_gbp |
|---|---|
| Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel | 36.94 |
| Under the Tuscan Sun | 37.33 |
| The Great Railway Bazaar | 30.54 |

### 5. IN — `05_in.csv`
```sql
SELECT title, rating FROM books WHERE rating IN (4, 5)
```
| title | rating |
|---|---|
| Full Moon over Noah's Ark: An Odyssey to Mount Ararat and Beyond | 4 |
| A Year in Provence (Provence #1) | 4 |
| 1,000 Places to See Before You Die | 5 |

### 6. JOIN — `06_join.csv`
```sql
SELECT b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock, c.category_name
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
ORDER BY c.category_name, b.title
```
| title | price_gbp | price_inr | rating | in_stock | category_name |
|---|---|---|---|---|---|
| A Flight of Arrows (The Pathfinders #2) | 55.53 | 5858.42 | 5 | 1 | Historical Fiction |
| A Paris Apartment | 39.01 | 4115.55 | 4 | 1 | Historical Fiction |
| Between Shades of Gray | 20.79 | 2193.34 | 5 | 1 | Historical Fiction |

---

## 7. pandas vs. SQL Verification

The JOIN query above is reproduced directly on in-memory DataFrames using `pandas.merge()` — no SQL involved:

```python
pandas_join = books_df.merge(categories_df, on="category_id", how="inner")
```

Both results are sorted identically (`category_name`, then `title`) and compared with `DataFrame.equals()`:

```text
Pandas JOIN verification:
SQL JOIN and pandas.merge() equivalent: True
```

The `pd.merge()` output is saved separately to `sql_outputs/pandas_merge_verification.csv` and is row-for-row identical to `sql_outputs/06_join.csv`, confirming the SQL JOIN and the pandas merge produce equivalent results.

---

## 8. How to Run

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd data_pipeline
python run_pipeline.py
```

This scrapes BooksToScrape live (requires internet access), cleans and converts the data, rebuilds `data/books.db` and `data/books_clean.csv` from scratch, runs all 6 SQL queries into `sql_outputs/`, and verifies the SQL JOIN against `pandas.merge()`. The script raises an error and stops if fewer than 60 books or fewer than 3 categories are collected, or if the SQL/pandas join results don't match — so a clean run to completion is itself a self-check that every acceptance criterion for this module was met.
